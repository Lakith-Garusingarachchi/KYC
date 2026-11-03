"""
OpenAICompatibleBase — the shared engine for OpenAI-compatible providers.

DeepSeek, Qwen, NVIDIA, Ollama, Groq and OpenAI all speak the OpenAI
chat-completions dialect. Previously each re-implemented the same `generate()`
retry loop, payload builder and structured-output handling (4×+ copies). This
base implements that flow once; each provider becomes a thin subclass that
overrides only what's genuinely provider-specific:

    class DeepSeekAdapter(OpenAICompatibleBase):
        provider_name = "DeepSeek"
        default_endpoint = "https://api.deepseek.com/v1/chat/completions"
        capabilities = Capabilities(text=True, structured_output=True)
        def provider_body_params(self, request): ...   # thinking, reasoning_effort

New interface:  `complete()` / `complete_multimodal()` → `LLMResponse`.
Backward compat: `generate()` / `generate_structured()` return the raw provider
dict (and parsed dict), exactly as the old adapters did, so existing direct
callers keep working until they are routed through the gateway.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from .config import ModelConfig
from .contracts import (
    Capabilities,
    Cost,
    LLMRequest,
    LLMResponse,
    Media,
    MediaKind,
    Usage,
    legacy_media,
    media_from_legacy,
)
from .interfaces import LLMAdapter
from .media_normalizer import MediaNormalizer, ResolvedMedia
from .usage_extractor import UsageExtractor
from ._openai_compat import (
    RETRYABLE_STATUS,
    compute_backoff_seconds,
    extract_first_json_object,
    prepare_strict_schema,
    preview_text,
    redact_payload_for_log,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleBase(LLMAdapter):
    """Shared implementation for OpenAI chat-completions compatible providers."""

    # --- provider-specific knobs (override in subclasses) ---------------------
    default_endpoint: str = "https://api.openai.com/v1/chat/completions"
    provider_name: str = "openai_compatible"
    capabilities: Capabilities = Capabilities(text=True, structured_output=True)
    #: OpenAI strict json_schema vs the json_object mode everyone else uses.
    supports_native_json_schema: bool = False
    #: Field name for the output-token cap. Groq/newer-OpenAI use the *_completion form.
    token_limit_field: str = "max_tokens"

    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_url = self._resolve_api_url(config.endpoint_url)
        raw_timeout = config.timeout or 180
        self.effective_timeout = float(max(raw_timeout, 180))
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.effective_timeout, connect=30.0),
            trust_env=True,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0),
        )

    def _resolve_api_url(self, custom_endpoint: Optional[str]) -> str:
        if not custom_endpoint or not custom_endpoint.strip():
            return self.default_endpoint
        url = custom_endpoint.strip()
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        if url.rstrip("/").endswith("/v1"):
            url = f"{url.rstrip('/')}/chat/completions"
        return url

    async def close(self) -> None:
        try:
            await self.client.aclose()
        except Exception:  # pragma: no cover - best effort
            pass

    # ---------------------------------------------------------------- hooks ---
    def resolve_model_id(self) -> str:
        """API model id. model_version is authoritative; model_name is display."""
        return (self.config.model_version or self.config.model_name or "").strip()

    def build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def provider_body_params(self, request: LLMRequest) -> Dict[str, Any]:
        """Provider-specific top-level body fields (thinking, reasoning_effort…)."""
        return {}

    def apply_token_limit(self, payload: Dict[str, Any], limit: Optional[int]) -> None:
        if limit is not None:
            payload[self.token_limit_field] = limit

    def apply_structured_output(self, payload: Dict[str, Any], request: LLMRequest) -> None:
        """Turn on JSON output. Override for providers with strict schema mode."""
        if request.wants_structured:
            payload["response_format"] = {"type": "json_object"}

    # ------------------------------------------------------------- messages ---
    def _base_messages(self, prompt: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def build_image_content_part(self, resolved: ResolvedMedia) -> Dict[str, Any]:
        """OpenAI-compatible image part. Providers may override the shape."""
        url = resolved.url if resolved.is_remote else resolved.data_url
        image_url: Dict[str, Any] = {"url": url}
        # Fidelity hint ('auto' | 'low' | 'high') from the model config. Only sent
        # when configured, so providers that reject the field aren't affected.
        detail = getattr(self.config, "image_detail", None)
        if detail:
            image_url["detail"] = detail
        return {"type": "image_url", "image_url": image_url}

    def build_document_content_part(self, resolved: ResolvedMedia) -> Dict[str, Any]:
        """OpenAI-compatible file part (PDF). Providers without doc support won't call this."""
        if resolved.is_remote:
            return {"type": "file", "file": {"filename": resolved.filename, "file_url": resolved.url}}
        # OpenAI expects file_data as a full data URL (data:application/pdf;base64,...),
        # not bare base64.
        return {"type": "file", "file": {"filename": resolved.filename, "file_data": resolved.data_url}}

    def _multimodal_messages(self, prompt: str, media: List[Media]) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for m in media:
            resolved = MediaNormalizer.resolve(m)
            mime = (resolved.mime_type or "").lower()
            fname = (resolved.filename or "").lower()
            is_text = (
                mime.startswith("text/")
                or mime in (
                    "application/json", "application/xml", "application/javascript",
                    "application/csv", "application/x-yaml", "application/yaml",
                    "application/sql", "application/x-sh",
                )
                or any(fname.endswith(ext) for ext in (
                    '.txt', '.csv', '.tsv', '.json', '.xml', '.md', '.py', '.js', '.ts',
                    '.html', '.htm', '.css', '.yaml', '.yml', '.sql', '.log', '.sh',
                    '.env', '.ini', '.conf', '.toml', '.rtf', '.c', '.cpp', '.h',
                    '.java', '.go', '.rs', '.bat', '.ps1'
                ))
            )
            if is_text:
                try:
                    text_content = resolved.read_bytes().decode("utf-8", errors="replace")
                    parts.append({
                        "type": "text",
                        "text": f"\n\n--- Attached Document: {resolved.filename or 'file'} ---\n{text_content}",
                    })
                    continue
                except Exception:
                    pass

            if resolved.kind == MediaKind.DOCUMENT:
                parts.append(self.build_document_content_part(resolved))
            else:
                parts.append(self.build_image_content_part(resolved))
        messages: List[Dict[str, Any]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": parts})
        return messages

    # -------------------------------------------------------------- payload ---
    def _build_payload(self, request: LLMRequest, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.resolve_model_id(), "messages": messages}

        params = request.params or {}
        limit = params.get("max_tokens", params.get("max_completion_tokens"))
        if limit is None:
            limit = self.config.max_tokens
        self.apply_token_limit(payload, limit)

        temperature = params.get("temperature")
        if temperature is not None:
            payload["temperature"] = temperature
        top_p = params.get("top_p")
        if top_p is not None:
            payload["top_p"] = top_p

        payload.update(self.provider_body_params(request))
        self.apply_structured_output(payload, request)

        # Confirmation log so reasoning/detail can be verified as actually sent
        # (visible in Dozzle). detail is read back from the built payload so it
        # reflects the real value, whatever the provider/override chose.
        logger.info(
            "[_build_payload] model=%s reasoning_effort=%s detail=%s",
            payload.get("model"),
            payload.get("reasoning_effort"),
            self._payload_image_detail(payload),
        )
        return payload

    @staticmethod
    def _payload_image_detail(payload: Dict[str, Any]) -> Optional[str]:
        """The `detail` on the first image part in the built payload, if any."""
        for message in payload.get("messages", []):
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return (part.get("image_url") or {}).get("detail")
        return None

    # ------------------------------------------------------------- transport --
    async def _post_with_retries(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        max_retries = self.config.max_retries or 3
        headers = self.build_headers()
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                logger.info(
                    "%s request model=%s (attempt %s/%s)",
                    self.provider_name, self.resolve_model_id(), attempt + 1, max_retries,
                )
                logger.debug("Payload: %s", redact_payload_for_log(payload))
                response = await self.client.post(self.api_url, json=payload, headers=headers)
                logger.debug("%s status=%s body=%s", self.provider_name,
                             response.status_code, preview_text(response.text, 1500))
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                logger.warning("%s error %s: %s (attempt %s/%s)", self.provider_name,
                               status, preview_text(e.response.text, 400), attempt + 1, max_retries)
                if status not in RETRYABLE_STATUS:
                    raise
                if attempt < max_retries - 1:
                    await asyncio.sleep(compute_backoff_seconds(attempt, e.response.headers.get("Retry-After")))
                    continue
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning("%s request timeout (%s): server did not respond within %ss (attempt %s/%s)",
                               self.provider_name, type(e).__name__, getattr(self, "effective_timeout", self.config.timeout), attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    await asyncio.sleep(compute_backoff_seconds(attempt))
                    continue
            except httpx.RequestError as e:
                last_error = e
                # The exception class matters: httpx.ReadTimeout stringifies to ""
                # so "connection error: " alone told you nothing about the cause.
                logger.warning("%s connection error: %s: %s (attempt %s/%s)", self.provider_name,
                               type(e).__name__, str(e) or "<no message>", attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    await asyncio.sleep(compute_backoff_seconds(attempt))
                    continue

        raise RuntimeError(f"{self.provider_name} API failed after {max_retries} attempts") from last_error

    # --------------------------------------------------------------- parsing --
    @staticmethod
    def _extract_content(raw: Dict[str, Any]) -> str:
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if choices:
            choice = choices[0] or {}
            return (choice.get("message", {}) or {}).get("content") or choice.get("text") or ""
        return ""

    @staticmethod
    def _extract_finish_reason(raw: Dict[str, Any]) -> Optional[str]:
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if choices:
            return (choices[0] or {}).get("finish_reason")
        return None

    def _to_response(self, raw: Dict[str, Any], want_data: bool) -> LLMResponse:
        content = self._extract_content(raw)
        data: Optional[Dict[str, Any]] = None
        if want_data and isinstance(content, str) and content.strip():
            try:
                parsed = json.loads(content)
                data = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                data = extract_first_json_object(content)
        return LLMResponse(
            text=content if isinstance(content, str) else "",
            data=data,
            model_used=UsageExtractor.extract_model(raw, self.resolve_model_id()),
            provider=self.provider_name,
            usage=UsageExtractor.extract(raw),
            finish_reason=self._extract_finish_reason(raw),
            raw=raw,
        )

    def _guide_prompt_with_schema(self, prompt: str, schema: Optional[Dict[str, Any]]) -> str:
        """Non-strict providers need the schema in the prompt + the word 'json'."""
        if not schema:
            return prompt
        try:
            schema_text = json.dumps(prepare_strict_schema(schema), ensure_ascii=False)
        except Exception:
            schema_text = str(schema)
        return (
            f"{prompt}\n\nRespond ONLY with a valid JSON object matching this schema "
            f"(no prose, no markdown fences):\n{schema_text}"
        )

    # ------------------------------------------------------ new interface -----
    async def complete(self, request: LLMRequest) -> LLMResponse:
        prompt = request.prompt
        if request.wants_structured and not self.supports_native_json_schema:
            prompt = self._guide_prompt_with_schema(prompt, request.response_schema)
        payload = self._build_payload(request, self._base_messages(prompt))
        raw = await self._post_with_retries(payload)
        return self._to_response(raw, want_data=request.wants_structured)

    async def complete_multimodal(self, request: LLMRequest) -> LLMResponse:
        if not (self.capabilities.image or self.capabilities.document):
            raise NotImplementedError(f"{self.provider_name} does not support media input")
        prompt = request.prompt
        if request.wants_structured and not self.supports_native_json_schema:
            prompt = self._guide_prompt_with_schema(prompt, request.response_schema)
        messages = self._multimodal_messages(prompt, request.media)
        payload = self._build_payload(request, messages)
        raw = await self._post_with_retries(payload)
        return self._to_response(raw, want_data=request.wants_structured)

    # ------------------------------------------------ backward-compat shims ---
    def _legacy_request(self, prompt: str, image_url: Optional[str], kwargs: Dict[str, Any]) -> LLMRequest:
        schema = kwargs.pop("response_schema", None)
        schema_name = kwargs.pop("response_schema_name", None)
        file_meta = kwargs.pop("file", None)
        mime = kwargs.pop("mime_type", None)
        media = legacy_media(image_url, file_meta, mime)
        return LLMRequest(
            prompt=prompt, config=self.config, media=media,
            response_schema=schema, response_schema_name=schema_name, params=dict(kwargs),
        )

    async def generate(self, prompt: str, image_url: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Legacy: returns the raw provider dict (unchanged contract)."""
        request = self._legacy_request(prompt, image_url, kwargs)
        resp = await (self.complete_multimodal(request) if request.media else self.complete(request))
        self.last_response = resp
        return resp.raw

    async def generate_structured(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        response_schema_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Legacy: returns the parsed dict (or raw when parsing fails)."""
        kwargs["response_schema"] = response_schema
        kwargs["response_schema_name"] = response_schema_name
        request = self._legacy_request(prompt, image_url, kwargs)
        resp = await (self.complete_multimodal(request) if request.media else self.complete(request))
        # The parsed dict drops `usage`; keep the full response for usage tracking.
        self.last_response = resp
        if resp.data is not None:
            return resp.data
        return resp.raw

    async def generate_multimodal(
        self,
        prompt: str,
        media: Optional[List[Dict[str, Any]]] = None,
        intro: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        response_schema_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Legacy multi-image/document entry point (used by VisionGateway and the
        document-comparison flow). Returns the raw provider dict. `media` is the
        legacy list-of-dicts format.
        """
        full_prompt = f"{intro}\n\n{prompt}" if intro else prompt
        request = LLMRequest(
            prompt=full_prompt,
            config=self.config,
            media=media_from_legacy(media),
            response_schema=response_schema,
            response_schema_name=response_schema_name,
            params=dict(kwargs),
        )
        resp = await (self.complete_multimodal(request) if request.media else self.complete(request))
        self.last_response = resp
        return resp.raw
