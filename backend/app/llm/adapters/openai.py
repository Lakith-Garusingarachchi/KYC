"""
OpenAI adapter (consolidated: text + image + document).

Text and images use the OpenAI-compatible **Chat Completions** API via the
shared OpenAICompatibleBase (the format every other provider also speaks).

Documents (PDF) are the one OpenAI-specific exception: they route to the
**Responses API** (`/v1/responses`) with a Files-API upload (`/v1/files`) — the
approach used on `develop`. The Responses API + file_id upload handles large
PDFs far better than embedding base64 in a Chat Completions request, and it's
OpenAI-only (no other provider supports it), so it stays isolated here.

OpenAI specifics vs the generic base:
  * newer models use `max_completion_tokens` instead of `max_tokens`
  * native strict structured output (Chat: response_format json_schema;
    Responses: text.format json_schema)
  * document requests -> Responses API
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from ..base_openai_compat import OpenAICompatibleBase
from ..contracts import Capabilities, LLMRequest, LLMResponse, MediaKind
from ..media_normalizer import MediaNormalizer, ResolvedMedia
from ..usage_extractor import UsageExtractor
from .._openai_compat import (
    RETRYABLE_STATUS,
    compute_backoff_seconds,
    extract_first_json_object,
    prepare_strict_schema,
    preview_text,
    redact_payload_for_log,
)

logger = logging.getLogger(__name__)

# Models that expect max_completion_tokens (o-series, gpt-4o/4.1/5 families, etc.)
_NEW_TOKEN_FIELD = re.compile(r"(^|[-_/])(o\d|gpt-4o|gpt-4\.1|gpt-5|gpt-4-turbo|chatgpt)", re.I)
# Older models that still use max_tokens
_OLD_TOKEN_FIELD = re.compile(r"(gpt-3\.5|text-davinci|gpt-4-0|gpt-4-32k)", re.I)
# Models that accept a `reasoning_effort` control (o-series + gpt-5 family only).
# Narrower than _NEW_TOKEN_FIELD, which also matches gpt-4o — sending
# reasoning_effort to gpt-4o would make the API reject the request.
_REASONING_MODELS = re.compile(r"(^|[-_/])(o\d|gpt-5)", re.I)

# Hardcoded OpenAI defaults (restored). Applied to every OpenAI model unless the
# model config / request explicitly overrides them.
_DEFAULT_IMAGE_DETAIL = "high"
_DEFAULT_REASONING_EFFORT = "medium"

_TEXT_MIMES = {
    "application/json", "application/xml", "application/javascript",
    "application/csv", "application/x-yaml", "application/yaml",
    "application/sql", "application/x-sh",
}
_TEXT_EXTENSIONS = (
    '.txt', '.csv', '.tsv', '.json', '.xml', '.md', '.py', '.js', '.ts',
    '.html', '.htm', '.css', '.yaml', '.yml', '.sql', '.log', '.sh',
    '.env', '.ini', '.conf', '.toml', '.rtf', '.c', '.cpp', '.h',
    '.java', '.go', '.rs', '.bat', '.ps1'
)


class OpenAIAdapter(OpenAICompatibleBase):
    provider_name = "OpenAI"
    default_endpoint = "https://api.openai.com/v1/chat/completions"
    capabilities = Capabilities(text=True, image=True, document=True, structured_output=True)
    supports_native_json_schema = True

    # ---------------------------------------------------- chat-completions ----
    def _uses_max_completion_tokens(self, model_id: str) -> bool:
        if _OLD_TOKEN_FIELD.search(model_id or ""):
            return False
        return bool(_NEW_TOKEN_FIELD.search(model_id or ""))

    def apply_token_limit(self, payload: Dict[str, Any], limit: Optional[int]) -> None:
        if limit is None:
            return
        field = "max_completion_tokens" if self._uses_max_completion_tokens(self.resolve_model_id()) else "max_tokens"
        payload[field] = limit

    def apply_structured_output(self, payload: Dict[str, Any], request: LLMRequest) -> None:
        if not request.wants_structured:
            return
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request.response_schema_name or "structured_response",
                "schema": prepare_strict_schema(request.response_schema),
                "strict": True,
            },
        }

    # --- effective reasoning/detail (hardcoded defaults, config/request override) ---
    def _effective_image_detail(self) -> str:
        """Image fidelity for OpenAI: hardcoded 'high' by default. A real explicit
        choice (e.g. 'low') wins; the unset default ('auto'/blank) maps to 'high'."""
        detail = getattr(self.config, "image_detail", None)
        if not detail or detail == "auto":
            return _DEFAULT_IMAGE_DETAIL
        return detail

    def _effective_reasoning_effort(self, request: LLMRequest) -> Optional[str]:
        """reasoning_effort resolution:

        1. An explicit value (request param or model config) is trusted for ANY
           model — covers reasoning models whose name the regex can't recognise
           (custom / Azure deployment names).
        2. Otherwise, name-recognised reasoning models (o-series / gpt-5) get the
           hardcoded 'medium' default.
        3. Everything else (gpt-4o, gpt-4.1, gpt-3.5 …) gets nothing — sending it
           would make the API reject the request.
        """
        explicit = (request.params or {}).get("reasoning_effort") or getattr(self.config, "reasoning_effort", None)
        if explicit:
            return explicit
        if _REASONING_MODELS.search(self.resolve_model_id() or ""):
            return _DEFAULT_REASONING_EFFORT
        return None

    def build_image_content_part(self, resolved: ResolvedMedia) -> Dict[str, Any]:
        """OpenAI image part with the fidelity `detail` always set (default 'high')."""
        part = super().build_image_content_part(resolved)
        part["image_url"]["detail"] = self._effective_image_detail()
        return part

    def provider_body_params(self, request: LLMRequest) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        effort = self._effective_reasoning_effort(request)
        if effort:
            params["reasoning_effort"] = effort
        return params

    # ----------------------------------------------------------- routing ------
    async def complete_multimodal(self, request: LLMRequest) -> LLMResponse:
        """
        Images -> Chat Completions (shared base). Any DOCUMENT present -> the
        OpenAI Responses API (better large-file handling via Files upload).
        """
        has_document = any(m.kind == MediaKind.DOCUMENT for m in request.media)
        if not has_document:
            return await super().complete_multimodal(request)
        return await self._complete_via_responses(request)

    # ------------------------------------------------ Responses API (docs) ----
    def _responses_url(self) -> str:
        base = self.config.endpoint_url
        if base and "/chat/completions" in base:
            return base.replace("/chat/completions", "/responses")
        if base and base.rstrip("/").endswith("/responses"):
            return base
        return "https://api.openai.com/v1/responses"

    def _files_url(self) -> str:
        base = self.config.endpoint_url
        if base and "/chat/completions" in base:
            return base.replace("/chat/completions", "/files")
        if base and "/responses" in base:
            return base.replace("/responses", "/files")
        return "https://api.openai.com/v1/files"

    async def _upload_file(self, resolved: ResolvedMedia) -> str:
        """Upload PDF bytes to the Files API, return the file_id for input_file."""
        try:
            file_bytes = resolved.read_bytes()
        except Exception as err:
            raise ValueError(f"Cannot upload document: no file bytes available ({err})") from err

        headers = {"Authorization": self.build_headers().get("Authorization", "")}
        filename = resolved.filename or "document.pdf"
        # Logged because this is a second, separate HTTP round-trip that happens
        # before the Responses request — without it the logs show an unexplained
        # call to api.openai.com that the image path never makes.
        logger.info("%s Files upload: %s (%s bytes)", self.provider_name, filename, len(file_bytes))
        resp = await self.client.post(
            self._files_url(),
            headers=headers,
            data={"purpose": "user_data"},
            files={"file": (filename, file_bytes, "application/pdf")},
        )
        resp.raise_for_status()
        file_id = resp.json().get("id")
        if not isinstance(file_id, str) or not file_id:
            raise ValueError("Files API did not return a valid file id")
        logger.info("%s Files upload complete: file_id=%s", self.provider_name, file_id)
        return file_id

    @staticmethod
    def _is_text_media(mime: str, fname: str) -> bool:
        if mime.startswith("text/") or mime in _TEXT_MIMES:
            return True
        return any(fname.endswith(ext) for ext in _TEXT_EXTENSIONS)

    @staticmethod
    def _build_text_input_part(resolved: ResolvedMedia) -> Optional[Dict[str, Any]]:
        mime = (resolved.mime_type or "").lower()
        fname = (resolved.filename or "").lower()
        if not OpenAIAdapter._is_text_media(mime, fname):
            return None
        try:
            text_content = resolved.read_bytes().decode("utf-8", errors="replace")
            return {
                "type": "input_text",
                "text": f"\n\n--- Attached Document: {resolved.filename or 'file'} ---\n{text_content}",
            }
        except Exception:
            return None

    async def _build_document_input_part(self, resolved: ResolvedMedia) -> Dict[str, Any]:
        part: Dict[str, Any] = {"type": "input_file", "detail": self._effective_image_detail()}
        if resolved.file_id and resolved.file_id.startswith("file-"):
            part["file_id"] = resolved.file_id
        elif resolved.is_remote and not (resolved.url or "").startswith("file://"):
            part["file_url"] = resolved.url
        else:
            part["file_id"] = await self._upload_file(resolved)
        return part

    def _build_image_input_part(self, resolved: ResolvedMedia) -> Dict[str, Any]:
        is_direct_remote = resolved.is_remote and not (resolved.url or "").startswith("file://")
        url = resolved.url if is_direct_remote else resolved.data_url
        return {
            "type": "input_image",
            "image_url": url,
            "detail": self._effective_image_detail(),
        }

    async def _build_responses_input(self, request: LLMRequest) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = [{"type": "input_text", "text": request.prompt}]
        for m in request.media:
            resolved = MediaNormalizer.resolve(m)
            text_part = self._build_text_input_part(resolved)
            if text_part:
                parts.append(text_part)
            elif resolved.kind == MediaKind.DOCUMENT:
                parts.append(await self._build_document_input_part(resolved))
            else:
                parts.append(self._build_image_input_part(resolved))
        return [{"role": "user", "content": parts}]

    async def _complete_via_responses(self, request: LLMRequest) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.resolve_model_id(),
            "input": await self._build_responses_input(request),
        }
        if self.config.system_prompt:
            payload["instructions"] = self.config.system_prompt

        limit = (request.params or {}).get("max_tokens") or self.config.max_tokens
        if limit is not None:
            payload["max_output_tokens"] = limit
        temperature = (request.params or {}).get("temperature")
        if temperature is not None:
            payload["temperature"] = temperature
        if request.wants_structured:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.response_schema_name or "structured_response",
                    "schema": prepare_strict_schema(request.response_schema),
                    "strict": True,
                }
            }

        # Reasoning effort for the Responses API is nested (reasoning.effort),
        # unlike Chat Completions' top-level reasoning_effort. Defaults to 'medium'
        # for reasoning-capable models (o-series / gpt-5 family) only.
        model_id = self.resolve_model_id()
        effort = self._effective_reasoning_effort(request)
        if effort:
            payload["reasoning"] = {"effort": effort}

        # Confirmation log (visible in Dozzle) so reasoning/detail can be verified
        # as actually sent on the document/Responses path. detail is read back from
        # the built payload (same approach as the Chat Completions path) so the log
        # can't drift from what was actually sent.
        logger.info(
            "[_complete_via_responses] model=%s reasoning_effort=%s detail=%s",
            model_id,
            (payload.get("reasoning") or {}).get("effort"),
            self._payload_responses_detail(payload),
        )

        raw = await self._post_responses(payload)
        return self._parse_responses(raw, want_data=request.wants_structured)

    @staticmethod
    def _payload_responses_detail(payload: Dict[str, Any]) -> Optional[str]:
        """The `detail` on the first image/file part in the built Responses payload.

        Mirrors OpenAICompatibleBase._payload_image_detail for the Responses shape,
        where media parts live under input[].content[] rather than messages[].
        """
        for message in payload.get("input", []):
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in {"input_image", "input_file"}:
                        return part.get("detail")
        return None

    async def _post_responses(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._responses_url()
        headers = self.build_headers()
        max_retries = self.config.max_retries or 3
        last_error: Optional[Exception] = None
        # Logging mirrors OpenAICompatibleBase._post_with_retries line for line, so
        # the document path reads identically to the image path in Dozzle — same
        # request line, payload dump, status line and failure warnings.
        for attempt in range(max_retries):
            try:
                logger.info(
                    "%s Responses request model=%s (attempt %s/%s)",
                    self.provider_name, self.resolve_model_id(), attempt + 1, max_retries,
                )
                logger.debug("Payload: %s", redact_payload_for_log(payload))
                resp = await self.client.post(url, json=payload, headers=headers)
                logger.debug("%s Responses status=%s body=%s", self.provider_name,
                             resp.status_code, preview_text(resp.text, 1500))
                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                logger.warning("%s Responses error %s: %s (attempt %s/%s)", self.provider_name,
                               status, preview_text(e.response.text, 400), attempt + 1, max_retries)
                if status not in RETRYABLE_STATUS:
                    raise
                if attempt < max_retries - 1:
                    await asyncio.sleep(compute_backoff_seconds(attempt, e.response.headers.get("Retry-After")))
                    continue
            except httpx.RequestError as e:
                last_error = e
                logger.warning("%s Responses connection error: %s: %s (attempt %s/%s)", self.provider_name,
                               type(e).__name__, str(e) or "<no message>", attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    await asyncio.sleep(compute_backoff_seconds(attempt))
                    continue

        raise RuntimeError(f"{self.provider_name} Responses API failed after {max_retries} attempts") from last_error

    def _parse_responses(self, raw: Dict[str, Any], want_data: bool) -> LLMResponse:
        text = raw.get("output_text")
        if not isinstance(text, str) or not text:
            frags: List[str] = []
            for item in raw.get("output") or []:
                if not isinstance(item, dict):
                    continue
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                        val = part.get("text")
                        if isinstance(val, str) and val:
                            frags.append(val)
            text = "\n".join(frags).strip()

        data: Optional[Dict[str, Any]] = None
        if want_data and text.strip():
            try:
                parsed = json.loads(text)
                data = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                data = extract_first_json_object(text)

        return LLMResponse(
            text=text,
            data=data,
            model_used=UsageExtractor.extract_model(raw, self.resolve_model_id()),
            provider=self.provider_name,
            usage=UsageExtractor.extract(raw),
            raw=raw,
        )
