"""
Google Gemini adapter (text + image + document).

Gemini is NOT OpenAI-compatible (it uses `contents`/`parts` with `inline_data`),
so it implements the LLMAdapter interface directly rather than subclassing the
OpenAI-compatible base. Media is normalized via the shared MediaNormalizer;
remote URLs are fetched and inlined as base64 (Gemini requires inline bytes).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from ..config import ModelConfig
from ..contracts import (
    Capabilities,
    LLMRequest,
    LLMResponse,
    Media,
    MediaKind,
    legacy_media,
    media_from_legacy,
)
from ..interfaces import LLMAdapter
from ..media_normalizer import MediaNormalizer
from ..usage_extractor import UsageExtractor
from .._openai_compat import (
    RETRYABLE_STATUS,
    compute_backoff_seconds,
    extract_first_json_object,
    preview_text,
)

logger = logging.getLogger(__name__)


class GeminiAdapter(LLMAdapter):
    provider_name = "Gemini"
    capabilities = Capabilities(text=True, image=True, document=True, structured_output=True)

    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout)

    async def close(self) -> None:
        try:
            await self.client.aclose()
        except Exception:  # pragma: no cover
            pass

    def _model_id(self) -> str:
        return (self.config.model_version or self.config.model_name or "").strip()

    def _endpoint(self) -> str:
        if self.config.endpoint_url:
            return self.config.endpoint_url
        # v1beta is the current API version — it serves the Gemini 1.5/2.x/2.5
        # models and supports responseSchema. The older /v1/ path 404s for them.
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model_id()}:generateContent?key={self.config.api_key}"
        )

    async def _auth_headers(self) -> Dict[str, str]:
        """
        Request headers. AI-Studio Gemini authenticates via the ?key= query param
        (added in _endpoint), so only Content-Type is needed here. The Vertex
        subclass overrides this to add an OAuth Bearer token.
        """
        return {"Content-Type": "application/json"}

    async def _inline_part(self, media: Media) -> Dict[str, Any]:
        resolved = MediaNormalizer.resolve(media)
        if resolved.is_remote:
            resp = await self.client.get(resolved.url)
            resp.raise_for_status()
            mime = (resp.headers.get("content-type") or resolved.mime_type).split(";", 1)[0].strip()
            data = base64.b64encode(resp.content).decode("ascii")
            return {"inline_data": {"mime_type": mime or resolved.mime_type, "data": data}}
        return {"inline_data": {"mime_type": resolved.mime_type, "data": resolved.base64_data}}

    async def _build_payload(self, request: LLMRequest) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = [{"text": request.prompt}]
        for m in request.media:
            parts.append(await self._inline_part(m))

        # "role": "user" is REQUIRED by Vertex AI generateContent (AI Studio
        # tolerates its absence); include it so both endpoints work.
        payload: Dict[str, Any] = {"contents": [{"role": "user", "parts": parts}]}

        gen_cfg: Dict[str, Any] = {}
        max_out = request.params.get("max_tokens") or self.config.max_tokens
        if max_out is not None:
            gen_cfg["maxOutputTokens"] = max_out
        temperature = request.params.get("temperature")
        if temperature is not None:
            gen_cfg["temperature"] = temperature
        if request.wants_structured:
            gen_cfg["responseMimeType"] = "application/json"
            if isinstance(request.response_schema, dict):
                gen_cfg["responseSchema"] = request.response_schema
        if gen_cfg:
            payload["generationConfig"] = gen_cfg

        if self.config.system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": self.config.system_prompt}]}
        return payload

    @staticmethod
    def _extract_text(raw: Dict[str, Any]) -> str:
        candidates = raw.get("candidates") or []
        if not candidates:
            return ""
        parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        max_retries = self.config.max_retries or 3
        url = self._endpoint()
        last_error: Optional[Exception] = None
        headers = await self._auth_headers()
        for attempt in range(max_retries):
            try:
                resp = await self.client.post(url, json=payload, headers=headers)
                logger.debug("%s status=%s body=%s", self.provider_name, resp.status_code, preview_text(resp.text, 1500))
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code not in RETRYABLE_STATUS:
                    raise
                if attempt < max_retries - 1:
                    await asyncio.sleep(compute_backoff_seconds(attempt, e.response.headers.get("Retry-After")))
            except httpx.RequestError as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(compute_backoff_seconds(attempt))
        raise RuntimeError(f"Gemini API failed after {max_retries} attempts") from last_error

    def _to_response(self, raw: Dict[str, Any], want_data: bool) -> LLMResponse:
        text = self._extract_text(raw)
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
            model_used=UsageExtractor.extract_model(raw, self._model_id()),
            provider=self.provider_name,
            usage=UsageExtractor.extract(raw),
            raw=raw,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raw = await self._post(await self._build_payload(request))
        return self._to_response(raw, want_data=request.wants_structured)

    async def complete_multimodal(self, request: LLMRequest) -> LLMResponse:
        # Same code path — Gemini treats media as extra parts on one request.
        return await self.complete(request)

    # ------------------------------------------------ backward-compat shims ---
    async def generate(self, prompt: str, image_url: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        schema = kwargs.pop("response_schema", None)
        kwargs.pop("response_schema_name", None)
        file_meta = kwargs.pop("file", None)
        mime = kwargs.pop("mime_type", None)
        media = legacy_media(image_url, file_meta, mime)
        req = LLMRequest(prompt=prompt, config=self.config, media=media, response_schema=schema, params=dict(kwargs))
        resp = await self.complete(req)
        self.last_response = resp
        return resp.raw

    async def generate_structured(self, prompt: str, image_url: Optional[str] = None,
                                  response_schema: Optional[Dict[str, Any]] = None,
                                  response_schema_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        file_meta = kwargs.pop("file", None)
        mime = kwargs.pop("mime_type", None)
        media = legacy_media(image_url, file_meta, mime)
        req = LLMRequest(prompt=prompt, config=self.config, media=media,
                         response_schema=response_schema, response_schema_name=response_schema_name, params=dict(kwargs))
        resp = await self.complete(req)
        # The parsed dict drops `usageMetadata`; keep the full response for usage tracking.
        self.last_response = resp
        return resp.data if resp.data is not None else resp.raw

    async def generate_multimodal(self, prompt: str, media: Optional[List[Dict[str, Any]]] = None,
                                  intro: Optional[str] = None, response_schema: Optional[Dict[str, Any]] = None,
                                  response_schema_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Legacy multi-image/document entry point (returns raw provider dict)."""
        full_prompt = f"{intro}\n\n{prompt}" if intro else prompt
        req = LLMRequest(prompt=full_prompt, config=self.config, media=media_from_legacy(media),
                         response_schema=response_schema, response_schema_name=response_schema_name, params=dict(kwargs))
        resp = await self.complete(req)
        self.last_response = resp
        return resp.raw
