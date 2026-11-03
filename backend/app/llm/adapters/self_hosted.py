"""
Self-hosted adapter (Echonlabs AI Team — OCR / duplicate detection).

Not OpenAI-compatible: it's schema-driven (request_schema / response_schema
decide the payload and how to parse the result). Ported from the original
self_hosted_adapter.py into the new interface. `complete()` returns an
LLMResponse whose `text` is the parsed result and `raw` is the model's JSON;
the legacy `generate()` returns the original {parsed, raw} shape.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from ..config import ModelConfig
from ..contracts import Capabilities, LLMRequest, LLMResponse, Media, MediaKind
from ..interfaces import LLMAdapter

logger = logging.getLogger(__name__)


class SelfHostedAdapter(LLMAdapter):
    provider_name = "Echonlabs AI Team"
    capabilities = Capabilities(text=True, image=True)

    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_url = config.endpoint_url or config.api_endpoint
        if not self.api_url:
            raise ValueError("endpoint_url or api_endpoint is required for self-hosted models")
        self.client = httpx.AsyncClient(timeout=config.timeout)
        self.headers = self._build_headers()

    async def close(self) -> None:
        try:
            await self.client.aclose()
        except Exception:  # pragma: no cover
            pass

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.authentication_type in ("api_key", "bearer") and self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    # -- schema-driven payload / parse (ported verbatim in behavior) -----------
    def _build_request_payload(self, prompt: str, image_url: Optional[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        schema_data = self.config.request_schema or self.config.request_format
        if not schema_data:
            if image_url:
                payload["image"] = image_url
            if prompt:
                payload["text"] = prompt
            return payload
        try:
            request_params = schema_data
            if isinstance(schema_data, str):
                request_params = json.loads(schema_data)
            if isinstance(request_params, dict) and "root" in request_params:
                request_params = request_params["root"]
        except (json.JSONDecodeError, TypeError):
            if image_url:
                payload["image"] = image_url
            if prompt:
                payload["text"] = prompt
            return payload

        for param in request_params:
            name = param.get("name") if isinstance(param, dict) else getattr(param, "name", None)
            value = param.get("value") if isinstance(param, dict) else getattr(param, "value", None)
            if not name:
                continue
            if name == "image" and image_url:
                payload["image"] = image_url
            elif name == "text" and prompt:
                payload["text"] = prompt
            elif name == "language":
                lower = self.config.model_name.lower()
                if "sinhala" in lower:
                    payload["language"] = "sinhala"
                elif "english" in lower:
                    payload["language"] = "english"
                elif value:
                    payload["language"] = value
            elif name == "threshold":
                payload["threshold"] = value if value is not None else 0.8
            elif value is not None:
                payload[name] = value
        return payload

    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        schema_data = self.config.response_schema or self.config.response_format
        if not schema_data:
            for field in ("text", "result", "output", "data", "content"):
                if field in response_data:
                    v = response_data[field]
                    return v if isinstance(v, str) else str(v)
            return str(response_data)
        try:
            schema = json.loads(schema_data) if isinstance(schema_data, str) else schema_data
            if isinstance(schema, dict):
                if "properties" in schema:
                    for prop in schema["properties"].keys():
                        if prop in response_data:
                            v = response_data[prop]
                            return v if isinstance(v, str) else str(v)
                elif "type" in schema:
                    for field in ("text", "result", "output", "data"):
                        if field in response_data:
                            v = response_data[field]
                            return v if isinstance(v, str) else str(v)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        return str(response_data)

    # -- interface -------------------------------------------------------------
    async def _run(self, prompt: str, image_url: Optional[str]) -> LLMResponse:
        payload = self._build_request_payload(prompt, image_url)
        response = await self.client.post(self.api_url, json=payload, headers=self.headers, timeout=self.config.timeout)
        response.raise_for_status()
        raw = response.json()
        parsed = self._parse_response(raw)
        return LLMResponse(text=parsed, provider=self.provider_name,
                           model_used=self.config.model_name, raw=raw if isinstance(raw, dict) else {"raw": raw})

    async def complete(self, request: LLMRequest) -> LLMResponse:
        image = next((m.source for m in request.media if m.kind == MediaKind.IMAGE), None)
        return await self._run(request.prompt, image)

    async def complete_multimodal(self, request: LLMRequest) -> LLMResponse:
        return await self.complete(request)

    # -- backward-compat: original returned {"parsed": ..., "raw": ...} --------
    async def generate(self, prompt: str, image_url: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        resp = await self._run(prompt, image_url)
        return {"parsed": resp.text, "raw": resp.raw}
