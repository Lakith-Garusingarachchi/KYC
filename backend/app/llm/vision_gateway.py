"""
Vision Gateway
==============

A single, additive entry point for OpenAI vision + structured-output calls.

It wraps the existing adapter framework (`AdapterFactory` → `OpenAIVisionAdapter`)
so callers get MIME sniffing, base64 validation, image `detail`, strict
structured output, redacted logging and backoff/retry "for free" — without each
route hand-rolling the OpenAI SDK.

Phase 2 of the migration: this module is *additive*. Nothing calls it unless the
`USE_VISION_GATEWAY` feature flag is enabled (Phase 3 wires the routes behind it,
defaulting OFF). The legacy direct-SDK paths remain the default until validated.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .config import ModelConfig
from .factory import AdapterFactory
from ._openai_compat import extract_first_json_object

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def gateway_enabled() -> bool:
    """True when the USE_VISION_GATEWAY feature flag is set (default OFF)."""
    return os.getenv("USE_VISION_GATEWAY", "").strip().lower() in _TRUTHY


def run_sync(coro) -> Any:
    """
    Run an async coroutine from synchronous code.

    Safe whether or not an event loop is already running: if one is (e.g. inside
    an async FastAPI handler that calls a sync helper), the coroutine is executed
    on a short-lived worker thread with its own loop. This mirrors the blocking
    behavior of the legacy synchronous OpenAI SDK calls it replaces.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is not None and running.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


class VisionGateway:
    """Thin orchestrator over OpenAIVisionAdapter for vision/text + schema calls."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        endpoint_url: Optional[str] = None,
        timeout: int = 180,
        max_retries: int = 3,
        image_detail: str = "auto",
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        provider: str = "openai_vision",
    ) -> None:
        self._config = ModelConfig(
            model_id=model,
            model_name=model,
            model_version=model,
            model_provider=provider,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            endpoint_url=endpoint_url,
            timeout=timeout,
            max_retries=max_retries,
            image_detail=image_detail,
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort or os.getenv("VISION_REASONING_EFFORT"),
        )
        if self._config.reasoning_effort:
            logger.info("VisionGateway initialized with reasoning effort: %s", self._config.reasoning_effort)
        else:
            logger.info("VisionGateway initialized without reasoning effort")
        self._adapter = AdapterFactory.get_adapter(self._config)

    async def aclose(self) -> None:
        close = getattr(self._adapter, "close", None)
        if callable(close):
            try:
                await self._adapter.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("VisionGateway adapter close() failed; ignoring", exc_info=True)

    @staticmethod
    def _normalize_usage(raw: Any) -> Dict[str, Optional[int]]:
        usage = (raw.get("usage") if isinstance(raw, dict) else None) or {}
        return {
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
            "total_tokens": usage.get("total_tokens"),
        }

    @staticmethod
    def _extract_content(raw: Any) -> str:
        try:
            content = raw["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else ""
        except (KeyError, IndexError, TypeError):
            return ""

    async def analyze(
        self,
        prompt: str,
        media: Optional[List[Dict[str, Any]]] = None,
        intro: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        response_schema_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run one OpenAI call and return a normalized result.

        media: list of {"data"|"base64"|"url", optional "label", "mime_type"}.
               When empty/None this is a text-only call.

        Returns: {"data": parsed dict|None, "text": str, "usage": {...}, "raw": ...}
        """
        gen_kwargs: Dict[str, Any] = {}
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            gen_kwargs["temperature"] = temperature

        if media:
            raw = await self._adapter.generate_multimodal(
                prompt,
                media=media,
                intro=intro,
                response_schema=response_schema,
                response_schema_name=response_schema_name,
                **gen_kwargs,
            )
        else:
            raw = await self._adapter.generate(
                prompt,
                response_schema=response_schema,
                response_schema_name=response_schema_name,
                **gen_kwargs,
            )

        text = self._extract_content(raw)
        data: Optional[Dict[str, Any]] = None
        if isinstance(text, str) and text.strip():
            try:
                parsed = json.loads(text)
                data = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                data = extract_first_json_object(text)

        return {"data": data, "text": text, "usage": self._normalize_usage(raw), "raw": raw}
