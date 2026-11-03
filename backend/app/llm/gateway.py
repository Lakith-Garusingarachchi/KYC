"""
LLMGateway — the central hub ("interceptor") for all LLM calls.

Every call flows through here, so it's the single place that:
  1. picks the adapter (AdapterFactory) for the model's provider,
  2. routes by the request payload — media present → complete_multimodal on a
     capable adapter; otherwise → complete,
  3. computes cost from the pricing rate table, and
  4. records one usage event (tokens + cost + user) to MongoDB — asynchronously,
     so the user's response is never delayed, and never failing the call.

Backward compatibility: `generate()` / `generate_structured()` mirror the old
adapter methods (returning the raw provider dict / parsed dict) so a call site
switches from `adapter.generate(...)` to `LLMGateway.generate(config, ...)` in
one line while gaining tracking.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from .config import ModelConfig
from .contracts import LLMRequest, LLMResponse, Media, MediaKind, media_from_legacy
from .factory import AdapterFactory

logger = logging.getLogger(__name__)

# Keep references to fire-and-forget tracking tasks so they aren't GC'd.
_background_tasks: Set[asyncio.Task] = set()


class UnsupportedCapabilityError(ValueError):
    """Raised when a request needs a capability the provider's adapter lacks."""


class LLMGateway:
    """Single entry point that routes, computes cost, and records usage."""

    # ------------------------------------------------------ core (new API) ---
    @classmethod
    async def complete(cls, request: LLMRequest) -> LLMResponse:
        adapter = AdapterFactory.get_adapter(request.config)
        try:
            needed = request.required_capability()
            if needed is not None and not adapter.capabilities.supports(needed):
                raise UnsupportedCapabilityError(
                    f"{adapter.provider_name} adapter does not support {needed.value} input "
                    f"(model '{request.config.model_name}')"
                )
            if request.media:
                resp = await adapter.complete_multimodal(request)
            else:
                resp = await adapter.complete(request)
        finally:
            await adapter.close()

        cls._attach_cost(request, resp)
        cls._schedule_tracking(request, resp)
        return resp

    # ----------------------------------------------- cost + tracking hooks ---
    @staticmethod
    def _attach_cost(request: LLMRequest, resp: LLMResponse) -> None:
        try:
            from src.pricing.pricing_store import PricingStore
            provider = resp.provider or request.config.model_provider
            model = resp.model_used or request.config.model_version or request.config.model_name
            resp.cost = PricingStore.compute_cost(provider, model, resp.usage)
        except Exception:  # cost must never break the call
            logger.debug("cost computation skipped", exc_info=True)

    @classmethod
    def _schedule_tracking(cls, request: LLMRequest, resp: LLMResponse) -> None:
        """Record usage asynchronously (fire-and-forget)."""
        try:
            task = asyncio.create_task(cls._record(request, resp))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
        except RuntimeError:
            # No running loop (rare sync context) — record inline, best-effort.
            try:
                asyncio.run(cls._record(request, resp))
            except Exception:
                logger.debug("inline usage recording failed", exc_info=True)

    @staticmethod
    async def _record(request: LLMRequest, resp: LLMResponse) -> None:
        try:
            from src.usage.usage_store import UsageStore
            await UsageStore.record_usage(
                user_id=request.user_id or "unknown",
                model_id=request.config.model_id or request.config.model_name,
                token_usage=resp.usage.as_dict(),
                raw_response=None,
                source=request.source,
                execution_id=request.execution_id,
                model_provider=resp.provider or request.config.model_provider,
                model_used=resp.model_used,
                cost=resp.cost.as_dict() if resp.cost else None,
            )
        except Exception:  # tracking must never affect the call
            logger.warning("usage recording failed", exc_info=True)

    # ---------------------------------------- backward-compatible wrappers ---
    @classmethod
    async def generate(
        cls,
        config: ModelConfig,
        prompt: str = "",
        image_url: Optional[str] = None,
        *,
        media: Optional[List[Dict[str, Any]]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        response_schema_name: Optional[str] = None,
        user_id: Optional[str] = None,
        source: str = "unknown",
        execution_id: Optional[str] = None,
        **params,
    ) -> Dict[str, Any]:
        """Legacy-style: returns the raw provider dict (and records usage)."""
        resp = await cls.complete(cls._build_request(
            config, prompt, image_url, media, response_schema, response_schema_name,
            user_id, source, execution_id, params,
        ))
        return resp.raw

    @classmethod
    async def generate_structured(
        cls,
        config: ModelConfig,
        prompt: str = "",
        image_url: Optional[str] = None,
        *,
        response_schema: Optional[Dict[str, Any]] = None,
        response_schema_name: Optional[str] = None,
        media: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        source: str = "unknown",
        execution_id: Optional[str] = None,
        **params,
    ) -> Dict[str, Any]:
        """Legacy-style: returns the parsed dict (falls back to raw)."""
        resp = await cls.complete(cls._build_request(
            config, prompt, image_url, media, response_schema, response_schema_name,
            user_id, source, execution_id, params,
        ))
        return resp.data if resp.data is not None else resp.raw

    @staticmethod
    def _build_request(config, prompt, image_url, media, response_schema,
                       response_schema_name, user_id, source, execution_id, params) -> LLMRequest:
        media_objs: List[Media] = media_from_legacy(media)
        if image_url:
            media_objs.append(Media(kind=MediaKind.IMAGE, source=image_url,
                                    mime_type=params.pop("mime_type", None)))
        return LLMRequest(
            prompt=prompt, config=config, media=media_objs,
            response_schema=response_schema, response_schema_name=response_schema_name,
            params=dict(params), user_id=user_id, source=source, execution_id=execution_id,
        )
