"""
Kimi (Moonshot AI) adapter — text + vision.

OpenAI-compatible Chat Completions via the Moonshot platform
(https://platform.moonshot.ai). Covers the current model line-up:

  * kimi-k3           — flagship, vision + thinking (returns reasoning_content)
  * kimi-k2.7-code    — coding model
  * kimi-k2-thinking  — reasoning model (returns reasoning_content)
  * kimi-latest / moonshot-v1-*-vision-preview — vision chat models

Kimi specifics vs the generic base:
  * images ride the standard `image_url` content part; local uploads go inline
    as base64 data URLs (the documented Moonshot vision flow, same as the
    shared base). The OpenAI-only `detail` field is stripped — Moonshot does
    not document it and unknown fields risk request rejection.
  * thinking models return the chain-of-thought in `message.reasoning_content`;
    a preview is logged so the reasoning debug trail matches the OpenAI
    adapter's (visible in Dozzle).
  * `reasoning_effort` is forwarded only when explicitly set on the request or
    model config — Moonshot has no documented name-based default, and sending
    it to non-thinking models could make the API reject the request.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

from ..base_openai_compat import OpenAICompatibleBase
from ..contracts import Capabilities, LLMRequest, LLMResponse
from ..media_normalizer import ResolvedMedia
from .._openai_compat import preview_text

logger = logging.getLogger(__name__)

_VALID_EFFORT = {"low", "medium", "high"}
# Thinking-capable Kimi models (kimi-k3 family, any *-thinking variant).
_THINKING_MODELS = re.compile(r"(^|[-_/])(kimi-k3|kimi-k\d[\d.]*-thinking|kimi-thinking)", re.I)


class KimiAdapter(OpenAICompatibleBase):
    provider_name = "Kimi"
    default_endpoint = "https://api.moonshot.ai/v1/chat/completions"
    capabilities = Capabilities(text=True, image=True, structured_output=True)

    # ------------------------------------------------------------------ vision
    def build_image_content_part(self, resolved: ResolvedMedia) -> Dict[str, Any]:
        """Standard image_url part (base64 data URL for local media), without
        the OpenAI-only `detail` field that Moonshot does not document."""
        url = resolved.url if resolved.is_remote else resolved.data_url
        return {"type": "image_url", "image_url": {"url": url}}

    # --------------------------------------------------------------- reasoning
    def provider_body_params(self, request: LLMRequest) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        effort = (request.params or {}).get("reasoning_effort") or getattr(self.config, "reasoning_effort", None)
        effort = str(effort).strip().lower() if effort else None
        if effort in _VALID_EFFORT:
            params["reasoning_effort"] = effort
        return params

    def _to_response(self, raw: Dict[str, Any], want_data: bool) -> LLMResponse:
        self._log_reasoning_debug(raw)
        return super()._to_response(raw, want_data)

    def _log_reasoning_debug(self, raw: Dict[str, Any]) -> None:
        """Thinking models (kimi-k3, kimi-k2-thinking) return the chain of
        thought in message.reasoning_content; log a preview for debugging."""
        try:
            choices = raw.get("choices") if isinstance(raw, dict) else None
            message = ((choices or [{}])[0] or {}).get("message") or {}
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                logger.info(
                    "[Kimi reasoning_content] model=%s chars=%s preview=%s",
                    self.resolve_model_id(), len(reasoning), preview_text(reasoning, 400),
                )
            elif _THINKING_MODELS.search(self.resolve_model_id() or ""):
                logger.debug(
                    "[Kimi reasoning_content] model=%s returned no reasoning_content",
                    self.resolve_model_id(),
                )
        except Exception:  # pragma: no cover - logging must never break parsing
            pass
