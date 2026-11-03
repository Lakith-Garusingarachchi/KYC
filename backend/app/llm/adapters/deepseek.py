"""DeepSeek adapter (deepseek-v4-pro / -flash). OpenAI-compatible, text-only."""

from __future__ import annotations

from typing import Any, Dict

from ..base_openai_compat import OpenAICompatibleBase
from ..contracts import Capabilities, LLMRequest

_VALID_EFFORT = {"low", "medium", "high"}
_TRUTHY = {"1", "true", "yes", "on", "enabled"}


class DeepSeekAdapter(OpenAICompatibleBase):
    provider_name = "DeepSeek"
    default_endpoint = "https://api.deepseek.com/v1/chat/completions"
    capabilities = Capabilities(text=True, structured_output=True)

    def provider_body_params(self, request: LLMRequest) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        effort = request.params.get("reasoning_effort") or getattr(self.config, "reasoning_effort", None)
        effort = str(effort).strip().lower() if effort else None
        if effort in _VALID_EFFORT:
            params["reasoning_effort"] = effort

        thinking = request.params.get("thinking")
        if thinking is not None:
            enabled = thinking if isinstance(thinking, bool) else str(thinking).strip().lower() in _TRUTHY
            params["thinking"] = {"type": "enabled" if enabled else "disabled"}
        elif effort in _VALID_EFFORT:
            params["thinking"] = {"type": "enabled"}
        return params
