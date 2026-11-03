"""
Ollama adapter (Cloud + local). OpenAI-compatible; model via model_version.

Local needs no key (a dummy is sent); Cloud requires one. Vision models (e.g.
llava) are supported through the standard image_url content part.
"""

from __future__ import annotations

from typing import Any, Dict

from ..base_openai_compat import OpenAICompatibleBase
from ..contracts import Capabilities, LLMRequest


class OllamaAdapter(OpenAICompatibleBase):
    provider_name = "Ollama"
    default_endpoint = "https://ollama.com/v1/chat/completions"
    capabilities = Capabilities(text=True, image=True, structured_output=True)

    def build_headers(self) -> Dict[str, str]:
        # Local Ollama needs no key; send a dummy so the header is well-formed.
        api_key = self.config.api_key or "ollama"
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def provider_body_params(self, request: LLMRequest) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        think = request.params.get("think")
        if think is not None:
            params["think"] = bool(think)
        effort = request.params.get("reasoning_effort") or getattr(self.config, "reasoning_effort", None)
        if effort:
            params["reasoning_effort"] = effort
        extra = request.params.get("extra_body")
        if isinstance(extra, dict):
            for k, v in extra.items():
                params.setdefault(k, v)
        return params
