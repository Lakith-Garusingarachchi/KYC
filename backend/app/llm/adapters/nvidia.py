"""NVIDIA NIM adapter. OpenAI-compatible catalog; model chosen via model_version."""

from __future__ import annotations

from typing import Any, Dict

from ..base_openai_compat import OpenAICompatibleBase
from ..contracts import Capabilities, LLMRequest


class NvidiaAdapter(OpenAICompatibleBase):
    provider_name = "NVIDIA"
    default_endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
    # NIM includes VLMs; image_url uses the standard OpenAI content-part shape.
    capabilities = Capabilities(text=True, image=True, structured_output=True)

    def provider_body_params(self, request: LLMRequest) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        effort = request.params.get("reasoning_effort") or getattr(self.config, "reasoning_effort", None)
        if effort:
            params["reasoning_effort"] = effort
        return params
