"""
Qwen adapter (consolidated: text + image).

Merges qwen_adapter.py (text) and qwen_vision_adapter.py (Qwen-VL images).
OpenAI-compatible via Alibaba Model Studio. Images use the standard image_url
content part. Document file-extract (qwen-long) is a separate upload flow and is
not enabled here (declared capability reflects that).
"""

from __future__ import annotations

from typing import Any, Dict

from ..base_openai_compat import OpenAICompatibleBase
from ..contracts import Capabilities, LLMRequest

_TRUTHY = {"1", "true", "yes", "on", "enabled"}


class QwenAdapter(OpenAICompatibleBase):
    provider_name = "Qwen"
    default_endpoint = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    capabilities = Capabilities(text=True, image=True, structured_output=True)

    def provider_body_params(self, request: LLMRequest) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        thinking = request.params.get("enable_thinking", request.params.get("thinking"))
        if thinking is not None:
            params["enable_thinking"] = (
                thinking if isinstance(thinking, bool) else str(thinking).strip().lower() in _TRUTHY
            )
        budget = request.params.get("thinking_budget")
        if budget is not None:
            params["thinking_budget"] = budget
        return params
