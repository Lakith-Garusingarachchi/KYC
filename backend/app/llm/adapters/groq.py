"""
Groq adapter (consolidated: OpenAI + Llama variants → one file).

OpenAI-compatible. Groq prefers `max_completion_tokens`. Vision-capable Llama
models use the standard image_url content part — this fixes the old
groq_adapter_openai bug that pasted the image URL into the prompt as text.
"""

from __future__ import annotations

from ..base_openai_compat import OpenAICompatibleBase
from ..contracts import Capabilities


class GroqAdapter(OpenAICompatibleBase):
    provider_name = "Groq"
    default_endpoint = "https://api.groq.com/openai/v1/chat/completions"
    capabilities = Capabilities(text=True, image=True, structured_output=True)
    token_limit_field = "max_completion_tokens"
