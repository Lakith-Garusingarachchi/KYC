"""
Consolidated LLM adapters — one file per provider.

Each provider has a single adapter that handles all its capabilities (text,
image, document) and dispatches by the request payload. The OpenAI-compatible
providers subclass `OpenAICompatibleBase`; Claude, Gemini and self-hosted
implement the `LLMAdapter` interface directly.
"""

from .openai import OpenAIAdapter
from .claude import ClaudeAdapter
from .kimi import KimiAdapter
from .deepseek import DeepSeekAdapter
from .qwen import QwenAdapter
from .nvidia import NvidiaAdapter
from .ollama import OllamaAdapter
from .groq import GroqAdapter
from .gemini import GeminiAdapter
from .gemini_vertex import GeminiVertexAdapter
from .self_hosted import SelfHostedAdapter

__all__ = [
    "OpenAIAdapter",
    "ClaudeAdapter",
    "KimiAdapter",
    "DeepSeekAdapter",
    "QwenAdapter",
    "NvidiaAdapter",
    "OllamaAdapter",
    "GroqAdapter",
    "GeminiAdapter",
    "GeminiVertexAdapter",
    "SelfHostedAdapter",
]
