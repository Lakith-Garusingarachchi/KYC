"""
Adapter Factory.

Maps a model's `model_provider` string to the consolidated adapter that serves
it. The provider-string keys are unchanged (so existing model configs keep
resolving); only the target classes moved to the consolidated `adapters/`
package. Vision/variant provider strings (openai_vision, qwen_vision,
groq_openai, groq_llama) now resolve to the single per-provider adapter, which
dispatches text/image/document by the request payload.

Lookup is case-insensitive, so registry keys are stored lowercase.
"""

import logging
from typing import Type

from .interfaces import LLMAdapter
from .config import ModelConfig
from .adapters import (
    OpenAIAdapter,
    ClaudeAdapter,
    KimiAdapter,
    GeminiAdapter,
    GeminiVertexAdapter,
    DeepSeekAdapter,
    QwenAdapter,
    NvidiaAdapter,
    OllamaAdapter,
    GroqAdapter,
    SelfHostedAdapter,
)

logger = logging.getLogger(__name__)


class AdapterFactory:
    """Factory that creates the right consolidated adapter for a provider."""

    # provider string (lowercase) -> adapter class. Kept in sync with the
    # strings stored in model configs. Consolidated: variant keys map to the
    # single per-provider adapter.
    _ADAPTER_REGISTRY = {
        # OpenAI — one adapter for text + vision + document
        "openai": OpenAIAdapter,
        "openai_text": OpenAIAdapter,
        "openai_vision": OpenAIAdapter,
        "openai_image": OpenAIAdapter,
        # Anthropic Claude — one adapter for text + vision + document
        "anthropic": ClaudeAdapter,
        "claude": ClaudeAdapter,
        "anthropic_vision": ClaudeAdapter,
        "claude_vision": ClaudeAdapter,
        # Kimi / Moonshot AI — one adapter for text + vision (kimi-k3,
        # kimi-k2.7-code, kimi-k2-thinking, moonshot-v1-*)
        "kimi": KimiAdapter,
        "kimi_vision": KimiAdapter,
        "kimi_code": KimiAdapter,
        "kimi_thinking": KimiAdapter,
        "moonshot": KimiAdapter,
        "moonshotai": KimiAdapter,
        "moonshot_ai": KimiAdapter,
        # Google Gemini (AI Studio — API key)
        "google": GeminiAdapter,
        "gemini": GeminiAdapter,
        # Google Gemini on Vertex AI (service-account JSON / GCP project billing)
        "gemini-vertex-ai": GeminiVertexAdapter,
        "gemini_vertex": GeminiVertexAdapter,
        "gemini-vertex": GeminiVertexAdapter,
        "vertex": GeminiVertexAdapter,
        "vertexai": GeminiVertexAdapter,
        "vertex_ai": GeminiVertexAdapter,
        # Groq — openai + llama variants → one adapter
        "groq": GroqAdapter,
        "groq_openai": GroqAdapter,
        "groq_llama": GroqAdapter,
        # DeepSeek (deepseek-v4-pro / -flash)
        "deepseek": DeepSeekAdapter,
        # Qwen — text + vision (image) → one adapter
        "qwen": QwenAdapter,
        "qwen_vision": QwenAdapter,
        "qwen_vl": QwenAdapter,
        "qwen_image": QwenAdapter,
        "qwen_doc": QwenAdapter,
        "qwen_long": QwenAdapter,
        # NVIDIA NIM catalog; model chosen via model_version
        "nvidia": NvidiaAdapter,
        "nvidia_nim": NvidiaAdapter,
        "nim": NvidiaAdapter,
        "nemotron": NvidiaAdapter,
        # Ollama Cloud + local
        "ollama": OllamaAdapter,
        "ollama_cloud": OllamaAdapter,
        # Self-hosted (Echonlabs AI Team)
        "echonlabs ai team": SelfHostedAdapter,
        "echonlabs": SelfHostedAdapter,
        "echonlabsaiteam": SelfHostedAdapter,
        "self_hosted": SelfHostedAdapter,
    }

    @staticmethod
    def get_adapter(config: ModelConfig) -> LLMAdapter:
        """
        Create and return the appropriate adapter based on configuration.

        The returned adapter exposes both the new interface (`complete` /
        `complete_multimodal`) and the legacy `generate` / `generate_structured`
        methods, so existing direct callers keep working unchanged.

        Raises:
            ValueError: If the provider is not supported.
        """
        provider_raw = config.model_provider or ""
        provider = provider_raw.strip().lower()

        adapter_class = AdapterFactory._ADAPTER_REGISTRY.get(provider)
        if adapter_class is None:
            supported = sorted(set(AdapterFactory._ADAPTER_REGISTRY.keys()))
            error_msg = (
                f"Unsupported model provider: '{provider_raw}'. Supported: {supported}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Creating %s for provider '%s'", adapter_class.__name__, provider_raw)
        return adapter_class(config)

    @staticmethod
    def register_adapter(provider_name: str, adapter_class: Type[LLMAdapter]):
        """Register a custom adapter for a provider string (stored lowercase)."""
        if not issubclass(adapter_class, LLMAdapter):
            raise TypeError(f"Adapter class must inherit from LLMAdapter, got {adapter_class}")
        AdapterFactory._ADAPTER_REGISTRY[provider_name.strip().lower()] = adapter_class
        logger.info("Registered adapter '%s' for provider '%s'", adapter_class.__name__, provider_name)
