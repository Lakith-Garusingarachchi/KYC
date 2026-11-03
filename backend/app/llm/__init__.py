"""
LLM Module — Unified Model Gateway.

Callers should prefer the gateway + contracts, but the consolidated adapters and
the factory remain importable. The old vision-variant names (OpenAIVisionAdapter,
QwenVisionAdapter) are kept as aliases to the consolidated adapters so existing
imports keep working.

Public API:
    - ModelConfig               : adapter configuration
    - LLMAdapter                : adapter interface (base contract)
    - LLMRequest / LLMResponse  : call contracts
    - AdapterFactory            : provider string -> adapter
    - <Provider>Adapter         : consolidated per-provider adapters
"""

from .config import ModelConfig
from .interfaces import LLMAdapter
from .contracts import (
    LLMRequest,
    LLMResponse,
    Media,
    MediaKind,
    Usage,
    Cost,
    CostModel,
    Capabilities,
)
from .adapters import (
    OpenAIAdapter,
    ClaudeAdapter,
    KimiAdapter,
    GeminiAdapter,
    DeepSeekAdapter,
    QwenAdapter,
    NvidiaAdapter,
    OllamaAdapter,
    GroqAdapter,
    SelfHostedAdapter,
)
from .factory import AdapterFactory

# Backward-compatible aliases: the vision variants are now the consolidated
# per-provider adapters (which dispatch text/image/document by payload).
OpenAIVisionAdapter = OpenAIAdapter
QwenVisionAdapter = QwenAdapter

# Legacy base class kept importable for any code still referencing it.
try:  # pragma: no cover - optional legacy import
    from .base_adapter import BaseAdapter  # noqa: F401
except Exception:  # base_adapter may be removed later
    BaseAdapter = LLMAdapter  # type: ignore

__all__ = [
    "ModelConfig",
    "LLMAdapter",
    "BaseAdapter",
    "LLMRequest",
    "LLMResponse",
    "Media",
    "MediaKind",
    "Usage",
    "Cost",
    "CostModel",
    "Capabilities",
    "OpenAIAdapter",
    "OpenAIVisionAdapter",
    "ClaudeAdapter",
    "KimiAdapter",
    "GeminiAdapter",
    "DeepSeekAdapter",
    "QwenAdapter",
    "QwenVisionAdapter",
    "NvidiaAdapter",
    "OllamaAdapter",
    "GroqAdapter",
    "SelfHostedAdapter",
    "AdapterFactory",
]

__version__ = "2.0.0"
