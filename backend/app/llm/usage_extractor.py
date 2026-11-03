"""
UsageExtractor — normalize token usage across provider response shapes.

No LLM provider returns a dollar cost; they return token counts, in slightly
different shapes. This turns any provider's raw response into a stable `Usage`:

  * OpenAI-compatible : usage.prompt_tokens / completion_tokens / total_tokens
  * Responses API     : usage.input_tokens / output_tokens
  * Gemini            : usageMetadata.promptTokenCount / candidatesTokenCount
  * Ollama (native)   : prompt_eval_count / eval_count

Consolidates the extraction logic previously duplicated in vision_gateway,
openai_adapter_vision, langfuse_client and usage_store.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .contracts import Usage


def _to_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


class UsageExtractor:
    """Extract a normalized `Usage` from a raw provider response."""

    @staticmethod
    def extract(raw: Any) -> Usage:
        if not isinstance(raw, dict):
            return Usage()

        # OpenAI-compatible / Responses API
        usage = raw.get("usage")
        if isinstance(usage, dict):
            inp = _to_int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
            out = _to_int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
            total = _to_int(usage.get("total_tokens", 0)) or (inp + out)
            return Usage(input_tokens=inp, output_tokens=out, total_tokens=total)

        # Gemini / Vertex — thinking ("thoughts") tokens are billed as output.
        meta = raw.get("usageMetadata")
        if isinstance(meta, dict):
            inp = _to_int(meta.get("promptTokenCount", 0))
            out = _to_int(meta.get("candidatesTokenCount", 0)) + _to_int(meta.get("thoughtsTokenCount", 0))
            total = _to_int(meta.get("totalTokenCount", 0)) or (inp + out)
            return Usage(input_tokens=inp, output_tokens=out, total_tokens=total)

        # Ollama native
        if "prompt_eval_count" in raw or "eval_count" in raw:
            inp = _to_int(raw.get("prompt_eval_count", 0))
            out = _to_int(raw.get("eval_count", 0))
            return Usage(input_tokens=inp, output_tokens=out, total_tokens=inp + out)

        return Usage()

    @staticmethod
    def extract_model(raw: Any, fallback: Optional[str] = None) -> Optional[str]:
        """Best-effort served-model name from the response."""
        if isinstance(raw, dict):
            model = raw.get("model") or raw.get("model_name") or raw.get("modelVersion")
            if model:
                return str(model)
        return fallback
