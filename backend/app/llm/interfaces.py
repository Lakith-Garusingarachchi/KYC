"""
Adapter interfaces (the clean contract).

Every provider adapter implements `LLMAdapter`. Adapters that can handle images
or documents additionally set the relevant `Capabilities` flags and implement
`complete_multimodal`. The gateway routes by the request payload: a text-only
request calls `complete`; a request carrying media calls `complete_multimodal`
on an adapter that declares the capability.

This replaces the old `generate(prompt, image_url=None)` where every text
adapter carried a vision parameter it usually rejected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .contracts import Capabilities, LLMRequest, LLMResponse


class LLMAdapter(ABC):
    """Base contract every adapter honors."""

    #: Declared per adapter; the gateway checks this before routing media.
    capabilities: Capabilities = Capabilities()

    #: Human-readable provider label, e.g. "OpenAI" (used on the usage record).
    provider_name: str = "unknown"

    #: The `LLMResponse` behind the most recent legacy `generate*` call. Those
    #: shims return plain dicts — for structured calls only the parsed data,
    #: which carries no token usage — so callers read usage from here instead.
    #: Adapters are built per execution, so this is never shared across requests.
    last_response: Optional[LLMResponse] = None

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run a text (and optionally structured-output) completion."""
        raise NotImplementedError

    async def complete_multimodal(self, request: LLMRequest) -> LLMResponse:
        """
        Run a completion that carries media (images/documents).

        Default implementation raises: adapters that declare `image`/`document`
        capability MUST override this. The gateway never calls it on an adapter
        whose capabilities don't allow the request's media kind.
        """
        raise NotImplementedError(
            f"{self.provider_name} adapter does not support multimodal input"
        )

    async def close(self) -> None:
        """Release any resources (HTTP client). Best-effort; safe to no-op."""
        return None

    async def __aenter__(self) -> "LLMAdapter":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
