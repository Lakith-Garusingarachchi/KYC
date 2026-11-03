"""Bridge between this app's .env settings and the el-aiee LLM adapters.

LLM_PROVIDER picks the adapter (gemini, gemini_vertex, openai, claude, deepseek);
the matching *_API_KEY / *_MODEL variables configure it. Calls go straight to
`AdapterFactory` (not LLMGateway) because the gateway's usage/pricing hooks
belong to el-aiee's MongoDB stack.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Settings, get_settings
from ..llm import AdapterFactory, LLMRequest, Media, MediaKind, ModelConfig
from ..llm._openai_compat import extract_first_json_object

logger = logging.getLogger(__name__)

# LLM_PROVIDER value -> provider string understood by AdapterFactory
_PROVIDER_ALIASES = {
    "gemini": "gemini",
    "google": "gemini",
    "gemini_vertex": "gemini_vertex",
    "gemini-vertex": "gemini_vertex",
    "vertex": "gemini_vertex",
    "openai": "openai",
    "claude": "claude",
    "anthropic": "claude",
    "deepseek": "deepseek",
}

SUPPORTED_PROVIDERS = ("gemini", "gemini_vertex", "openai", "claude", "deepseek")

_REASONING_MODEL = re.compile(r"(^|[-_/])(o\d|gpt-5)|reasoner", re.I)


class LLMNotConfiguredError(RuntimeError):
    """The selected provider is missing credentials or cannot handle the request."""


@dataclass
class Attachment:
    data: bytes
    mime_type: str
    filename: str

    def to_media(self) -> Media:
        kind = MediaKind.DOCUMENT if self.mime_type == "application/pdf" else MediaKind.IMAGE
        encoded = base64.b64encode(self.data).decode("ascii")
        return Media(
            kind=kind,
            source=f"data:{self.mime_type};base64,{encoded}",
            mime_type=self.mime_type,
            meta={"filename": self.filename},
        )


def normalize_provider(name: str) -> str:
    key = (name or "").strip().lower()
    if key not in _PROVIDER_ALIASES:
        raise LLMNotConfiguredError(
            f"Unknown LLM provider '{name}'. Use one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    return _PROVIDER_ALIASES[key]


def _vertex_service_account(settings: Settings) -> Optional[str]:
    if settings.vertex_service_account_json:
        return settings.vertex_service_account_json
    if settings.vertex_service_account_file:
        path = Path(settings.vertex_service_account_file)
        if not path.is_file():
            raise LLMNotConfiguredError(f"VERTEX_SERVICE_ACCOUNT_FILE not found: {path}")
        return path.read_text(encoding="utf-8")
    return None


def build_model_config(provider: str, settings: Optional[Settings] = None) -> ModelConfig:
    """Translate the .env settings for one provider into an el-aiee ModelConfig."""
    s = settings or get_settings()
    p = normalize_provider(provider)
    common: Dict[str, Any] = {
        "model_id": f"kyc-{p}",
        "model_type": "third_party",
        "timeout": s.llm_timeout_seconds,
        "max_retries": 3,
        "max_tokens": s.llm_max_tokens,
    }

    if p == "gemini":
        if not s.gemini_api_key:
            raise LLMNotConfiguredError("GEMINI_API_KEY is not set")
        return ModelConfig(**common, model_provider="gemini", model_name=s.gemini_model,
                           model_version=s.gemini_model, api_key=s.gemini_api_key)
    if p == "gemini_vertex":
        sa_json = _vertex_service_account(s)
        if not sa_json:
            raise LLMNotConfiguredError("Set VERTEX_SERVICE_ACCOUNT_FILE or VERTEX_SERVICE_ACCOUNT_JSON")
        return ModelConfig(**common, model_provider="gemini_vertex", model_name=s.vertex_model,
                           model_version=s.vertex_model, service_account_json=sa_json,
                           vertex_project=s.vertex_project_id or None, vertex_location=s.vertex_location)
    if p == "openai":
        if not s.openai_api_key:
            raise LLMNotConfiguredError("OPENAI_API_KEY is not set")
        return ModelConfig(**common, model_provider="openai", model_name=s.openai_model,
                           model_version=s.openai_model, api_key=s.openai_api_key,
                           endpoint_url=s.openai_endpoint or None, image_detail="high")
    if p == "claude":
        if not s.anthropic_api_key:
            raise LLMNotConfiguredError("ANTHROPIC_API_KEY is not set")
        return ModelConfig(**common, model_provider="claude", model_name=s.claude_model,
                           model_version=s.claude_model, api_key=s.anthropic_api_key,
                           endpoint_url=s.claude_endpoint or None)
    if p == "deepseek":
        if not s.deepseek_api_key:
            raise LLMNotConfiguredError("DEEPSEEK_API_KEY is not set")
        return ModelConfig(**common, model_provider="deepseek", model_name=s.deepseek_model,
                           model_version=s.deepseek_model, api_key=s.deepseek_api_key)
    raise LLMNotConfiguredError(f"Unsupported provider '{provider}'")  # pragma: no cover


async def ask_json(
    prompt: str,
    schema: Dict[str, Any],
    schema_name: str,
    attachments: Optional[List[Attachment]] = None,
    provider: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one structured-output call on the configured adapter and return the JSON object."""
    settings = get_settings()
    config = build_model_config(provider or settings.llm_provider, settings)
    if system_prompt:
        config = config.model_copy(update={"system_prompt": system_prompt})

    media = [a.to_media() for a in attachments or []]
    params: Dict[str, Any] = {}
    # Reasoning models (gpt-5*, o-series) reject a temperature other than the default.
    if not _REASONING_MODEL.search(config.model_version or config.model_name or ""):
        params["temperature"] = 0
    request = LLMRequest(
        prompt=prompt,
        config=config,
        media=media,
        response_schema=schema,
        response_schema_name=schema_name,
        params=params,
        source="kyc",
    )

    adapter = AdapterFactory.get_adapter(config)
    try:
        needed = request.required_capability()
        if needed is not None and not adapter.capabilities.supports(needed):
            raise LLMNotConfiguredError(
                f"{adapter.provider_name} ({config.model_name}) cannot read {needed.value} input. "
                f"Pick a vision provider (gemini, gemini_vertex, openai, claude) in LLM_PROVIDER."
            )
        response = await (adapter.complete_multimodal(request) if media else adapter.complete(request))
    finally:
        await adapter.close()

    data = response.data
    if data is None:
        data = extract_first_json_object(response.text or "") or {}
        if not data:
            try:
                parsed = json.loads(response.text)
                data = parsed if isinstance(parsed, dict) else {}
            except (TypeError, json.JSONDecodeError):
                data = {}
    logger.info("LLM %s/%s %s: in=%s out=%s tokens", response.provider, response.model_used,
                schema_name, response.usage.input_tokens, response.usage.output_tokens)
    if not data:
        raise RuntimeError(f"{schema_name}: the model returned no JSON")
    data["_model"] = f"{response.provider}/{response.model_used}"
    return data


def provider_status() -> Dict[str, Any]:
    """Which providers are configured (no secrets), for the health endpoint."""
    settings = get_settings()
    status: Dict[str, Any] = {"selected": settings.llm_provider, "text": settings.text_provider, "providers": {}}
    for name in SUPPORTED_PROVIDERS:
        try:
            cfg = build_model_config(name, settings)
            status["providers"][name] = {"configured": True, "model": cfg.model_name}
        except LLMNotConfiguredError as err:
            status["providers"][name] = {"configured": False, "reason": str(err)}
    return status
