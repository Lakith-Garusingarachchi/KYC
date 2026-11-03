"""
LLM layer contracts.

These are the single, provider-agnostic request/response shapes that flow
through the LLM gateway and every adapter. They replace the old situation where
each adapter returned a different thing (raw provider JSON here, a normalized
dict there) and where `image_url` was bolted onto text calls.

Design notes
------------
- `LLMRequest`  : everything a call needs, incl. the tracking context (user_id,
  source) so the gateway can record usage without threading extra args.
- `LLMResponse` : one normalized result. `raw` preserves the untouched provider
  payload so existing callers that read `response["choices"]` / `["model"]`
  keep working through a backward-compatible shim (see gateway).
- Capabilities  : each adapter declares what it supports; the gateway routes by
  the *payload* (does it carry media / a document?) instead of guessing.

Plain dataclasses (no pydantic) keep this dependency-light and importable from
anywhere in the LLM layer without circular imports.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CostModel(str, Enum):
    """How a model is billed. Drives cost computation in the rate table."""

    PER_TOKEN = "per_token"       # OpenAI, DeepSeek, Qwen, NVIDIA, Gemini, Groq
    FREE = "free"                 # local Ollama / self-hosted (no per-token charge)
    SUBSCRIPTION = "subscription" # Ollama Cloud flat monthly fee (no per-call $)


class MediaKind(str, Enum):
    """The kind of attachment on a request. Determines the capability required."""

    IMAGE = "image"
    DOCUMENT = "document"


@dataclass
class Media:
    """
    One attachment on a request (image or document).

    `source` accepts what the adapters already accept: a data URL, an http(s)
    URL, a local file path, or a raw base64 string. The MediaNormalizer resolves
    it to the right provider part and sniffs the MIME type when absent.
    """

    kind: MediaKind
    source: str
    mime_type: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)  # e.g. original filename for PDFs


@dataclass(frozen=True)
class Capabilities:
    """What an adapter can do. Declared per adapter; checked by the gateway."""

    text: bool = True
    image: bool = False
    document: bool = False
    structured_output: bool = False

    def supports(self, media_kind: MediaKind) -> bool:
        if media_kind == MediaKind.IMAGE:
            return self.image
        if media_kind == MediaKind.DOCUMENT:
            return self.document
        return True


@dataclass
class Usage:
    """Token usage read from the provider response (never counted locally)."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Cost:
    """Computed cost for one call, derived from Usage + the rate table."""

    cost_model: CostModel = CostModel.PER_TOKEN
    input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    total_cost_usd: Optional[float] = None
    currency: str = "USD"
    rate_used: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cost_model": self.cost_model.value,
            "input_cost_usd": self.input_cost_usd,
            "output_cost_usd": self.output_cost_usd,
            "total_cost_usd": self.total_cost_usd,
            "currency": self.currency,
            "rate_used": self.rate_used,
        }


@dataclass
class LLMRequest:
    """A single, provider-agnostic LLM call."""

    prompt: str
    config: "Any"                                   # ModelConfig (avoid import cycle)
    media: List[Media] = field(default_factory=list)
    response_schema: Optional[Dict[str, Any]] = None
    response_schema_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)  # temperature, max_tokens, thinking…

    # Tracking context (flows straight into the usage record; optional so the
    # gateway still works when a caller has no user context).
    user_id: Optional[str] = None
    source: str = "unknown"                          # "function" | "agent" | "report" | …
    execution_id: Optional[str] = None

    @property
    def wants_structured(self) -> bool:
        return self.response_schema is not None

    def required_capability(self) -> Optional[MediaKind]:
        """The strongest media capability this request needs, if any."""
        if any(m.kind == MediaKind.DOCUMENT for m in self.media):
            return MediaKind.DOCUMENT
        if any(m.kind == MediaKind.IMAGE for m in self.media):
            return MediaKind.IMAGE
        return None


def media_from_legacy(items: Optional[List[Dict[str, Any]]]) -> List["Media"]:
    """
    Convert the legacy multimodal media format into `Media` objects.

    Legacy items are dicts like {"data"|"base64"|"url": ..., "label"|"name":
    ..., "mime_type": ...} as used by VisionGateway / the document-comparison
    flow. Kind is DOCUMENT when the mime/label looks like a PDF, else IMAGE.
    """
    result: List[Media] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        source = item.get("data") or item.get("base64") or item.get("url")
        if not source:
            continue
        mime = item.get("mime_type")
        label = item.get("label") or item.get("name")
        is_pdf = (mime or "").lower() == "application/pdf" or str(label or "").lower().endswith(".pdf")
        # Unlabeled/generic mime: fall back to the bytes so a raw base64 PDF isn't
        # treated as an image (same defect _looks_like_document guards against).
        if not is_pdf and (mime or "").lower() in ("", "application/octet-stream"):
            is_pdf = _sniff_is_pdf(source)
        meta = {"filename": label} if label else {}
        fid = item.get("file_id") or item.get("id")
        if fid and (str(fid).startswith("file-") or str(fid).startswith("files/")):
            meta["file_id"] = str(fid)
        result.append(
            Media(
                kind=MediaKind.DOCUMENT if is_pdf else MediaKind.IMAGE,
                source=source,
                mime_type=mime,
                meta=meta,
            )
        )
    return result


def _sniff_is_pdf(source: Optional[str]) -> bool:
    """Decode the leading bytes of a base64 / data-URL source and test the PDF magic header.

    Guards against a generic mime label (e.g. application/octet-stream) hiding a
    real PDF: the label lies, the bytes don't. Remote URLs and file paths carry no
    inline bytes, so they're skipped here (their mime/extension is trusted instead).
    """
    if not source or not isinstance(source, str):
        return False
    data = source
    if data.startswith("data:") and ";base64," in data:
        data = data.split(";base64,", 1)[1]
    elif data.startswith(("http://", "https://")):
        return False
    # Only the first few base64 chars are needed to read the magic bytes.
    prefix = "".join(data[:24].split()).replace("-", "+").replace("_", "/")
    prefix += "=" * (-len(prefix) % 4)
    try:
        head = base64.b64decode(prefix, validate=False)
    except Exception:
        return False
    return head.startswith(b"%PDF-")


def _looks_like_document(mime: Optional[str], source: Optional[str]) -> bool:
    """True when a mime/source points at a document (PDF/office) rather than an image."""
    m = (mime or "").lower()
    if m.startswith("application/pdf") or m == "application/msword" or "officedocument" in m:
        return True
    s = (source or "")[:64].lower()
    if s.startswith("data:application/pdf") or s.endswith(".pdf"):
        return True
    # A generic/unknown mime hides the real type — trust the bytes, not the label,
    # so an unlabeled base64 PDF isn't misrouted to the image path.
    if m in ("", "application/octet-stream"):
        return _sniff_is_pdf(source)
    return False


def legacy_media(
    image_url: Optional[str] = None,
    file_meta: Optional[Dict[str, Any]] = None,
    mime_type: Optional[str] = None,
) -> List["Media"]:
    """
    Build `Media` from the legacy adapter-call arguments used by the function
    executor: an `image_url` (str) and/or a `file` metadata dict. A PDF/office
    mime routes to DOCUMENT; everything else to IMAGE. `file_meta` takes
    precedence to avoid double-adding the same source.
    """
    media: List[Media] = []
    if isinstance(file_meta, dict):
        src = file_meta.get("url") or file_meta.get("path") or file_meta.get("base64") or file_meta.get("data")
        fmime = file_meta.get("mime_type") or mime_type
        if src:
            kind = MediaKind.DOCUMENT if _looks_like_document(fmime, src) else MediaKind.IMAGE
            fname = file_meta.get("original_name") or file_meta.get("name")
            fid = file_meta.get("file_id") or file_meta.get("id")
            meta = {}
            if fname:
                meta["filename"] = fname
            if fid and (str(fid).startswith("file-") or str(fid).startswith("files/")):
                meta["file_id"] = str(fid)
            media.append(Media(kind=kind, source=src, mime_type=fmime, meta=meta))
    if image_url and not media:
        kind = MediaKind.DOCUMENT if _looks_like_document(mime_type, image_url) else MediaKind.IMAGE
        media.append(Media(kind=kind, source=image_url, mime_type=mime_type))
    return media


@dataclass
class LLMResponse:
    """One normalized LLM result. `raw` keeps the provider payload for compat."""

    text: str = ""
    data: Optional[Dict[str, Any]] = None            # parsed JSON when a schema was requested
    model_used: Optional[str] = None                 # exact served model, from the response
    provider: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    cost: Optional[Cost] = None
    finish_reason: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "data": self.data,
            "model_used": self.model_used,
            "provider": self.provider,
            "usage": self.usage.as_dict(),
            "cost": self.cost.as_dict() if self.cost else None,
            "finish_reason": self.finish_reason,
        }
