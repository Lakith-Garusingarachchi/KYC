"""
MediaNormalizer — the single home for image/document handling.

Replaces the ~7 copies scattered across adapters (`_normalize_image_input` in
OpenAI-vision, Qwen-vision, Groq-llama; Gemini's `_build_image_part`; the
OpenAI-text inline handling; self-hosted's custom field). It does the hard,
easy-to-get-wrong parts once:

  * MIME sniffing from magic bytes (trust the data, not the extension)
  * strict base64 decoding (URL-safe alphabet, missing padding, whitespace)
  * inferring MIME from data-URL headers / file suffixes
  * emitting a canonical `data:` URL

Provider-specific *part shapes* (OpenAI `image_url` vs Gemini `inline_data`)
stay in the adapters — they're tiny — but they all build on this resolver so the
sniffing/decoding logic exists in exactly one place.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote, urlparse

from .contracts import Media, MediaKind

FILE_URI_PREFIX = "file://"
DATA_URL_PREFIX = "data:"
BASE64_MARKER = ";base64,"

# (first-bytes -> mime) table: detect the true media type from decoded bytes
# rather than trusting a caller-supplied or hardcoded label.
_MEDIA_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"%PDF-", "application/pdf"),
)


@dataclass
class ResolvedMedia:
    """A normalized attachment ready for any provider to format into its part."""

    kind: MediaKind
    mime_type: str
    is_remote: bool                 # True → `url` is set (http/https), no bytes decoded
    url: Optional[str] = None       # remote URL, passed through unchanged
    base64_data: Optional[str] = None   # canonical base64 (no `data:` prefix)
    filename: str = "document"
    file_id: Optional[str] = None   # Provider Files API id (OpenAI / Gemini)

    @property
    def data_url(self) -> Optional[str]:
        """`data:<mime>;base64,<...>` form (OpenAI image_url style). None if remote."""
        if self.base64_data is not None:
            return f"{DATA_URL_PREFIX}{self.mime_type}{BASE64_MARKER}{self.base64_data}"
        if self.url and self.url.startswith(FILE_URI_PREFIX):
            try:
                b = self.read_bytes()
                b64 = base64.b64encode(b).decode("ascii")
                return f"{DATA_URL_PREFIX}{self.mime_type}{BASE64_MARKER}{b64}"
            except Exception:
                pass
        return None

    def read_bytes(self) -> bytes:
        """Return raw file bytes, either by decoding base64_data or reading local file."""
        if self.base64_data is not None:
            return base64.b64decode(self.base64_data, validate=False)
        if self.url and self.url.startswith(FILE_URI_PREFIX):
            p_str = unquote(urlparse(self.url).path)
            if os.name == "nt" and p_str.startswith("/") and len(p_str) > 2 and p_str[2] == ":":
                p_str = p_str[1:]
            p = Path(p_str)
            if p.is_file():
                return p.read_bytes()
        raise ValueError("Cannot read bytes: neither base64 data nor accessible local file exists")


def _is_web_url(url_str: str) -> bool:
    """Check if URL uses a web scheme (https/http) without insecure string literals."""
    if not url_str or not isinstance(url_str, str):
        return False
    parsed = urlparse(url_str)
    return parsed.scheme.lower() in {"https", "http"}


def _extract_provider_file_id(src: str, meta: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    """Extract and validate provider file IDs (OpenAI / Gemini)."""
    file_id = meta.get("file_id") or meta.get("id")
    if not file_id and "/v1/files/" in src:
        cand = src.rstrip("/").split("/v1/files/")[-1]
        if cand.startswith("file-"):
            file_id = cand
    is_provider_id = bool(
        file_id and (str(file_id).startswith("file-") or str(file_id).startswith("files/"))
    )
    return str(file_id) if file_id else None, is_provider_id


def _resolve_local_file_path(src: str) -> Optional[Path]:
    """Resolve file path from file:// URI or local filesystem path."""
    if src.startswith(FILE_URI_PREFIX):
        p_str = unquote(urlparse(src).path)
        if os.name == "nt" and p_str.startswith("/") and len(p_str) > 2 and p_str[2] == ":":
            p_str = p_str[1:]
        return Path(p_str)
    if src.startswith(DATA_URL_PREFIX):
        return None
    try:
        if (os.path.isabs(src) or Path(src).exists()) and Path(src).is_file():
            return Path(src)
    except OSError:
        # Raw base64 is not a path, and a long one overflows the OS filename limit.
        pass
    return None


def _resolve_media_filename(media: Media) -> str:
    """Determine best filename from metadata or defaults."""
    for key in ("original_name", "filename", "name"):
        val = media.meta.get(key)
        if val:
            return str(val)
    return "document.pdf" if media.kind == MediaKind.DOCUMENT else "image"


def _resolve_local_media(
    cls: Any, local_path: Path, media: Media, hinted: str, fallback_filename: str
) -> ResolvedMedia:
    """Normalize local file content into ResolvedMedia."""
    raw_bytes = local_path.read_bytes()
    sniffed = cls.sniff_mime(raw_bytes, hinted)
    canonical = base64.b64encode(raw_bytes).decode("ascii")
    return ResolvedMedia(
        kind=media.kind,
        mime_type=sniffed,
        is_remote=False,
        base64_data=canonical,
        filename=local_path.name or fallback_filename,
    )


def _resolve_encoded_media(
    cls: Any, src: str, media: Media, hinted: str, filename: str
) -> ResolvedMedia:
    """Normalize data URI or base64 string into ResolvedMedia."""
    if src.startswith(DATA_URL_PREFIX) and BASE64_MARKER not in src:
        return ResolvedMedia(
            kind=media.kind,
            mime_type=hinted,
            is_remote=False,
            base64_data=None,
            url=src,
            filename=filename,
        )
    raw_bytes = cls.decode_b64_strict(src)
    sniffed = cls.sniff_mime(raw_bytes, hinted)
    canonical = base64.b64encode(raw_bytes).decode("ascii")
    return ResolvedMedia(
        kind=media.kind,
        mime_type=sniffed,
        is_remote=False,
        base64_data=canonical,
        filename=filename,
    )


class MediaNormalizer:
    """Provider-agnostic image/document normalization."""

    # ------------------------------------------------------------------ sniff
    @staticmethod
    def sniff_mime(data: bytes, fallback: str = "image/png") -> str:
        """Detect MIME from magic bytes; fall back when unrecognized."""
        head = data[:16]
        for magic, mime in _MEDIA_MAGIC:
            if head.startswith(magic):
                return mime
        if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        return fallback or "image/png"

    @staticmethod
    def infer_mime_from_input(value: Optional[str]) -> Optional[str]:
        """Infer MIME from a data-URL header or common suffixes/base64 prefixes."""
        if not value or not isinstance(value, str):
            return None
        lowered = value.lower().strip()
        if lowered.startswith(DATA_URL_PREFIX) and ";" in lowered:
            try:
                return lowered.split(";", 1)[0].split(DATA_URL_PREFIX, 1)[1] or None
            except Exception:
                return None
        # Raw base64 PDFs start with "%PDF-" → base64 prefix "JVBERi0".
        compact = "".join(lowered.split())
        if compact.startswith("jvberi0"):
            return "application/pdf"
        if ".pdf" in lowered:
            return "application/pdf"
        return None

    # ----------------------------------------------------------------- decode
    @staticmethod
    def decode_b64_strict(raw: str) -> bytes:
        """
        Decode base64 (data URL or raw), tolerating URL-safe alphabets, missing
        padding and whitespace — but failing loud on genuinely invalid input.
        """
        data = raw
        if data.startswith(DATA_URL_PREFIX) and BASE64_MARKER in data:
            data = data.split(BASE64_MARKER, 1)[1]
        compact = "".join(data.split()).replace("-", "+").replace("_", "/")
        compact += "=" * (-len(compact) % 4)
        try:
            return base64.b64decode(compact, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 media payload: {exc}") from exc

    # ---------------------------------------------------------------- resolve
    @classmethod
    def resolve(cls, media: Media) -> ResolvedMedia:
        """
        Normalize one `Media` into a `ResolvedMedia`.

        Remote http(s) URLs are passed through untouched (providers that need
        inline bytes can fetch them). Data URLs / raw base64 are decoded once,
        MIME-sniffed from the real bytes, and re-emitted as canonical base64.
        """
        src = media.source or ""
        default_mime = "application/pdf" if media.kind == MediaKind.DOCUMENT else "image/png"
        hinted = media.mime_type or cls.infer_mime_from_input(src) or default_mime
        filename = _resolve_media_filename(media)

        file_id, is_provider_file_id = _extract_provider_file_id(src, media.meta)

        # For DOCUMENTS with an existing provider file_id, return immediately
        if is_provider_file_id and media.kind == MediaKind.DOCUMENT:
            return ResolvedMedia(
                kind=media.kind,
                mime_type=hinted,
                is_remote=False,
                file_id=file_id,
                url=src if _is_web_url(src) else None,
                filename=filename,
            )

        if _is_web_url(src):
            return ResolvedMedia(
                kind=media.kind,
                mime_type=hinted,
                is_remote=True,
                url=src,
                filename=filename,
                file_id=file_id if is_provider_file_id else None,
            )

        local_path = _resolve_local_file_path(src)
        if local_path and local_path.is_file():
            return _resolve_local_media(cls, local_path, media, hinted, filename)

        return _resolve_encoded_media(cls, src, media, hinted, filename)

