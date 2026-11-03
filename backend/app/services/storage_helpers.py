"""Glue between stored upload metadata and LLM attachments."""

from __future__ import annotations

from typing import Dict, Optional

from .. import storage
from .llm_service import Attachment


def attachment(files: Dict, key: str) -> Optional[Attachment]:
    meta = (files or {}).get(key)
    if not meta:
        return None
    return Attachment(data=storage.load_file(meta["path"]), mime_type=meta["mime_type"],
                      filename=meta.get("filename") or key)
