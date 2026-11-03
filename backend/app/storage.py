"""Encrypted file storage for uploaded documents and face frames.

Files are written under UPLOAD_DIR/<submission_id>/ and encrypted with Fernet
(FILE_ENCRYPTION_KEY). Without a key they are stored in plain form and a
warning is logged — fine for a laptop demo, not for real customer data.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet

from .config import get_settings

logger = logging.getLogger(__name__)

_fernet: Optional[Fernet] = None
_warned = False


def _cipher() -> Optional[Fernet]:
    global _fernet, _warned
    key = get_settings().file_encryption_key
    if not key:
        if not _warned:
            logger.warning("FILE_ENCRYPTION_KEY is not set; uploads are stored unencrypted")
            _warned = True
        return None
    if _fernet is None:
        _fernet = Fernet(key.encode("ascii"))
    return _fernet


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_file(submission_id: str, name: str, data: bytes) -> str:
    """Store bytes; returns the path relative to UPLOAD_DIR."""
    root = Path(get_settings().upload_dir)
    folder = root / submission_id
    folder.mkdir(parents=True, exist_ok=True)
    cipher = _cipher()
    payload = cipher.encrypt(data) if cipher else data
    path = folder / (name + (".enc" if cipher else ""))
    path.write_bytes(payload)
    return str(path.relative_to(root))


def load_file(relative_path: str) -> bytes:
    root = Path(get_settings().upload_dir)
    path = (root / relative_path).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("invalid storage path")
    data = path.read_bytes()
    if path.suffix == ".enc":
        cipher = _cipher()
        if cipher is None:
            raise RuntimeError("file is encrypted but FILE_ENCRYPTION_KEY is not set")
        return cipher.decrypt(data)
    return data
