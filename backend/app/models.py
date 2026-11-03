"""Database tables."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(200), default="")
    password_hash: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    auth_provider: Mapped[str] = mapped_column(String(20), default="password")  # password | google
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Submission(Base):
    """One KYC attempt: documents + address + liveness frames -> checks -> decision."""

    __tablename__ = "kyc_submissions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)

    # draft -> documents_uploaded -> processing -> completed | failed
    status: Mapped[str] = mapped_column(String(30), default="draft")
    # approved | manual_review | rejected (set when completed)
    decision: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    id_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # nic | passport
    address_input: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    files: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    id_front_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    quality: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    document_result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    face_result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    checks: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
