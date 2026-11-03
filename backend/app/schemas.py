"""API request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(default="", max_length=200)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: str
    full_name: str
    auth_provider: str


class ProvidersOut(BaseModel):
    password: bool = True
    google: bool = False


class CheckOut(BaseModel):
    id: str
    category: str
    label: str
    status: str
    detail: str = ""


class ChallengeOut(BaseModel):
    challenge_id: str
    steps: List[str]
    expires_in: int


class LivenessOut(BaseModel):
    accepted: bool
    liveness_pass: bool
    attempts_left: int
    issues: List[str]
    face_match: Dict[str, Any]
    submission_status: str


class SubmissionOut(BaseModel):
    id: str
    status: str
    decision: Optional[str] = None
    progress: Optional[str] = None
    id_type: Optional[str] = None
    address_input: Optional[str] = None
    uploaded: List[str] = Field(default_factory=list)
    quality: Dict[str, Any] = Field(default_factory=dict)
    extracted: Dict[str, Any] = Field(default_factory=dict)
    face: Dict[str, Any] = Field(default_factory=dict)
    checks: List[CheckOut] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
