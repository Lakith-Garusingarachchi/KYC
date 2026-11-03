"""Password hashing, JWT access tokens and the current-user dependency.

Login follows the OAuth2 password flow (POST /auth/token, form fields
username + password) and returns a JWT signed with AUTH_SECRET_KEY. The token is
also set as an httpOnly cookie so the browser app never touches it; API clients
(Swagger, scripts) can send it as `Authorization: Bearer <token>` instead.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .database import get_db
from .models import User

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

_ephemeral_secret: Optional[str] = None


def _secret() -> str:
    global _ephemeral_secret
    key = get_settings().auth_secret_key
    if key:
        return key
    if _ephemeral_secret is None:
        logger.warning("AUTH_SECRET_KEY is not set; using a random key (all tokens die on restart)")
        _ephemeral_secret = secrets.token_urlsafe(48)
    return _ephemeral_secret


# --- passwords -----------------------------------------------------------------
def _prehash(password: str) -> bytes:
    # bcrypt only reads 72 bytes; pre-hashing keeps long passwords fully significant.
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: Optional[str]) -> bool:
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(_prehash(password), password_hash.encode("ascii"))
    except ValueError:
        return False


# --- tokens --------------------------------------------------------------------
def create_access_token(user_id: str) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
        "type": "access",
    }
    return jwt.encode(payload, _secret(), algorithm=settings.auth_algorithm)


def decode_access_token(token: str) -> Optional[str]:
    settings = get_settings()
    try:
        payload = jwt.decode(token, _secret(), algorithms=[settings.auth_algorithm])
    except jwt.PyJWTError:
        return None
    if payload.get("type") != "access":
        return None
    return payload.get("sub")


def set_auth_cookie(response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
        max_age=settings.access_token_expire_minutes * 60,
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(get_settings().auth_cookie_name, path="/")


async def get_current_user(
    request: Request,
    bearer: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    token = bearer or request.cookies.get(get_settings().auth_cookie_name)
    user_id = decode_access_token(token) if token else None
    user = await db.get(User, user_id) if user_id else None
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
