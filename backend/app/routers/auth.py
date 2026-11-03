"""Login: OAuth2 password flow with JWT (+ optional Google OAuth sign-in)."""

from __future__ import annotations

import logging
import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..database import get_db
from ..models import User
from ..redis_client import get_redis
from ..schemas import ProvidersOut, RegisterIn, TokenOut, UserOut
from ..security import (
    clear_auth_cookie,
    create_access_token,
    get_current_user,
    hash_password,
    set_auth_cookie,
    verify_password,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

MAX_FAILED_LOGINS = 10
FAILED_LOGIN_WINDOW = 15 * 60

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


def _user_out(user: User) -> UserOut:
    return UserOut(id=user.id, email=user.email, full_name=user.full_name, auth_provider=user.auth_provider)


def _issue(response: Response, user: User) -> TokenOut:
    token = create_access_token(user.id)
    set_auth_cookie(response, token)
    return TokenOut(access_token=token)


@router.get("/providers", response_model=ProvidersOut)
async def providers() -> ProvidersOut:
    return ProvidersOut(password=True, google=get_settings().google_oauth_enabled)


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterIn, response: Response, db: AsyncSession = Depends(get_db)) -> TokenOut:
    email = body.email.lower()
    existing = await db.scalar(select(User).where(User.email == email))
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    user = User(email=email, full_name=body.full_name.strip(), password_hash=hash_password(body.password))
    db.add(user)
    await db.commit()
    return _issue(response, user)


@router.post("/token", response_model=TokenOut)
async def login(
    response: Response,
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> TokenOut:
    email = form.username.strip().lower()
    redis = get_redis()
    fail_key = f"kyc:login_fail:{email}"
    if int(await redis.get(fail_key) or 0) >= MAX_FAILED_LOGINS:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many failed attempts. Try again later.")

    user = await db.scalar(select(User).where(User.email == email))
    if user is None or not verify_password(form.password, user.password_hash):
        await redis.incr(fail_key)
        await redis.expire(fail_key, FAILED_LOGIN_WINDOW)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password",
                            headers={"WWW-Authenticate": "Bearer"})
    await redis.delete(fail_key)
    return _issue(response, user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response) -> Response:
    clear_auth_cookie(response)
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)) -> UserOut:
    return _user_out(user)


# --- Google OAuth (authorization code flow) ----------------------------------------
@router.get("/google/login")
async def google_login() -> RedirectResponse:
    settings = get_settings()
    if not settings.google_oauth_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Google sign-in is not configured")
    state = secrets.token_urlsafe(24)
    await get_redis().set(f"kyc:oauth_state:{state}", "1", ex=600)
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}")


@router.get("/google/callback")
async def google_callback(request: Request, db: AsyncSession = Depends(get_db)) -> RedirectResponse:
    settings = get_settings()
    login_url = f"{settings.frontend_url}/login"
    code, state = request.query_params.get("code"), request.query_params.get("state")
    if not code or not state or not await get_redis().getdel(f"kyc:oauth_state:{state}"):
        return RedirectResponse(f"{login_url}?error=oauth_state")

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            token_resp = await client.post(GOOGLE_TOKEN_URL, data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            })
            token_resp.raise_for_status()
            access_token = token_resp.json()["access_token"]
            info_resp = await client.get(GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"})
            info_resp.raise_for_status()
            info = info_resp.json()
    except (httpx.HTTPError, KeyError, ValueError):
        logger.warning("Google OAuth exchange failed", exc_info=True)
        return RedirectResponse(f"{login_url}?error=oauth_exchange")

    email = (info.get("email") or "").lower()
    if not email or not info.get("email_verified"):
        return RedirectResponse(f"{login_url}?error=oauth_email")

    user = await db.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(email=email, full_name=info.get("name") or "", auth_provider="google")
        db.add(user)
        await db.commit()

    redirect = RedirectResponse(f"{settings.frontend_url}/kyc")
    set_auth_cookie(redirect, create_access_token(user.id))
    return redirect
