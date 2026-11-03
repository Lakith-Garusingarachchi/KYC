"""FastAPI entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import init_db
from .redis_client import close_redis, get_redis
from .routers import auth, kyc
from .services.llm_service import provider_status

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("kyc")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    status = provider_status()
    selected = status["providers"].get(status["selected"]) or {}
    if not selected.get("configured"):
        logger.warning("LLM provider '%s' is not configured: %s", status["selected"], selected.get("reason"))
    else:
        logger.info("LLM provider: %s (%s)", status["selected"], selected.get("model"))
    yield
    await close_redis()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)
app.include_router(kyc.router)


@app.get("/health", tags=["health"])
async def health() -> dict:
    redis_ok = True
    try:
        await get_redis().ping()
    except Exception:
        redis_ok = False
    return {"status": "ok" if redis_ok else "degraded", "redis": redis_ok, "llm": provider_status()}
