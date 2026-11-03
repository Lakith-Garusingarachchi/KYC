"""Redis connection. Holds short-lived state: liveness challenges, OAuth state,
login rate limits, processing progress and finalize locks."""

from __future__ import annotations

from typing import Optional

from redis.asyncio import Redis

from .config import get_settings

_client: Optional[Redis] = None


def get_redis() -> Redis:
    global _client
    if _client is None:
        _client = Redis.from_url(get_settings().redis_url, decode_responses=True)
    return _client


def set_redis(client: Redis) -> None:
    """Swap the client (tests use fakeredis)."""
    global _client
    _client = client


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
