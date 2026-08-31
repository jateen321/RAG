"""Distributed admission control for costly API operations.

Rate limiting is disabled by default for local development. A public deployment
must set ``RAG_RATE_LIMIT_ENABLED=1`` and ``REDIS_URL``. When enabled, Redis is
the source of truth across every API worker and replica.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterable
from uuid import uuid4


logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """The caller exhausted a bucket or all concurrency slots are occupied."""

    def __init__(self, retry_after: float, detail: str) -> None:
        self.retry_after = max(1, math.ceil(retry_after))
        self.detail = detail
        super().__init__(detail)


class RateLimitUnavailable(Exception):
    """Admission could not be verified against the shared Redis state."""


@dataclass(frozen=True)
class RatePolicy:
    name: str
    capacity: int
    refill_period_s: float


@dataclass(frozen=True)
class ConcurrencyPolicy:
    name: str
    limit: int
    lease_s: int


def _positive_number(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero.")
    return value


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _per_minute(name: str, default: float) -> float:
    return 60 / _positive_number(name, default)


# Capacities are burst sizes. Refill periods derive from environment-configured
# sustained rates, keeping product policy separate from the Redis mechanism.
RATE_POLICIES = {
    "ask": RatePolicy(
        "ask",
        capacity=_positive_int("RAG_RATE_LIMIT_ASK_BURST", 3),
        refill_period_s=_per_minute("RAG_RATE_LIMIT_ASK_PER_MINUTE", 10),
    ),
    "web": RatePolicy(
        "web",
        capacity=_positive_int("RAG_RATE_LIMIT_WEB_BURST", 2),
        refill_period_s=_per_minute("RAG_RATE_LIMIT_WEB_PER_MINUTE", 5),
    ),
    "image": RatePolicy(
        "image",
        capacity=_positive_int("RAG_RATE_LIMIT_IMAGE_BURST", 1),
        refill_period_s=_per_minute("RAG_RATE_LIMIT_IMAGE_PER_MINUTE", 2),
    ),
    "ingest": RatePolicy(
        "ingest",
        capacity=_positive_int("RAG_RATE_LIMIT_INGEST_BURST", 5),
        refill_period_s=(
            3600 / _positive_number("RAG_RATE_LIMIT_INGEST_PER_HOUR", 5)
        ),
    ),
}

CONCURRENCY_POLICIES = {
    "interactive": ConcurrencyPolicy(
        "interactive",
        limit=_positive_int("RAG_CONCURRENCY_INTERACTIVE", 4),
        lease_s=180,
    ),
    "web": ConcurrencyPolicy(
        "web", limit=_positive_int("RAG_CONCURRENCY_WEB", 2), lease_s=180,
    ),
    "image": ConcurrencyPolicy(
        "image", limit=_positive_int("RAG_CONCURRENCY_IMAGE", 1), lease_s=300,
    ),
    # OCR and embedding can make one ingestion request last many minutes.
    "ingest": ConcurrencyPolicy(
        "ingest", limit=_positive_int("RAG_CONCURRENCY_INGEST", 1), lease_s=900,
    ),
}


_TOKEN_BUCKET_SCRIPT = """
local now = redis.call('TIME')
local now_ms = (tonumber(now[1]) * 1000) + math.floor(tonumber(now[2]) / 1000)
local states = {}
local max_retry_ms = 0

for i, key in ipairs(KEYS) do
  local capacity = tonumber(ARGV[(i - 1) * 2 + 1])
  local refill_ms = tonumber(ARGV[(i - 1) * 2 + 2])
  local values = redis.call('HMGET', key, 'tokens', 'updated_ms')
  local tokens = tonumber(values[1]) or capacity
  local updated_ms = tonumber(values[2]) or now_ms
  tokens = math.min(capacity, tokens + math.max(0, now_ms - updated_ms) / refill_ms)
  states[i] = tokens
  if tokens < 1 then
    max_retry_ms = math.max(max_retry_ms, math.ceil((1 - tokens) * refill_ms))
  end
end

if max_retry_ms > 0 then
  return {0, max_retry_ms}
end

for i, key in ipairs(KEYS) do
  local capacity = tonumber(ARGV[(i - 1) * 2 + 1])
  local refill_ms = tonumber(ARGV[(i - 1) * 2 + 2])
  redis.call('HSET', key, 'tokens', states[i] - 1, 'updated_ms', now_ms)
  redis.call('PEXPIRE', key, math.ceil(refill_ms * capacity * 2))
end
return {1, 0}
"""


_ACQUIRE_CONCURRENCY_SCRIPT = """
local now = redis.call('TIME')
local now_ms = (tonumber(now[1]) * 1000) + math.floor(tonumber(now[2]) / 1000)
local token = ARGV[1]
local max_retry_ms = 0

for i, key in ipairs(KEYS) do
  local limit = tonumber(ARGV[(i - 1) * 2 + 2])
  redis.call('ZREMRANGEBYSCORE', key, '-inf', now_ms)
  if redis.call('ZCARD', key) >= limit then
    local earliest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    if earliest[2] then
      max_retry_ms = math.max(max_retry_ms, tonumber(earliest[2]) - now_ms)
    end
  end
end

if max_retry_ms > 0 then
  return {0, max_retry_ms}
end

for i, key in ipairs(KEYS) do
  local lease_ms = tonumber(ARGV[(i - 1) * 2 + 3])
  redis.call('ZADD', key, now_ms + lease_ms, token)
  redis.call('PEXPIRE', key, lease_ms * 2)
end
return {1, 0}
"""


_RENEW_CONCURRENCY_SCRIPT = """
local now = redis.call('TIME')
local now_ms = (tonumber(now[1]) * 1000) + math.floor(tonumber(now[2]) / 1000)
local token = ARGV[1]
local renewed = 0
for i, key in ipairs(KEYS) do
  local lease_ms = tonumber(ARGV[i + 1])
  if redis.call('ZSCORE', key, token) then
    redis.call('ZADD', key, now_ms + lease_ms, token)
    redis.call('PEXPIRE', key, lease_ms * 2)
    renewed = renewed + 1
  end
end
return renewed
"""


_RELEASE_CONCURRENCY_SCRIPT = """
local token = ARGV[1]
for _, key in ipairs(KEYS) do
  redis.call('ZREM', key, token)
end
return 1
"""


def _enabled_from_environment() -> bool:
    return os.getenv("RAG_RATE_LIMIT_ENABLED", "0").strip().lower() in {
        "1", "true", "yes",
    }


def _identity_tag(identity: str) -> str:
    """Hash authenticated IDs before placing them in Redis keys and hash tags."""
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


class DistributedRateLimiter:
    """Atomic per-user rates plus global, renewable concurrency leases."""

    def __init__(self, redis_client=None, *, enabled: bool | None = None) -> None:
        self.enabled = _enabled_from_environment() if enabled is None else enabled
        self._redis = redis_client
        if self.enabled and self._redis is None:
            redis_url = os.getenv("REDIS_URL", "").strip()
            if not redis_url:
                raise RuntimeError(
                    "REDIS_URL is required when RAG_RATE_LIMIT_ENABLED=1."
                )
            try:
                from redis.asyncio import Redis
            except ImportError as exc:  # pragma: no cover - packaging error
                raise RuntimeError(
                    "The redis package is required when rate limiting is enabled."
                ) from exc
            self._redis = Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
            )

    @staticmethod
    def _rate_keys(identity: str, policies: Iterable[RatePolicy]) -> list[str]:
        identity_tag = _identity_tag(identity)
        return [
            f"sarthi:rate:{{{identity_tag}}}:{policy.name}"
            for policy in policies
        ]

    @staticmethod
    def _concurrency_keys(policies: Iterable[ConcurrencyPolicy]) -> list[str]:
        return [f"sarthi:concurrency:{{global}}:{policy.name}" for policy in policies]

    async def _eval(self, script: str, keys: list[str], args: list[object]):
        try:
            return await self._redis.eval(script, len(keys), *keys, *args)
        except Exception as exc:
            logger.error("Redis admission control failed: %s", type(exc).__name__)
            raise RateLimitUnavailable(
                "Request admission is temporarily unavailable. Please try again."
            ) from exc

    async def charge(self, identity: str, policy_names: Iterable[str]) -> None:
        if not self.enabled:
            return
        policies = [RATE_POLICIES[name] for name in dict.fromkeys(policy_names)]
        if not policies:
            return
        result = await self._eval(
            _TOKEN_BUCKET_SCRIPT,
            self._rate_keys(identity, policies),
            [
                value
                for policy in policies
                for value in (policy.capacity, policy.refill_period_s * 1000)
            ],
        )
        if not int(result[0]):
            raise RateLimitExceeded(
                float(result[1]) / 1000,
                "Too many costly requests. Please wait before trying again.",
            )

    async def _acquire(
        self, policies: list[ConcurrencyPolicy], token: str
    ) -> None:
        result = await self._eval(
            _ACQUIRE_CONCURRENCY_SCRIPT,
            self._concurrency_keys(policies),
            [
                token,
                *[
                    value
                    for policy in policies
                    for value in (policy.limit, policy.lease_s * 1000)
                ],
            ],
        )
        if not int(result[0]):
            raise RateLimitExceeded(
                float(result[1]) / 1000,
                "The service is busy. Please wait before trying again.",
            )

    async def _renew(self, policies: list[ConcurrencyPolicy], token: str) -> None:
        await self._eval(
            _RENEW_CONCURRENCY_SCRIPT,
            self._concurrency_keys(policies),
            [token, *[policy.lease_s * 1000 for policy in policies]],
        )

    async def _release(self, policies: list[ConcurrencyPolicy], token: str) -> None:
        await self._eval(
            _RELEASE_CONCURRENCY_SCRIPT,
            self._concurrency_keys(policies),
            [token],
        )

    async def _heartbeat(
        self, policies: list[ConcurrencyPolicy], token: str
    ) -> None:
        interval = max(1, min(policy.lease_s for policy in policies) / 3)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._renew(policies, token)
            except RateLimitUnavailable:
                # Already-admitted work should finish. Expiry prevents a permanent
                # slot leak if this process dies while Redis is unavailable.
                logger.warning("Could not renew a concurrency lease.")

    @asynccontextmanager
    async def admit(
        self,
        identity: str,
        *,
        rates: Iterable[str] = (),
        concurrency: Iterable[str] = (),
    ) -> AsyncIterator[None]:
        """Reserve global capacity, then atomically charge every user bucket."""
        if not self.enabled:
            yield
            return
        policies = [
            CONCURRENCY_POLICIES[name]
            for name in dict.fromkeys(concurrency)
        ]
        if not policies:
            await self.charge(identity, rates)
            yield
            return

        token = str(uuid4())
        await self._acquire(policies, token)
        try:
            await self.charge(identity, rates)
        except Exception:
            try:
                await self._release(policies, token)
            except RateLimitUnavailable:
                logger.warning("Concurrency lease release failed; it will expire.")
            raise
        heartbeat = asyncio.create_task(self._heartbeat(policies, token))
        try:
            yield
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            try:
                await self._release(policies, token)
            except RateLimitUnavailable:
                logger.warning("Concurrency lease release failed; it will expire.")


_limiter: DistributedRateLimiter | None = None


def get_rate_limiter() -> DistributedRateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = DistributedRateLimiter()
    return _limiter
