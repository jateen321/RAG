import os
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import api
from auth import AuthenticatedUser, get_current_user, get_optional_user
from rate_limit import (
    DistributedRateLimiter,
    RateLimitExceeded,
    RateLimitUnavailable,
    _identity_tag,
    _positive_int,
    _positive_number,
)


class StubLimiter:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    @asynccontextmanager
    async def admit(self, identity, *, rates=(), concurrency=()):
        self.calls.append((identity, tuple(rates), tuple(concurrency)))
        if self.error:
            raise self.error
        yield


class DistributedRateLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_limiter_does_not_contact_redis(self):
        redis = AsyncMock()
        limiter = DistributedRateLimiter(redis, enabled=False)

        async with limiter.admit(
            "firebase-user", rates=("ask",), concurrency=("interactive",)
        ):
            pass

        redis.eval.assert_not_called()

    async def test_all_selected_rate_buckets_are_charged_atomically(self):
        redis = AsyncMock()
        redis.eval.return_value = [1, 0]
        limiter = DistributedRateLimiter(redis, enabled=True)

        await limiter.charge("firebase-user", ("ask", "web", "image", "web"))

        args = redis.eval.await_args.args
        self.assertEqual(args[1], 3)
        self.assertEqual(len(args[2:5]), 3)
        self.assertIn(3, args[5:])
        self.assertIn(2, args[5:])
        self.assertIn(1, args[5:])

    async def test_rate_rejection_rounds_retry_after_up(self):
        redis = AsyncMock()
        redis.eval.return_value = [0, 1201]
        limiter = DistributedRateLimiter(redis, enabled=True)

        with self.assertRaises(RateLimitExceeded) as raised:
            await limiter.charge("firebase-user", ("image",))

        self.assertEqual(raised.exception.retry_after, 2)

    async def test_concurrency_lease_is_released(self):
        redis = AsyncMock()
        redis.eval.return_value = [1, 0]
        limiter = DistributedRateLimiter(redis, enabled=True)

        async with limiter.admit(
            "firebase-user", concurrency=("interactive", "web")
        ):
            self.assertEqual(redis.eval.await_count, 1)

        self.assertEqual(redis.eval.await_count, 2)
        release_args = redis.eval.await_args.args
        self.assertEqual(release_args[1], 2)

    async def test_busy_request_does_not_consume_a_rate_token(self):
        redis = AsyncMock()
        redis.eval.return_value = [0, 5000]
        limiter = DistributedRateLimiter(redis, enabled=True)

        with self.assertRaises(RateLimitExceeded):
            async with limiter.admit(
                "firebase-user", rates=("ask",), concurrency=("interactive",)
            ):
                pass

        redis.eval.assert_awaited_once()

    async def test_rate_rejection_releases_reserved_concurrency(self):
        redis = AsyncMock()
        redis.eval.side_effect = ([1, 0], [0, 5000], [1, 0])
        limiter = DistributedRateLimiter(redis, enabled=True)

        with self.assertRaises(RateLimitExceeded):
            async with limiter.admit(
                "firebase-user", rates=("ask",), concurrency=("interactive",)
            ):
                pass

        self.assertEqual(redis.eval.await_count, 3)

    async def test_redis_failure_fails_closed(self):
        redis = AsyncMock()
        redis.eval.side_effect = ConnectionError("secret connection details")
        limiter = DistributedRateLimiter(redis, enabled=True)

        with self.assertRaisesRegex(RateLimitUnavailable, "temporarily unavailable"):
            await limiter.charge("firebase-user", ("ask",))

    def test_identity_is_hashed_before_becoming_a_key(self):
        identity = "firebase-user@example.com/{unsafe}"
        tag = _identity_tag(identity)

        self.assertNotIn(identity, tag)
        self.assertEqual(len(tag), 64)

    def test_enabled_limiter_requires_redis_url(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "REDIS_URL is required"):
                DistributedRateLimiter(enabled=True)

    def test_policy_environment_rejects_fractional_integer_and_nan(self):
        with patch.dict(os.environ, {"TEST_LIMIT": "0.5"}):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                _positive_int("TEST_LIMIT", 1)
        with patch.dict(os.environ, {"TEST_LIMIT": "nan"}):
            with self.assertRaisesRegex(ValueError, "finite number"):
                _positive_number("TEST_LIMIT", 1)


class ApiRateLimitResponseTests(unittest.TestCase):
    def setUp(self):
        api.app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
            uid="firebase-user"
        )
        api.app.dependency_overrides[get_optional_user] = lambda: AuthenticatedUser(
            uid="firebase-user"
        )
        self.client = TestClient(api.app)

    def tearDown(self):
        api.app.dependency_overrides.clear()

    def test_combined_modes_return_429_and_charge_every_bucket(self):
        limiter = StubLimiter(
            RateLimitExceeded(7.1, "Too many costly requests. Please wait.")
        )
        with patch.object(api, "get_rate_limiter", return_value=limiter):
            response = self.client.post(
                "/ask",
                json={
                    "question": "Current information with a visual",
                    "use_web": True,
                    "generate_image": True,
                },
            )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "8")
        self.assertEqual(
            limiter.calls,
            [(
                "firebase-user",
                ("ask", "web", "image"),
                ("interactive", "web", "image"),
            )],
        )

    def test_cors_exposes_retry_after_to_the_browser(self):
        limiter = StubLimiter(RateLimitExceeded(3, "Please wait."))
        with patch.object(api, "get_rate_limiter", return_value=limiter):
            response = self.client.post(
                "/ask",
                headers={"Origin": "http://localhost:3000"},
                json={"question": "Explain this"},
            )

        exposed = response.headers["Access-Control-Expose-Headers"].lower()
        self.assertIn("retry-after", exposed)

    def test_redis_admission_failure_returns_503(self):
        limiter = StubLimiter(
            RateLimitUnavailable(
                "Request admission is temporarily unavailable. Please try again."
            )
        )
        with patch.object(api, "get_rate_limiter", return_value=limiter):
            response = self.client.post(
                "/ask", json={"question": "Explain the indexed text"}
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Request admission is temporarily unavailable. Please try again.",
        )

    def test_missing_redis_configuration_returns_safe_503(self):
        with patch.object(
            api,
            "get_rate_limiter",
            side_effect=RuntimeError("rediss://secret@example.invalid"),
        ):
            response = self.client.post(
                "/ask", json={"question": "Explain the indexed text"}
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Request admission is not configured. Please try again later.",
        )
        self.assertNotIn("secret", response.text)


if __name__ == "__main__":
    unittest.main()
