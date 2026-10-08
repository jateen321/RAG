import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

import index_jobs
import worker


class DurableJobFallbackTests(unittest.TestCase):
    def setUp(self):
        # Force the explicitly documented local fallback without requiring Redis.
        self.redis_state = patch.object(index_jobs, "_redis_checked", True)
        self.redis_client = patch.object(index_jobs, "_redis", None)
        self.redis_state.start()
        self.redis_client.start()
        index_jobs.reset()
        self.addCleanup(index_jobs.reset)
        self.addCleanup(self.redis_state.stop)
        self.addCleanup(self.redis_client.stop)

    def test_jobs_are_tenant_isolated(self):
        job = index_jobs.create("user-a", "youtube", "url")
        self.assertEqual(index_jobs.get("user-a", job["job_id"])["job_id"], job["job_id"])
        self.assertIsNone(index_jobs.get("user-b", job["job_id"]))

    def test_active_slot_is_released_when_job_finishes(self):
        job = index_jobs.create(
            "user-a",
            "file",
            "notes.txt",
            total=1,
            payload={"kind": "file", "path": "/data/notes.txt"},
        )
        self.assertTrue(index_jobs.is_running("user-a"))
        with self.assertRaises(index_jobs.ActiveJobError):
            index_jobs.create("user-a", "youtube", "another")
        index_jobs.mark_running("user-a", job["job_id"])
        index_jobs.update_progress(
            "user-a", job["job_id"], {"total": 1, "indexed": 1}
        )
        index_jobs.finish(
            "user-a", job["job_id"], "finished", result={"source": "notes.txt"}
        )
        self.assertEqual(index_jobs.current("user-a")["status"], "finished")
        self.assertFalse(index_jobs.is_running("user-a"))

    def test_configured_redis_failure_never_falls_back_to_process_memory(self):
        with (
            patch.dict(os.environ, {"REDIS_URL": "redis://unavailable:6379/0"}),
            patch.object(index_jobs, "_redis_checked", False),
            patch.object(index_jobs, "_redis", None),
            patch("redis.Redis.from_url", side_effect=ConnectionError("offline")) as connect,
        ):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "Redis is unavailable"):
                    index_jobs.create("user-a", "youtube", "url")
        self.assertEqual(connect.call_count, 2)


class WorkerConfigurationTests(unittest.TestCase):
    def test_timeout_never_drops_below_one_hour(self):
        with patch.dict(os.environ, {"RAG_INDEX_JOB_TIMEOUT_S": "30"}):
            self.assertEqual(worker.queue_timeout(), 3600)
        with patch.dict(os.environ, {"RAG_INDEX_JOB_TIMEOUT_S": "7200"}):
            self.assertEqual(worker.queue_timeout(), 7200)

    def test_success_records_result_and_ordinary_user_takes_lease(self):
        calls = []

        class Limiter:
            @asynccontextmanager
            async def admit(self, identity, *, rates=(), concurrency=()):
                calls.append((identity, tuple(rates), tuple(concurrency)))
                yield

        with (
            patch.object(index_jobs, "mark_running", return_value={"job_id": "j"}),
            patch.object(index_jobs, "get_payload", return_value={"kind": "youtube"}),
            patch.object(index_jobs, "finish", return_value={"status": "finished"}) as finish,
            patch.object(worker, "_run", return_value={"videos_indexed": 1}),
            patch("rate_limit.get_rate_limiter", return_value=Limiter()),
        ):
            worker.process_job("student", "j")
        self.assertEqual(calls, [("student", (), ("ingest",))])
        finish.assert_called_once()

    def test_admin_bypass_does_not_take_api_concurrency_lease_and_cleans_failed_upload(self):
        with tempfile.NamedTemporaryFile() as uploaded:
            payload = {"kind": "file", "path": uploaded.name, "created_by_job": True, "rate_limit_bypass": True}
            with (
                patch.object(index_jobs, "mark_running", return_value={"job_id": "j"}),
                patch.object(index_jobs, "get_payload", return_value=payload),
                patch.object(index_jobs, "finish") as finish,
                patch.object(worker, "_run", side_effect=ValueError("bad document")),
                patch("rate_limit.get_rate_limiter") as limiter,
            ):
                with self.assertRaises(ValueError):
                    worker.process_job("admin", "j")
            limiter.return_value.admit.assert_not_called()
            finish.assert_called_once_with("admin", "j", "failed", "bad document")
            self.assertFalse(os.path.exists(uploaded.name))

    def test_duplicate_new_upload_is_removed_but_existing_copy_is_not(self):
        with tempfile.NamedTemporaryFile() as uploaded:
            payload = {"kind": "file", "path": uploaded.name, "created_by_job": True, "rate_limit_bypass": True}
            with (
                patch.object(index_jobs, "mark_running", return_value={"job_id": "j"}),
                patch.object(index_jobs, "get_payload", return_value=payload),
                patch.object(index_jobs, "update_progress"),
                patch.object(index_jobs, "finish", return_value={"status": "finished"}),
                patch.object(worker, "_run", return_value={"status": "already_indexed", "source": "notes.txt"}),
            ):
                worker.process_job("admin", "j")
            self.assertFalse(os.path.exists(uploaded.name))

        with tempfile.NamedTemporaryFile() as existing:
            payload = {"kind": "file", "path": existing.name, "created_by_job": False, "rate_limit_bypass": True}
            with (
                patch.object(index_jobs, "mark_running", return_value={"job_id": "j"}),
                patch.object(index_jobs, "get_payload", return_value=payload),
                patch.object(index_jobs, "update_progress"),
                patch.object(index_jobs, "finish", return_value={"status": "finished"}),
                patch.object(worker, "_run", return_value={"status": "already_indexed", "source": "notes.txt"}),
            ):
                worker.process_job("admin", "j")
            self.assertTrue(os.path.exists(existing.name))

    def test_rq_callbacks_release_failed_or_stopped_jobs(self):
        rq_job = type("RQJob", (), {"args": ("student", "job-1")})()
        with patch.object(index_jobs, "finish") as finish:
            worker.record_rq_failure(
                rq_job, None, RuntimeError, RuntimeError("worker crashed"), None
            )
            finish.assert_called_once_with(
                "student", "job-1", "failed", "worker crashed"
            )

        with patch.object(index_jobs, "finish") as finish:
            worker.record_rq_stop(rq_job, None)
            finish.assert_called_once_with(
                "student",
                "job-1",
                "failed",
                "The indexing worker stopped before the import completed.",
            )


if __name__ == "__main__":
    unittest.main()
