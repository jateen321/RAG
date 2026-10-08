import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
import auth
import index_jobs


class AllowAdmission:
    @asynccontextmanager
    async def admit(self, identity, *, rates=(), concurrency=()):
        yield


class IndexJobApiTests(unittest.TestCase):
    def setUp(self):
        self.redis_state = patch.object(index_jobs, "_redis_checked", True)
        self.redis_client = patch.object(index_jobs, "_redis", None)
        self.redis_state.start()
        self.redis_client.start()
        index_jobs.reset()
        self.user = auth.AuthenticatedUser(uid="user-a")
        api.app.dependency_overrides[auth.get_current_user] = lambda: self.user
        api.app.dependency_overrides[auth.require_admin] = lambda: auth.AuthenticatedUser(uid="admin-a", is_admin=True)
        self.admission = patch.object(api, "get_rate_limiter", return_value=AllowAdmission())
        self.admission.start()
        self.client = TestClient(api.app)
        self.addCleanup(api.app.dependency_overrides.clear)
        self.addCleanup(index_jobs.reset)
        self.addCleanup(self.admission.stop)
        self.addCleanup(self.redis_state.stop)
        self.addCleanup(self.redis_client.stop)

    def test_youtube_returns_202_and_enqueue_payload_without_inline_work(self):
        with patch.object(api, "_dispatch_job") as dispatch, patch("youtube_ingester.ingest_youtube") as ingest:
            response = self.client.post("/index/youtube", json={"url": "https://www.youtube.com/watch?v=video123"})
        self.assertEqual(response.status_code, 202)
        job = response.json()["job"]
        self.assertEqual(job["status"], "queued")
        self.assertEqual(response.headers["Location"], f"/index/jobs/{job['job_id']}")
        dispatch.assert_called_once_with(self.user.uid, job["job_id"])
        ingest.assert_not_called()

    def test_upload_streams_to_disk_then_enqueues_child_job(self):
        with tempfile.TemporaryDirectory() as data_dir:
            tenant = Path(data_dir) / "tenant"
            with patch.object(api, "_tenant_data_root", return_value=tenant), patch.object(api, "_dispatch_job") as dispatch:
                response = self.client.post("/upload", files={"file": ("notes.txt", b"notes", "text/plain")})
            self.assertEqual(response.status_code, 202)
            job = response.json()["job"]
            self.assertEqual((tenant / "notes.txt").read_bytes(), b"notes")
            dispatch.assert_called_once_with(self.user.uid, job["job_id"])
            self.assertEqual(self.client.get(f"/index/jobs/{job['job_id']}").json()["job"]["status"], "queued")

    def test_upload_job_polling_is_private_to_its_tenant(self):
        with (
            patch.object(api, "_dispatch_job"),
            tempfile.TemporaryDirectory() as data_dir,
            patch.object(api, "_tenant_data_root", return_value=Path(data_dir)),
        ):
            response = self.client.post(
                "/upload", files={"file": ("a.txt", b"a", "text/plain")}
            )
        self.assertEqual(response.status_code, 202)
        job = response.json()["job"]
        api.app.dependency_overrides[auth.get_current_user] = lambda: auth.AuthenticatedUser(uid="user-b")
        self.assertEqual(
            self.client.get(f"/index/jobs/{job['job_id']}").status_code, 404
        )

    def test_folder_validates_allowlist_before_queueing(self):
        admin = auth.AuthenticatedUser(uid="admin-a", is_admin=True)
        api.app.dependency_overrides[auth.get_current_user] = lambda: admin
        with tempfile.TemporaryDirectory() as allowed:
            with patch.object(api, "INDEX_FOLDER_ROOTS", [allowed]), patch.object(api, "_dispatch_job") as dispatch:
                response = self.client.post("/index/folder", json={"folder_path": allowed, "recursive": True})
        self.assertEqual(response.status_code, 202)
        dispatch.assert_called_once()

    def test_admin_bypass_is_server_derived(self):
        admin = auth.AuthenticatedUser(uid="admin-a", is_admin=True)
        api.app.dependency_overrides[auth.get_current_user] = lambda: admin
        with patch.object(api, "_dispatch_job"):
            response = self.client.post("/index/youtube", json={"url": "https://youtu.be/video123"})
        job = response.json()["job"]
        self.assertTrue(index_jobs.get_payload(admin.uid, job["job_id"])["rate_limit_bypass"])


if __name__ == "__main__":
    unittest.main()
