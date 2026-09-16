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


class AsyncIndexApiTests(unittest.TestCase):
    def setUp(self):
        index_jobs.reset()
        self.client = TestClient(api.app)
        self.user = auth.AuthenticatedUser(uid="student-async", is_admin=False)
        api.app.dependency_overrides[auth.get_current_user] = lambda: self.user
        api.app.dependency_overrides[auth.require_admin] = lambda: self.user
        self.admission = patch.object(api, "get_rate_limiter", return_value=AllowAdmission())
        self.admission.start()
        self.addCleanup(api.app.dependency_overrides.clear)
        self.addCleanup(self.admission.stop)
        self.addCleanup(index_jobs.reset)

    def test_youtube_returns_queued_job_and_does_not_call_ingester(self):
        with patch.object(api, "_dispatch_job") as dispatch, patch("youtube_ingester.ingest_youtube") as ingest:
            response = self.client.post("/index/youtube", json={"url": "https://www.youtube.com/watch?v=abc"})
        self.assertEqual(response.status_code, 202)
        body = response.json()["job"]
        self.assertEqual(body["status"], "queued")
        self.assertEqual(response.headers["location"], f"/index/jobs/{body['job_id']}")
        dispatch.assert_called_once()
        ingest.assert_not_called()

    def test_upload_persists_before_enqueue_and_returns_child_job(self):
        with tempfile.TemporaryDirectory() as temp:
            tenant = Path(temp) / "tenant"
            with patch.object(api, "_tenant_data_root", return_value=tenant), patch.object(api, "_dispatch_job") as dispatch, patch("document_ingester.extract_document") as extract:
                response = self.client.post("/upload", files={"file": ("notes.txt", b"persist me", "text/plain")})
            self.assertEqual(response.status_code, 202)
            job = response.json()["job"]
            self.assertEqual((tenant / "notes.txt").read_bytes(), b"persist me")
            self.assertEqual(list(tenant.glob(".*.part")), [])
            self.assertEqual(job["status"], "queued")
            dispatch.assert_called_once()
            extract.assert_not_called()

    def test_polling_is_tenant_scoped(self):
        with patch.object(api, "_dispatch_job"):
            response = self.client.post("/index/youtube", json={"url": "https://www.youtube.com/watch?v=abc"})
        job_id = response.json()["job"]["job_id"]
        api.app.dependency_overrides[auth.get_current_user] = lambda: auth.AuthenticatedUser(uid="other-user")
        self.assertEqual(self.client.get(f"/index/jobs/{job_id}").status_code, 404)

    def test_admin_payload_contains_verified_bypass_flag(self):
        admin = auth.AuthenticatedUser(uid="admin-async", is_admin=True)
        api.app.dependency_overrides[auth.get_current_user] = lambda: admin
        with patch.object(api, "_dispatch_job"):
            response = self.client.post("/index/youtube", json={"url": "https://www.youtube.com/watch?v=abc"})
        job_id = response.json()["job"]["job_id"]
        self.assertTrue(index_jobs.get_payload(admin.uid, job_id)["rate_limit_bypass"])

    def test_folder_returns_queued_job_after_allowlist_validation(self):
        with tempfile.TemporaryDirectory() as allowed:
            with patch.object(api, "INDEX_FOLDER_ROOTS", [allowed]), patch.object(api, "_dispatch_job") as dispatch:
                api.app.dependency_overrides[auth.require_admin] = lambda: auth.AuthenticatedUser(uid="admin-folder", is_admin=True)
                response = self.client.post("/index/folder", json={"folder_path": allowed, "recursive": True})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["job"]["status"], "queued")
        dispatch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
