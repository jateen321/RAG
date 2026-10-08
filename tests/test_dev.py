"""Unit tests for the unified development server supervisor."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dev


class WatchStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / ".git").mkdir()
        (self.root / ".git" / "HEAD").write_text(
            "ref: refs/heads/main\n", encoding="utf-8"
        )
        (self.root / "frontend").mkdir()
        (self.root / "frontend" / "package.json").write_text(
            '{"name":"sarthi-ai"}', encoding="utf-8"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_branch_switch_restarts_both_services(self):
        before = dev.capture_watch_state(self.root)
        (self.root / ".git" / "HEAD").write_text(
            "ref: refs/heads/feature\n", encoding="utf-8"
        )

        plan = dev.classify_changes(before, dev.capture_watch_state(self.root))

        self.assertTrue(plan.backend)
        self.assertTrue(plan.frontend)
        self.assertIn("Git branch changed", plan.reasons)

    def test_backend_environment_change_restarts_only_backend(self):
        before = dev.capture_watch_state(self.root)
        (self.root / ".env").write_text("LLM_BACKEND=developer\n", encoding="utf-8")

        plan = dev.classify_changes(before, dev.capture_watch_state(self.root))

        self.assertTrue(plan.backend)
        self.assertFalse(plan.frontend)

    def test_frontend_environment_and_dependency_changes_restart_frontend(self):
        before = dev.capture_watch_state(self.root)
        (self.root / "frontend" / ".env.local").write_text(
            "NEXT_PUBLIC_API_URL=http://127.0.0.1:8000\n", encoding="utf-8"
        )
        (self.root / "frontend" / "package-lock.json").write_text(
            '{"lockfileVersion":3}', encoding="utf-8"
        )

        plan = dev.classify_changes(before, dev.capture_watch_state(self.root))

        self.assertFalse(plan.backend)
        self.assertTrue(plan.frontend)
        self.assertEqual(len(plan.reasons), 2)

    def test_unwatched_source_file_relies_on_native_reloaders(self):
        before = dev.capture_watch_state(self.root)
        (self.root / "api.py").write_text("app = object()\n", encoding="utf-8")

        plan = dev.classify_changes(before, dev.capture_watch_state(self.root))

        self.assertFalse(plan.backend)
        self.assertFalse(plan.frontend)
        self.assertEqual(plan.reasons, ())

    def test_git_worktree_pointer_is_resolved(self):
        actual_git = self.root / "worktree-git"
        actual_git.mkdir()
        (actual_git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        git_pointer = self.root / ".git"
        (git_pointer / "HEAD").unlink()
        git_pointer.rmdir()
        git_pointer.write_text("gitdir: worktree-git\n", encoding="utf-8")

        state = dev.capture_watch_state(self.root)

        self.assertIsNotNone(state.git_head)


class CommandTests(unittest.TestCase):
    def test_backend_uses_current_python_and_reload(self):
        command = dev._backend_command()
        self.assertEqual(command[0], dev.sys.executable)
        self.assertIn("--reload", command)
        self.assertIn("--reload-dir", command)
        self.assertIn("8000", command)
        self.assertIn("::1", command)

    def test_frontend_uses_fixed_development_port(self):
        command = dev._frontend_command()
        self.assertEqual(command[:3], ["npm", "run", "dev"])
        self.assertIn("3000", command)

    def test_service_environment_applies_overrides_without_losing_parent_values(self):
        service = dev.ManagedService(
            "test",
            ["test-command"],
            Path("."),
            environment_overrides={"SESSION_COOKIE_SECURE": "0"},
        )

        with (
            patch.dict(dev.os.environ, {"PARENT_SETTING": "kept"}, clear=True),
            patch.object(dev.subprocess, "Popen") as popen,
        ):
            service.start()

        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["SESSION_COOKIE_SECURE"], "0")
        self.assertEqual(environment["PARENT_SETTING"], "kept")

    def test_development_auth_uses_one_site_and_allows_http_cookie(self):
        self.assertEqual(dev.BACKEND_ENVIRONMENT["SESSION_COOKIE_SECURE"], "0")
        self.assertEqual(
            dev.FRONTEND_ENVIRONMENT["NEXT_PUBLIC_RAG_API_URL"],
            "http://localhost:8000",
        )


if __name__ == "__main__":
    unittest.main()
