#!/usr/bin/env python3
"""Run the Gyaan Sarthi backend and frontend as one reloadable dev application.

Python and frontend source changes are handled by Uvicorn's reloader and
Vinext/Vite HMR. This supervisor handles the changes those native watchers do
not reliably cover: Git branch switches, environment files, and frontend
dependency manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
# On this macOS environment ``localhost`` resolves to the IPv6 loopback first.
# Bind both services there so the documented localhost URLs are reachable and
# remain one cookie site.
HOST = "::1"
BACKEND_PORT = 8000
FRONTEND_PORT = 3000
LOCAL_API_URL = f"http://localhost:{BACKEND_PORT}"

# Development is intentionally plain HTTP. Keep this override inside the local
# supervisor so production continues to default to Secure session cookies.
BACKEND_ENVIRONMENT = {"SESSION_COOKIE_SECURE": "0"}
FRONTEND_ENVIRONMENT = {"NEXT_PUBLIC_RAG_API_URL": LOCAL_API_URL}


def _fingerprint(path: Path) -> str | None:
    """Return a content fingerprint, or None when the marker does not exist."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _git_directory(root: Path) -> Path:
    """Resolve both normal repositories and Git worktree .git pointer files."""
    dot_git = root / ".git"
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():
        prefix, separator, value = dot_git.read_text(encoding="utf-8").partition(":")
        if separator and prefix.strip().casefold() == "gitdir":
            target = Path(value.strip())
            return target if target.is_absolute() else (root / target).resolve()
    return dot_git


@dataclass(frozen=True)
class WatchState:
    git_head: str | None
    backend_env: str | None
    frontend_env: tuple[str | None, ...]
    frontend_dependencies: tuple[str | None, ...]


@dataclass(frozen=True)
class RestartPlan:
    backend: bool
    frontend: bool
    reasons: tuple[str, ...]


def capture_watch_state(root: Path = PROJECT_ROOT) -> WatchState:
    """Capture small control files whose changes require a process restart."""
    frontend = root / "frontend"
    return WatchState(
        git_head=_fingerprint(_git_directory(root) / "HEAD"),
        backend_env=_fingerprint(root / ".env"),
        frontend_env=tuple(
            _fingerprint(frontend / name)
            for name in (".env", ".env.local", ".env.development")
        ),
        frontend_dependencies=tuple(
            _fingerprint(frontend / name)
            for name in ("package.json", "package-lock.json")
        ),
    )


def classify_changes(previous: WatchState, current: WatchState) -> RestartPlan:
    """Map marker changes to the minimum safe service restart."""
    backend = False
    frontend = False
    reasons: list[str] = []

    if previous.git_head != current.git_head:
        backend = True
        frontend = True
        reasons.append("Git branch changed")
    if previous.backend_env != current.backend_env:
        backend = True
        reasons.append(".env changed")
    if previous.frontend_env != current.frontend_env:
        frontend = True
        reasons.append("frontend environment changed")
    if previous.frontend_dependencies != current.frontend_dependencies:
        frontend = True
        reasons.append("frontend dependency manifest changed")

    return RestartPlan(backend, frontend, tuple(reasons))


def _port_available(host: str, port: int) -> bool:
    """Return True when a local TCP port can be bound."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _backend_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "api:app",
        "--host",
        HOST,
        "--port",
        str(BACKEND_PORT),
        "--reload",
        "--reload-dir",
        str(PROJECT_ROOT),
    ]


def _frontend_command() -> list[str]:
    return [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        HOST,
        "--port",
        str(FRONTEND_PORT),
    ]


@dataclass
class ManagedService:
    name: str
    command: list[str]
    cwd: Path
    environment_overrides: dict[str, str] | None = None
    process: subprocess.Popen | None = None

    def start(self) -> None:
        print(f"\n▶ Starting {self.name}: {' '.join(self.command)}", flush=True)
        kwargs = {"cwd": str(self.cwd)}
        if self.environment_overrides:
            kwargs["env"] = {**os.environ, **self.environment_overrides}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self.process = subprocess.Popen(self.command, **kwargs)

    def stop(self, timeout: float = 8.0) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            self.process = None
            return

        print(f"\n■ Stopping {self.name}...", flush=True)
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        self.process = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def exit_code(self) -> int | None:
        return None if self.process is None else self.process.poll()


def _check_prerequisites() -> list[str]:
    errors: list[str] = []
    try:
        __import__("uvicorn")
    except ImportError:
        errors.append(
            f"Uvicorn is missing from {sys.executable}; run "
            "`.venv/bin/pip install -r requirements.txt`."
        )
    if shutil.which("npm") is None:
        errors.append("npm is not available on PATH; install the Node.js version in frontend/package.json.")
    if not (FRONTEND_ROOT / "node_modules").is_dir():
        errors.append("frontend/node_modules is missing; run `cd frontend && npm install`.")
    return errors


def _occupied_port_errors() -> list[str]:
    errors = []
    for name, port in (("frontend", FRONTEND_PORT), ("backend", BACKEND_PORT)):
        if not _port_available(HOST, port):
            errors.append(
                f"Port {port} is already occupied ({name}). Stop its existing "
                "server with Ctrl+C, or inspect it with "
                f"`lsof -nP -iTCP:{port} -sTCP:LISTEN`."
            )
    return errors


_shutdown_requested = False


def _request_shutdown(_signum=None, _frame=None) -> None:
    global _shutdown_requested
    _shutdown_requested = True


def run(poll_interval: float = 1.0) -> int:
    """Start both services and supervise restart-worthy control files."""
    errors = [*_check_prerequisites(), *_occupied_port_errors()]
    if errors:
        print("Cannot start the development application:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    backend = ManagedService(
        "FastAPI backend",
        _backend_command(),
        PROJECT_ROOT,
        environment_overrides=BACKEND_ENVIRONMENT,
    )
    frontend = ManagedService(
        "React frontend",
        _frontend_command(),
        FRONTEND_ROOT,
        environment_overrides=FRONTEND_ENVIRONMENT,
    )
    services = (backend, frontend)
    watch_state = capture_watch_state()

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    try:
        backend.start()
        frontend.start()
        print(
            "\n✅ Gyaan Sarthi development servers are running\n"
            f"   Frontend: http://localhost:{FRONTEND_PORT}\n"
            f"   Backend:  {LOCAL_API_URL}\n"
            "   Press Ctrl+C to stop both.",
            flush=True,
        )

        while not _shutdown_requested:
            for service in services:
                code = service.exit_code()
                if code is not None:
                    print(
                        f"\n❌ {service.name} exited unexpectedly with code {code}.",
                        file=sys.stderr,
                    )
                    return code or 1

            time.sleep(poll_interval)
            current = capture_watch_state()
            plan = classify_changes(watch_state, current)
            watch_state = current
            if not plan.reasons:
                continue

            print(f"\n↻ {'; '.join(plan.reasons)}", flush=True)
            if plan.backend and plan.frontend:
                # Stop both first so a branch switch never leaves mixed versions live.
                frontend.stop()
                backend.stop()
                backend.start()
                frontend.start()
            elif plan.backend:
                backend.restart()
            elif plan.frontend:
                frontend.restart()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        frontend.stop()
        backend.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Gyaan Sarthi backend and frontend with automatic reloads."
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between Git/config marker checks (default: 1.0).",
    )
    args = parser.parse_args()
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be greater than zero")
    return run(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
