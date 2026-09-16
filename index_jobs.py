"""Durable, tenant-isolated state for background indexing jobs.

Redis is the production source of truth.  A small in-memory fallback is used
only when REDIS_URL is unset, keeping local tests and the CLI useful.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from time import monotonic, time
from uuid import uuid4

ENDED_JOB_RETENTION_S = 24 * 3600
_ENDED_STATUSES = {"finished", "failed"}
_COUNTERS = ("indexed", "already_indexed", "skipped")


class ActiveJobError(Exception):
    """The user already has an active indexing job."""


_lock = threading.RLock()
_jobs: dict[str, dict] = {}
_redis = None
_redis_checked = False


def _owner_tag(uid: str) -> str:
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()


def _owner_key(uid: str) -> str:
    return f"sarthi:index:owner:{_owner_tag(uid)}"


def _active_key(uid: str) -> str:
    return f"sarthi:index:active:{_owner_tag(uid)}"


def _job_key(uid: str, job_id: str) -> str:
    return f"sarthi:index:job:{_owner_tag(uid)}:{job_id}"


def _client():
    global _redis, _redis_checked
    if _redis_checked:
        return _redis
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        _redis_checked = True
        return None
    try:
        from redis import Redis
        client = Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30,
        )
        client.ping()
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The redis package is required for durable indexing jobs.") from exc
    except Exception as exc:
        raise RuntimeError("Redis is unavailable for indexing job state.") from exc
    _redis = client
    _redis_checked = True
    return _redis


def _public(job: dict) -> dict:
    return {
        key: value
        for key, value in job.items()
        if key not in {"payload", "created_at"}
    }


def _new_job(kind, label, total, *, payload=None) -> dict:
    return {
        "job_id": uuid4().hex, "kind": kind, "label": label, "status": "queued",
        "done": 0, "total": total, "indexed": 0, "already_indexed": 0,
        "skipped": 0, "message": None, "result": None,
        "created_at": time(), "payload": payload or {},
    }


def _redis_job(client, uid, job_id):
    raw = client.get(_job_key(uid, job_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _save_redis_job(client, uid, job):
    client.set(_job_key(uid, job["job_id"]), json.dumps(job, ensure_ascii=False), ex=ENDED_JOB_RETENTION_S)
    client.zadd(_owner_key(uid), {job["job_id"]: job["created_at"]})
    client.expire(_owner_key(uid), ENDED_JOB_RETENTION_S)


def _clear_active_job(client, uid: str, job_id: str) -> None:
    client.eval(
        """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
          return redis.call('DEL', KEYS[1])
        end
        return 0
        """,
        1,
        _active_key(uid),
        job_id,
    )


def reset() -> None:
    """Reset the local fallback (test helper)."""
    with _lock:
        _jobs.clear()


def _prune_memory(now: float) -> None:
    for uid, owner in list(_jobs.items()):
        for job_id, record in list(owner.items()):
            if job_id == "active" or record.get("ended") is None:
                continue
            if now - record["ended"] > ENDED_JOB_RETENTION_S:
                owner.pop(job_id, None)
        if not any(key != "active" for key in owner):
            _jobs.pop(uid, None)


def create(uid, kind, label, total=None, *, payload=None) -> dict:
    """Create a job, atomically reserving the tenant's active slot."""
    job = _new_job(kind, label, total, payload=payload)
    client = _client()
    if client is not None:
        if not client.set(
            _active_key(uid), job["job_id"], nx=True, ex=ENDED_JOB_RETENTION_S
        ):
            raise ActiveJobError
        try:
            _save_redis_job(client, uid, job)
        except Exception:
            client.delete(_active_key(uid))
            raise
        return _public(job)
    with _lock:
        _prune_memory(monotonic())
        owner = _jobs.setdefault(uid, {})
        active = owner.get("active")
        if active and owner[active]["job"].get("status") in {"queued", "running"}:
            raise ActiveJobError
        owner["active"] = job["job_id"]
        owner[job["job_id"]] = {"job": job, "updated": monotonic(), "ended": None}
    return _public(job)


def _load(uid, job_id):
    client = _client()
    if client is not None:
        return _redis_job(client, uid, job_id)
    with _lock:
        record = _jobs.get(uid, {}).get(job_id)
        return record["job"] if record else None


def _save(uid, job):
    client = _client()
    if client is not None:
        _save_redis_job(client, uid, job)
    else:
        with _lock:
            record = _jobs.get(uid, {}).get(job["job_id"])
            if record:
                record["job"] = job
                record["updated"] = monotonic()


def get(uid, job_id):
    job = _load(uid, job_id)
    return _public(job) if job else None


def get_payload(uid, job_id):
    job = _load(uid, job_id)
    return job.get("payload") if job else None


def set_payload(uid, job_id, payload) -> bool:
    job = _load(uid, job_id)
    if not job:
        return False
    job["payload"] = payload
    _save(uid, job)
    return True


def is_running(uid) -> bool:
    client = _client()
    if client is not None:
        active_id = client.get(_active_key(uid))
        if not active_id:
            return False
        job = _redis_job(client, uid, active_id)
        if not job or job.get("status") not in {"queued", "running"}:
            client.delete(_active_key(uid))
            return False
        return True
    with _lock:
        _prune_memory(monotonic())
        owner = _jobs.get(uid, {})
        active = owner.get("active")
        return bool(active and owner[active]["job"].get("status") in {"queued", "running"})


def current(uid, job_id=None):
    client = _client()
    if client is not None:
        if job_id is None:
            job_id = client.get(_active_key(uid))
            if not job_id:
                ids = client.zrevrange(_owner_key(uid), 0, 49)
                job_id = ids[0] if ids else None
        job = _redis_job(client, uid, job_id) if job_id else None
        return _public(job) if job else None
    with _lock:
        _prune_memory(monotonic())
        owner = _jobs.get(uid, {})
        job_id = job_id or owner.get("active")
        if not job_id:
            ids = [key for key in owner if key != "active"]
            job_id = ids[-1] if ids else None
        record = owner.get(job_id) if job_id else None
        return _public(record["job"]) if record else None


def mark_running(uid, job_id):
    job = _load(uid, job_id)
    if not job or job.get("status") not in {"queued", "running"}:
        return None
    if job.get("status") == "queued":
        job["status"] = "running"
        _save(uid, job)
    return _public(job)


def update_progress(uid, job_id, progress) -> None:
    job = _load(uid, job_id)
    if not job or job.get("status") not in {"queued", "running"}:
        return
    for key in _COUNTERS:
        if key in progress:
            job[key] = int(progress[key])
    job["done"] = sum(int(job.get(key, 0)) for key in _COUNTERS)
    if progress.get("total") is not None:
        job["total"] = int(progress["total"])
    _save(uid, job)


def finish(uid, job_id, status=None, message=None, *, result=None):
    if status is not None and status not in _ENDED_STATUSES:
        raise ValueError(f"Unknown final job status: {status}")
    job = _load(uid, job_id)
    if not job:
        return None
    if job.get("status") in _ENDED_STATUSES:
        return _public(job)
    if status is None:
        status = "failed" if job.get("done", 0) and job.get("skipped", 0) == job.get("done", 0) else "finished"
    job["status"] = status
    job["message"] = message
    if result is not None:
        job["result"] = result
    _save(uid, job)
    client = _client()
    if client is not None:
        _clear_active_job(client, uid, job_id)
    else:
        with _lock:
            record = _jobs.get(uid, {}).get(job_id)
            if record:
                record["ended"] = monotonic()
            if _jobs.get(uid, {}).get("active") == job_id:
                _jobs[uid].pop("active", None)
    return _public(job)
