"""RQ entrypoint for durable document, folder, and YouTube ingestion."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import index_jobs


def queue_timeout() -> int:
    """RQ jobs must outlive the slowest OCR/embedding import."""
    try:
        configured = int(os.getenv("RAG_INDEX_JOB_TIMEOUT_S", "3600"))
    except ValueError:
        configured = 3600
    return max(3600, configured)


def _queue():
    from redis import Redis
    from rq import Queue

    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        raise RuntimeError("REDIS_URL is required for the indexing worker.")
    # RQ stores pickled job data in Redis; response decoding corrupts those
    # bytes.  Job-state Redis in index_jobs.py is separate and text-decoded.
    connection = Redis.from_url(url, decode_responses=False, socket_connect_timeout=2, socket_timeout=2)
    return Queue(os.getenv("RAG_INDEX_QUEUE", "indexing"), connection=connection, default_timeout=queue_timeout())


def enqueue(owner_id: str, job_id: str):
    """Enqueue by import path; arguments are stable scalar server IDs only."""
    return _queue().enqueue(
        "worker.process_job",
        owner_id,
        job_id,
        job_id=f"index-{job_id}",
        job_timeout=queue_timeout(),
        result_ttl=24 * 3600,
        failure_ttl=7 * 24 * 3600,
        on_failure=record_rq_failure,
        on_stopped=record_rq_stop,
    )


def _rq_job_ids(rq_job) -> tuple[str, str] | None:
    args = tuple(getattr(rq_job, "args", ()) or ())
    if len(args) < 2 or not all(isinstance(value, str) for value in args[:2]):
        return None
    return args[0], args[1]


def record_rq_failure(rq_job, connection, exc_type, exc_value, traceback) -> None:
    """Mirror failures raised outside the task body into application job state."""
    identifiers = _rq_job_ids(rq_job)
    if identifiers:
        owner_id, job_id = identifiers
        index_jobs.finish(owner_id, job_id, "failed", str(exc_value))


def record_rq_stop(rq_job, connection) -> None:
    """Release the tenant slot when RQ stops a job before normal completion."""
    identifiers = _rq_job_ids(rq_job)
    if identifiers:
        owner_id, job_id = identifiers
        index_jobs.finish(owner_id, job_id, "failed", "The indexing worker stopped before the import completed.")


def _index_file(payload: dict, owner_id: str) -> dict:
    from document_ingester import SOURCE_TYPES, extract_document
    from indexer import index_document, is_document_indexed

    path = Path(payload["path"]).resolve()
    source = payload["source_name"]
    extension = payload["extension"]
    if is_document_indexed(source, file_path=path, owner_id=owner_id):
        return {"source": source, "status": "already_indexed", "pages_with_text": 0, "chunks_indexed": 0}
    pages = extract_document(path)
    if not pages:
        raise ValueError("No readable text could be extracted from this document.")
    chunks = index_document(pages, source, SOURCE_TYPES[extension], file_path=path, owner_id=owner_id)
    return {
        "source": source, "status": "indexed" if chunks else "already_indexed",
        "pages_with_text": len(pages), "chunks_indexed": chunks,
    }


def _run(payload: dict, owner_id: str, job_id: str) -> dict:
    kind = payload["kind"]
    if kind == "youtube":
        from youtube_ingester import ingest_youtube
        return ingest_youtube(payload["url"], payload["corpus_owner_id"], lambda progress: index_jobs.update_progress(owner_id, job_id, progress))
    if kind == "file":
        return _index_file(payload, payload["corpus_owner_id"])
    if kind == "folder":
        from document_ingester import index_folder
        return index_folder(
            payload["folder_path"], payload["recursive"], owner_id=payload["corpus_owner_id"],
            on_progress=lambda progress: index_jobs.update_progress(owner_id, job_id, progress),
        )
    raise ValueError(f"Unknown indexing job kind: {kind}")


def process_job(owner_id: str, job_id: str) -> dict:
    """RQ calls this function in a worker process; all state is Redis-backed."""
    job = index_jobs.mark_running(owner_id, job_id)
    if job is None:
        raise RuntimeError("Indexing job state no longer exists.")
    payload = index_jobs.get_payload(owner_id, job_id) or {}
    created_by_job = bool(payload.get("created_by_job"))
    path = Path(payload["path"]) if payload.get("path") else None
    try:
        from rate_limit import get_rate_limiter

        async def run_with_lease():
            limiter = get_rate_limiter()
            if payload.get("rate_limit_bypass") is True:
                return _run(payload, owner_id, job_id)
            async with limiter.admit(owner_id, rates=(), concurrency=("ingest",)):
                return _run(payload, owner_id, job_id)

        result = asyncio.run(run_with_lease())
    except Exception as exc:
        if created_by_job and path is not None:
            path.unlink(missing_ok=True)
        index_jobs.finish(owner_id, job_id, "failed", str(exc))
        raise

    status = "finished"
    if payload.get("kind") == "file":
        outcome = "indexed" if result.get("status") == "indexed" else "already_indexed"
        index_jobs.update_progress(owner_id, job_id, {outcome: 1, "total": 1})
        # A duplicate upload can be detected only after bytes are persisted.
        # Remove that newly-created copy, but never remove a pre-existing file
        # that the request deliberately reused.
        if result.get("status") == "already_indexed" and created_by_job and path is not None:
            path.unlink(missing_ok=True)
    elif payload.get("kind") == "folder":
        index_jobs.update_progress(owner_id, job_id, {
            "total": result.get("files_found", 0),
            "indexed": result.get("files_indexed", 0),
            "already_indexed": 0,
            "skipped": result.get("files_skipped", 0) + result.get("files_failed", 0),
        })
    message = _message(result, payload.get("kind"))
    final = index_jobs.finish(owner_id, job_id, status, message, result=result)
    return final or result


def _message(result: dict, kind: str | None) -> str:
    if kind == "youtube":
        return f"{result.get('videos_indexed', 0)} indexed, {result.get('videos_already_indexed', 0)} already indexed, {result.get('videos_skipped', 0)} skipped."
    if kind == "folder":
        return f"{result.get('files_indexed', 0)} indexed, {result.get('files_skipped', 0)} skipped, {result.get('files_failed', 0)} failed."
    return f"{result.get('source', 'Document')} indexed."


def main() -> None:
    from rq import Worker

    queue = _queue()
    Worker([queue], connection=queue.connection).work()


if __name__ == "__main__":
    main()
