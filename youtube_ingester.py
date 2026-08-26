"""YouTube video/playlist metadata and transcript ingestion.

Media is never downloaded. yt-dlp resolves video/playlist metadata and
youtube-transcript-api fetches timestamped captions without a YouTube API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re
from urllib.parse import parse_qs, urlparse

from config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_CHUNK_OVERLAP, MIN_CHUNK_LENGTH
from indexer import _content_hash, index_chunks


YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
PREFERRED_LANGUAGES = ("hi", "en")
_HTML_TAG = re.compile(r"<[^>]+>")


@dataclass
class VideoResult:
    video_id: str
    title: str
    status: str
    chunks_indexed: int = 0
    reason: str | None = None


def validate_youtube_url(url: str) -> str:
    """Return a normalized URL or raise for unsupported/malformed input."""
    url = url.strip()
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in YOUTUBE_HOSTS:
        raise ValueError("Provide a valid youtube.com or youtu.be video/playlist URL.")
    query = parse_qs(parsed.query)
    path_parts = [part for part in parsed.path.split("/") if part]
    is_short = host == "youtu.be" and bool(path_parts)
    is_watch = parsed.path.rstrip("/") == "/watch" and bool(query.get("v"))
    is_playlist = parsed.path.rstrip("/") == "/playlist" and bool(query.get("list"))
    is_short_form = len(path_parts) >= 2 and path_parts[0] in {"shorts", "live"}
    if not (is_short or is_watch or is_playlist or is_short_form):
        raise ValueError("Only YouTube video and playlist URLs are supported.")
    return url


def _language_match(code: str, preferred: str) -> bool:
    code = code.casefold()
    return code == preferred or code.startswith(f"{preferred}-")


def _select_transcript(transcript_list):
    """Manual before generated; within each kind prefer Hindi then English."""
    available = list(transcript_list)
    for generated in (False, True):
        candidates = [t for t in available if bool(t.is_generated) is generated]
        for preferred in PREFERRED_LANGUAGES:
            match = next(
                (t for t in candidates if _language_match(t.language_code, preferred)),
                None,
            )
            if match:
                return match
        if candidates:
            return candidates[0]
    raise ValueError("No transcript is available for this video.")


def _clean_snippet(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(_HTML_TAG.sub("", text))).strip()


def _timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _transcript_chunks(snippets) -> list[dict]:
    """Pack complete caption snippets while preserving time boundaries."""
    units = []
    for snippet in snippets:
        text = _clean_snippet(snippet.text)
        if text:
            units.append({
                "text": text,
                "start": float(snippet.start),
                "end": float(snippet.start + snippet.duration),
            })

    chunks: list[dict] = []
    current: list[dict] = []

    def size(items: list[dict]) -> int:
        return sum(len(item["text"]) for item in items) + max(0, len(items) - 1)

    def flush() -> None:
        text = " ".join(item["text"] for item in current).strip()
        if len(text) >= MIN_CHUNK_LENGTH:
            chunks.append({
                "text": text,
                "chunk_index": len(chunks),
                "content_hash": _content_hash(text),
                "start_seconds": round(current[0]["start"], 3),
                "end_seconds": round(current[-1]["end"], 3),
                "timestamp": _timestamp(current[0]["start"]),
                "extraction_method": "youtube_transcript",
            })

    for unit in units:
        if current and size(current) + len(unit["text"]) + 1 > CHUNK_SIZE:
            flush()
            overlap: list[dict] = []
            for previous in reversed(current[1:]):
                proposed = [previous, *overlap]
                if size(proposed) > MAX_CHUNK_OVERLAP:
                    break
                overlap = proposed
                if size(overlap) >= CHUNK_OVERLAP:
                    break
            current = overlap
        current.append(unit)
    if current:
        flush()
    return chunks


def _source_name(title: str, video_id: str) -> str:
    return f"YouTube: {title} [{video_id}]"


def _video_metadata(info: dict, playlist: dict | None) -> dict:
    video_id = str(info.get("id") or "")
    url = info.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"
    metadata = {
        "video_id": video_id,
        "video_title": info.get("title") or video_id,
        "channel_name": info.get("channel") or info.get("uploader") or "unknown",
        "channel_id": info.get("channel_id") or info.get("uploader_id"),
        "duration_seconds": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "source_url": url,
    }
    if playlist:
        metadata.update({
            "playlist_id": playlist.get("id"),
            "playlist_title": playlist.get("title"),
            "playlist_index": info.get("playlist_index"),
            "playlist_url": playlist.get("webpage_url"),
        })
    return metadata


def _index_video(info: dict, playlist: dict | None = None) -> VideoResult:
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = str(info.get("id") or "")
    title = str(info.get("title") or video_id or "Unknown video")
    if not video_id:
        raise ValueError("YouTube returned a video without an ID.")

    transcript = _select_transcript(YouTubeTranscriptApi().list(video_id))
    fetched = transcript.fetch()
    chunks = _transcript_chunks(fetched)
    if not chunks:
        raise ValueError("The selected transcript did not contain usable text.")

    metadata = _video_metadata(info, playlist)
    metadata.update({
        "transcript_language": transcript.language,
        "transcript_language_code": transcript.language_code,
        "transcript_is_generated": bool(transcript.is_generated),
    })
    count = index_chunks(
        chunks,
        _source_name(title, video_id),
        "youtube",
        document_key=video_id,
        source_metadata=metadata,
    )
    return VideoResult(video_id, title, "indexed", count)


def ingest_youtube(url: str) -> dict:
    """Index a YouTube video or playlist and return a structured report."""
    from yt_dlp import YoutubeDL

    url = validate_youtube_url(url)
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "ignoreerrors": True,
    }
    with YoutubeDL(options) as ydl:
        raw = ydl.extract_info(url, download=False)
        info = ydl.sanitize_info(raw) if raw else None
    if not info:
        raise ValueError("YouTube metadata could not be retrieved for this URL.")

    is_playlist = info.get("_type") in {"playlist", "multi_video"}
    entries = list(info.get("entries") or []) if is_playlist else [info]
    if not entries:
        raise ValueError("The playlist contains no accessible videos.")

    results: list[VideoResult] = []
    for entry in entries:
        if not entry:
            results.append(VideoResult("unknown", "Unavailable video", "skipped", reason="Unavailable or private"))
            continue
        try:
            results.append(_index_video(entry, info if is_playlist else None))
        except Exception as exc:
            results.append(VideoResult(
                str(entry.get("id") or "unknown"),
                str(entry.get("title") or "Unknown video"),
                "skipped",
                reason=str(exc),
            ))

    indexed = [result for result in results if result.status == "indexed"]
    skipped = [result for result in results if result.status == "skipped"]
    if not indexed:
        reasons = "; ".join(result.reason or "unknown error" for result in skipped[:3])
        raise RuntimeError(f"No videos were indexed. {reasons}")
    return {
        "source_type": "playlist" if is_playlist else "video",
        "playlist_id": info.get("id") if is_playlist else None,
        "playlist_title": info.get("title") if is_playlist else None,
        "videos_total": len(results),
        "videos_indexed": len(indexed),
        "videos_skipped": len(skipped),
        "chunks_indexed": sum(result.chunks_indexed for result in indexed),
        "results": [result.__dict__ for result in results],
    }
