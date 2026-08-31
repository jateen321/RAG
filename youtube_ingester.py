"""YouTube video/playlist metadata and transcript ingestion.

Media is never downloaded. yt-dlp resolves video/playlist metadata and
youtube-transcript-api fetches timestamped captions without a YouTube API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re
from urllib.parse import parse_qs, urlparse

from config import (
    MIN_CHUNK_LENGTH,
    YOUTUBE_CHUNK_MAX_CHARS,
    YOUTUBE_CHUNK_MAX_SECONDS,
    YOUTUBE_CHUNK_OVERLAP_SECONDS,
    YOUTUBE_CHUNK_TARGET_CHARS,
    YOUTUBE_CHUNK_TARGET_SECONDS,
)
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


@dataclass(frozen=True)
class TranscriptChunkConfig:
    """Text and time constraints for transcript chunks."""

    target_chars: int = YOUTUBE_CHUNK_TARGET_CHARS
    max_chars: int = YOUTUBE_CHUNK_MAX_CHARS
    target_seconds: float = YOUTUBE_CHUNK_TARGET_SECONDS
    max_seconds: float = YOUTUBE_CHUNK_MAX_SECONDS
    overlap_seconds: float = YOUTUBE_CHUNK_OVERLAP_SECONDS

    def __post_init__(self) -> None:
        if self.target_chars <= 0 or self.target_seconds <= 0:
            raise ValueError("Transcript chunk targets must be positive.")
        if self.max_chars < self.target_chars:
            raise ValueError("Transcript max_chars must be >= target_chars.")
        if self.max_seconds < self.target_seconds:
            raise ValueError("Transcript max_seconds must be >= target_seconds.")
        if not 0 <= self.overlap_seconds < self.target_seconds:
            raise ValueError(
                "Transcript overlap_seconds must be non-negative and below "
                "target_seconds."
            )


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


def _transcript_chunks(
    snippets,
    config: TranscriptChunkConfig | None = None,
    *,
    video_id: str | None = None,
) -> list[dict]:
    """Pack complete captions using both semantic text and temporal bounds."""
    config = config or TranscriptChunkConfig()
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

    def duration(items: list[dict]) -> float:
        return items[-1]["end"] - items[0]["start"] if items else 0.0

    def overlap_from(items: list[dict]) -> list[dict]:
        overlap: list[dict] = []
        for previous in reversed(items[1:]):
            proposed = [previous, *overlap]
            if duration(proposed) > config.overlap_seconds:
                break
            overlap = proposed
        return overlap

    def flush() -> None:
        text = " ".join(item["text"] for item in current).strip()
        if len(text) >= MIN_CHUNK_LENGTH:
            chunk = {
                "text": text,
                "chunk_index": len(chunks),
                "content_hash": _content_hash(text),
                "start_seconds": round(current[0]["start"], 3),
                "end_seconds": round(current[-1]["end"], 3),
                "timestamp": _timestamp(current[0]["start"]),
                "extraction_method": "youtube_transcript",
            }
            if video_id:
                chunk["timestamp_url"] = (
                    f"https://www.youtube.com/watch?v={video_id}"
                    f"&t={max(0, int(current[0]['start']))}s"
                )
            chunks.append(chunk)

    for unit in units:
        proposed_chars = size(current) + len(unit["text"]) + (1 if current else 0)
        proposed_seconds = (
            unit["end"] - current[0]["start"] if current else unit["end"] - unit["start"]
        )
        reached_soft_target = current and (
            size(current) >= config.target_chars
            or duration(current) >= config.target_seconds
        )
        exceeds_hard_limit = current and (
            proposed_chars > config.max_chars
            or proposed_seconds > config.max_seconds
        )
        if reached_soft_target or exceeds_hard_limit:
            flush()
            current = overlap_from(current)
        current.append(unit)
    if current:
        flush()
    return chunks


def _transcript_quality(snippets, video_duration: float | None = None) -> dict:
    """Compute scalar diagnostics for deciding whether captions are usable."""
    cleaned = [_clean_snippet(snippet.text) for snippet in snippets]
    cleaned = [text for text in cleaned if text]
    combined = " ".join(cleaned)
    coverage_end = max(
        (float(snippet.start + snippet.duration) for snippet in snippets),
        default=0.0,
    )
    letters = [char for char in combined if char.isalpha()]
    devanagari = sum("\u0900" <= char <= "\u097f" for char in letters)
    latin = sum(("a" <= char.lower() <= "z") for char in letters)
    repeated = len(cleaned) - len({text.casefold() for text in cleaned})
    return {
        "transcript_snippet_count": len(cleaned),
        "transcript_word_count": len(combined.split()),
        "transcript_character_count": len(combined),
        "transcript_coverage_seconds": round(coverage_end, 3),
        "transcript_coverage_ratio": (
            round(min(coverage_end / video_duration, 1.0), 4)
            if video_duration
            else None
        ),
        "transcript_repeated_snippet_ratio": (
            round(repeated / len(cleaned), 4) if cleaned else 0.0
        ),
        "transcript_devanagari_letter_ratio": (
            round(devanagari / len(letters), 4) if letters else 0.0
        ),
        "transcript_latin_letter_ratio": (
            round(latin / len(letters), 4) if letters else 0.0
        ),
    }


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


def _index_video(
    info: dict, playlist: dict | None = None, owner_id: str | None = None,
) -> VideoResult:
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = str(info.get("id") or "")
    title = str(info.get("title") or video_id or "Unknown video")
    if not video_id:
        raise ValueError("YouTube returned a video without an ID.")

    transcript = _select_transcript(YouTubeTranscriptApi().list(video_id))
    fetched = transcript.fetch()
    chunks = _transcript_chunks(fetched, video_id=video_id)
    if not chunks:
        raise ValueError("The selected transcript did not contain usable text.")

    metadata = _video_metadata(info, playlist)
    metadata.update({
        "transcript_language": transcript.language,
        "transcript_language_code": transcript.language_code,
        "transcript_is_generated": bool(transcript.is_generated),
        **_transcript_quality(fetched, info.get("duration")),
    })
    count = index_chunks(
        chunks,
        _source_name(title, video_id),
        "youtube",
        document_key=video_id,
        source_metadata=metadata,
        owner_id=owner_id,
    )
    return VideoResult(video_id, title, "indexed", count)


def ingest_youtube(url: str, owner_id: str | None = None) -> dict:
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
            results.append(
                _index_video(entry, info if is_playlist else None, owner_id=owner_id)
            )
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
