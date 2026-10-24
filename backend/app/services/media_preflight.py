"""Inspect existing text before any audio/video inference or frame decoding."""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import threading

from app.services.media_metadata import probe_local_media, read_local_subtitles

MEDIA_POLICY_VERSION = 2
LONG_MEDIA_SECONDS = 15 * 60
TEXT_SUFFIXES = (".srt", ".vtt", ".lrc", ".txt", ".ass", ".ssa", ".sub")
_cache = OrderedDict()
_cache_lock = threading.RLock()
_permission = ContextVar("media_processing_permission", default=None)


def clear_media_cache():
    with _cache_lock:
        _cache.clear()


def companion_files(path):
    path = Path(path)
    candidates = {Path(str(path) + ".meta.json"), path.with_suffix(".meta.json")}
    for suffix in TEXT_SUFFIXES:
        candidates.add(path.with_suffix(suffix))
        candidates.add(Path(str(path) + suffix))
    # Match common language variants: film.en.srt, film.ml.vtt, etc.
    for candidate in path.parent.iterdir():
        if (candidate.is_file() and candidate.name.casefold().startswith(path.stem.casefold() + ".")
                and candidate.suffix.lower() in TEXT_SUFFIXES):
            candidates.add(candidate)
    return sorted(p for p in candidates if p != path and p.is_file())


def media_fingerprint(path):
    path = Path(path).resolve()
    identities = []
    for item in [path, *companion_files(path)]:
        stat = item.stat()
        identities.append((str(item), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return hashlib.sha256(json.dumps(identities).encode()).hexdigest()


def _read_text(path):
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16" if data.startswith((b'\xff\xfe', b'\xfe\xff')) else "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeError:
            continue
    return data.decode("cp1252", errors="replace")


def clean_transcript(text, suffix=".txt"):
    if not isinstance(text, str) or "\x00" in text:
        return ""
    if suffix in {".srt", ".vtt", ".sub"}:
        if suffix == ".sub" and re.search(r"\{\d+\}\{\d+\}", text):
            text = re.sub(r"\{\d+\}\{\d+\}", "", text).replace("|", " ")
        elif not re.search(r"\d{1,2}:\d{2}(?::\d{2})?[.,]\d+\s*-->", text):
            return ""
    if suffix in {".ass", ".ssa"}:
        lines = []
        for line in text.splitlines():
            if line.strip().lower().startswith("dialogue:"):
                fields = line.split(",", 9)
                if len(fields) == 10:
                    lines.append(f"[{fields[1]}] {fields[9]}")
        text = "\n".join(lines)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.isdigit() or line in {"WEBVTT", "NOTE", "STYLE", "REGION"}:
            continue
        line = re.sub(r"<[^>]*>|\{[^}]*\}", "", line)
        line = line.replace("\\N", " ").replace("\\n", " ")
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    # Timestamps, LRC credits, and whitespace alone are not spoken content.
    words = re.sub(r"\[[^]]*\]|\d[\d:.,]*\s*-->[^\n]*", "", cleaned)
    if sum(char.isalpha() for char in words) < 3:
        return ""
    return cleaned


@dataclass(frozen=True)
class MediaPlan:
    filepath: str
    fingerprint: str
    duration_seconds: float | None
    transcript: str
    sources: tuple[str, ...]

    @property
    def strategy(self):
        return "text" if self.transcript else "direct"

    @property
    def requires_confirmation(self):
        return not self.transcript and (
            self.duration_seconds is None or self.duration_seconds > LONG_MEDIA_SECONDS
        )


class MediaConfirmationRequired(RuntimeError):
    def __init__(self, plan):
        self.plan = plan
        super().__init__("No usable transcript was found. Confirm full media indexing or schedule it for later.")


def inspect_media(path):
    path = Path(path).resolve()
    fingerprint = media_fingerprint(path)
    with _cache_lock:
        if fingerprint in _cache:
            _cache.move_to_end(fingerprint)
            return _cache[fingerprint]
    try:
        metadata = probe_local_media(path)
    except Exception as error:
        raise RuntimeError(f"Media inspection failed for {path.name}: {error}") from error
    durations = [metadata.get("format", {}).get("duration"), *(
        stream.get("duration") for stream in metadata.get("streams", [])
    )]
    durations = [float(value) for value in durations if value and math.isfinite(float(value)) and float(value) > 0]
    duration = max(durations) if durations else None
    contents, sources = [], []
    for sidecar in companion_files(path):
        try:
            raw = _read_text(sidecar)
            if sidecar.suffix.lower() == ".json":
                data = json.loads(raw)
                raw = "\n".join(data.get(key, "") for key in ("transcript", "lyrics", "subtitles")
                                if isinstance(data, dict) and isinstance(data.get(key), str))
            content = clean_transcript(raw, sidecar.suffix.lower())
            if content:
                contents.append(content)
                sources.append(sidecar.name)
        except (OSError, ValueError, TypeError):
            continue
    tag_sets = [metadata.get("format", {}).get("tags", {}), *(
        stream.get("tags", {}) for stream in metadata.get("streams", [])
    )]
    for tags in tag_sets:
        for key, value in tags.items():
            if key.lower().split("-")[0] in {"lyrics", "unsyncedlyrics", "unsynced_lyrics", "transcript"}:
                content = clean_transcript(value)
                if content and content not in contents:
                    contents.append(content)
                    sources.append("Embedded " + key)
    # External transcripts avoid even demuxing a large movie's subtitle track.
    if not contents and any(stream["codec_type"] == "subtitle" for stream in metadata.get("streams", [])):
        lines = []
        from app.services.audio_service import timestamp
        for language, start, content in read_local_subtitles(path):
            content = clean_transcript(content)
            if content:
                lines.append(f"[{timestamp(start)} {language}] {content}")
        if lines:
            contents.append("\n".join(lines))
            sources.append("Embedded subtitles")
    plan = MediaPlan(str(path), fingerprint, duration, "\n\n".join(contents), tuple(sources))
    # Reject a moving target rather than cache a partial or mismatched transcript.
    if media_fingerprint(path) != fingerprint:
        raise RuntimeError("Media or its transcript changed during inspection. Please try again.")
    with _cache_lock:
        _cache[fingerprint] = plan
        while len(_cache) > 16:
            _cache.popitem(last=False)
    return plan


def require_media_permission(plan):
    if plan.requires_confirmation and _permission.get() != (plan.filepath, plan.fingerprint):
        raise MediaConfirmationRequired(plan)


@contextmanager
def allow_media_processing(plan):
    token = _permission.set((plan.filepath, plan.fingerprint))
    try:
        yield
    finally:
        _permission.reset(token)


def transcript_chunks(plan):
    """Bound text length even for a transcript with no paragraph breaks."""
    chunks = []
    text = plan.transcript
    start = 0
    while start < len(text):
        end = min(start + 800, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start + 400, end), text.rfind(" ", start + 400, end))
            if boundary > start:
                end = boundary
        content = text[start:end].strip()
        if content:
            chunks.append({"content": "Media transcript:\n" + content,
                           "page_number": None, "slide_number": None})
        if end == len(text):
            break
        start = max(start + 1, end - 100)
    return chunks
