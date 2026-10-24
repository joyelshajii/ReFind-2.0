import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from app.core.config import settings

logger = logging.getLogger("refind.processors.video")

LANGUAGES = [
    "malayalam", "hindi", "tamil", "telugu", "kannada", "bengali",
    "english", "spanish", "french", "german", "italian", "japanese",
    "korean", "chinese", "mandarin", "russian", "arabic", "portuguese"
]

QUALITY_TAGS = [
    "1080p", "720p", "480p", "4k", "2160p", "hdrip", "webrip",
    "bluray", "bdrip", "dvdrip", "hevc", "x264", "x265", "h264", "h265",
    "web-dl", "remux", "hq"
]


def _format_duration(seconds: float) -> str:
    total_sec = int(round(seconds))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _probe_media(file_path: Path) -> Optional[Dict[str, Any]]:
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None
    try:
        cmd = [
            ffprobe_bin,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            str(file_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if res.returncode == 0 and res.stdout.strip():
            return json.loads(res.stdout)
    except Exception as exc:
        logger.debug("ffprobe failed on %s: %s", file_path.name, exc)
    return None


def _get_resolution_label(width: Optional[int], height: Optional[int]) -> str:
    if not width or not height:
        return ""
    if width >= 3840 or height >= 2160:
        return f"4K Ultra HD ({width}x{height})"
    if width >= 2560 or height >= 1440:
        return f"2K Quad HD ({width}x{height})"
    if width >= 1920 or height >= 1080:
        return f"1080p Full HD ({width}x{height})"
    if width >= 1280 or height >= 720:
        return f"720p HD ({width}x{height})"
    if width >= 854 or height >= 480:
        return f"480p SD ({width}x{height})"
    return f"{width}x{height}"


def _parse_filename_clues(filename: str) -> Dict[str, Any]:
    clues: Dict[str, Any] = {
        "title": "",
        "year": "",
        "language": "",
        "quality": []
    }
    stem = Path(filename).stem
    clean = re.sub(r"[_\.\-\+]+", " ", stem).strip()

    # Look for 4-digit year
    year_match = re.search(r"\b(19\d\d|20\d\d)\b", clean)
    if year_match:
        clues["year"] = year_match.group(1)
        parts = re.split(r"\b(19\d\d|20\d\d)\b", clean, maxsplit=1)
        if parts[0].strip():
            clues["title"] = parts[0].strip()

    if not clues["title"]:
        pattern = r"\b(" + "|".join(QUALITY_TAGS) + r")\b"
        parts = re.split(pattern, clean, flags=re.IGNORECASE)
        clues["title"] = parts[0].strip() if parts[0].strip() else clean

    # Language detection
    for lang in LANGUAGES:
        if re.search(rf"\b{lang}\b", clean, re.IGNORECASE):
            clues["language"] = lang.capitalize()
            break

    # Quality tags
    for q in QUALITY_TAGS:
        if re.search(rf"\b{q}\b", clean, re.IGNORECASE):
            clues["quality"].append(q.upper())

    return clues


def _parse_srt_entries(srt_text: str) -> List[Tuple[str, str]]:
    """Parses SRT subtitle text into (timestamp_label, dialogue_text) entries."""
    entries: List[Tuple[str, str]] = []
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) >= 2:
            # Check for timestamp line e.g. "00:01:20,000 --> 00:01:23,000"
            time_line = lines[1] if ("-->" in lines[1]) else (lines[0] if "-->" in lines[0] else "")
            text_lines = lines[2:] if ("-->" in lines[1]) else lines[1:]
            if time_line and text_lines:
                times = time_line.split("-->")
                start_time = times[0].strip().split(",")[0].strip()
                dialogue = " ".join(text_lines)
                # Strip HTML tags
                dialogue = re.sub(r"<[^>]+>", "", dialogue).strip()
                if dialogue:
                    entries.append((start_time, dialogue))
    return entries


def _extract_all_subtitles(file_path: Path, subtitle_streams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts complete subtitles from all embedded streams and external sidecar files.
    Returns list of subtitle scene chunks with timestamps.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    all_entries: List[Tuple[str, str, str]] = [] # (lang, timestamp, text)

    # 1. Embedded subtitle streams
    if ffmpeg_bin and subtitle_streams:
        for idx, s in enumerate(subtitle_streams):
            lang = s.get("tags", {}).get("language") or s.get("tags", {}).get("title") or f"Track {idx+1}"
            try:
                cmd = [
                    ffmpeg_bin, "-y",
                    "-i", str(file_path),
                    "-map", f"0:s:{idx}",
                    "-f", "srt",
                    "-"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if res.returncode == 0 and res.stdout.strip():
                    entries = _parse_srt_entries(res.stdout)
                    for start_time, text in entries:
                        all_entries.append((lang.upper(), start_time, text))
            except Exception as exc:
                logger.debug("Failed extracting embedded subtitle stream %d from %s: %s", idx, file_path.name, exc)

    # 2. External sidecar subtitle files (.srt, .vtt)
    for companion_ext in [".srt", ".vtt", ".sub"]:
        comp_file = file_path.with_suffix(companion_ext)
        if comp_file.exists() and comp_file != file_path:
            try:
                content = comp_file.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    entries = _parse_srt_entries(content)
                    for start_time, text in entries:
                        all_entries.append(("EXTERNAL", start_time, text))
            except Exception:
                pass

    if not all_entries:
        return []

    # Group subtitle dialogue into ~600 char scene chunks with timestamps
    chunks: List[Dict[str, Any]] = []
    current_lines = []
    current_start = all_entries[0][1]
    current_lang = all_entries[0][0]
    chunk_idx = 1

    for lang, timestamp, text in all_entries:
        current_lines.append(f"[{timestamp}] {text}")
        if len("\n".join(current_lines)) > 600:
            chunks.append({
                "content": f"[Scene Subtitles {current_start} - {timestamp} ({current_lang})]:\n" + "\n".join(current_lines),
                "page_number": chunk_idx,
                "slide_number": None,
                "timestamp": current_start
            })
            current_lines = []
            current_start = timestamp
            current_lang = lang
            chunk_idx += 1

    if current_lines:
        chunks.append({
            "content": f"[Scene Subtitles {current_start} ({current_lang})]:\n" + "\n".join(current_lines),
            "page_number": chunk_idx,
            "slide_number": None,
            "timestamp": current_start
        })

    return chunks


def _analyze_video_scenes(file_path: Path, duration_sec: float) -> List[Dict[str, Any]]:
    """
    Samples keyframes across multiple scenes (e.g. 5%, 20%, 35%, 50%, 65%, 80%, 95%).
    Runs OCR on each frame for on-screen text, titles, signs, and slides.
    If Gemini is available, performs multimodal visual scene analysis.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return []

    # Pick up to 6 scene sample points across video
    if duration_sec <= 30:
        sample_times = [max(1.0, duration_sec * 0.5)]
    elif duration_sec <= 300:
        sample_times = [duration_sec * 0.15, duration_sec * 0.5, duration_sec * 0.85]
    else:
        sample_times = [
            duration_sec * 0.05,
            duration_sec * 0.20,
            duration_sec * 0.40,
            duration_sec * 0.60,
            duration_sec * 0.80,
            duration_sec * 0.95,
        ]

    temp_dir = tempfile.mkdtemp(prefix="refind_vscenes_")
    scene_results: List[Dict[str, Any]] = []

    try:
        from app.services.ocr_service import ocr_service

        client = None
        if settings.GEMINI_API_KEY:
            try:
                from google import genai
                client = genai.Client(api_key=settings.GEMINI_API_KEY, http_options={"timeout": 10000})
            except Exception:
                client = None

        for idx, t_sec in enumerate(sample_times):
            frame_path = Path(temp_dir) / f"frame_{idx}.jpg"
            seek_str = f"{int(t_sec)}"
            cmd = [
                ffmpeg_bin, "-y",
                "-ss", seek_str,
                "-i", str(file_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(frame_path)
            ]
            res = subprocess.run(cmd, capture_output=True, timeout=10)
            if res.returncode == 0 and frame_path.exists():
                time_label = _format_duration(t_sec)
                ocr_text = ""
                try:
                    with Image.open(str(frame_path)) as img:
                        ocr_lines = ocr_service.extract_text_from_image(img)
                        ocr_text = " ".join(ocr_lines).strip()
                except Exception:
                    pass

                scene_desc = ""
                if client:
                    try:
                        from google.genai import types
                        with open(frame_path, "rb") as f:
                            fb = f.read()
                        part = types.Part.from_bytes(data=fb, mime_type="image/jpeg")
                        prompt = "Describe this movie/video scene in 1-2 concise sentences: setting, people/action, visual atmosphere."
                        resp = client.models.generate_content(
                            model=settings.GEMINI_MODEL,
                            contents=[part, prompt]
                        )
                        if resp and resp.text:
                            scene_desc = resp.text.strip()
                    except Exception as e:
                        logger.debug("Gemini scene analysis skipped: %s", e)

                if ocr_text or scene_desc:
                    scene_results.append({
                        "timestamp": time_label,
                        "description": scene_desc,
                        "ocr_text": ocr_text
                    })
    except Exception as exc:
        logger.debug("Scene analysis error on %s: %s", file_path.name, exc)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return scene_results


def process_video(file_path: str) -> Dict[str, Any]:
    """
    Performs deep inner data indexing of video files:
    - Extracts complete embedded subtitle dialogue across all tracks with timestamp ranges.
    - Samples key scenes across video for visual scene analysis and on-screen OCR.
    - Extracts chapters, audio tracks, languages, resolution, and codecs.
    - Chunks dialogue chronologically so specific scenes and quotes match exact timestamps.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    filename = path.name
    stat = path.stat()
    file_size = stat.st_size
    ext = path.suffix.lower().lstrip(".")

    filename_clues = _parse_filename_clues(filename)
    probe_data = _probe_media(path) or {}
    fmt = probe_data.get("format", {})
    tags = {k.lower(): str(v) for k, v in fmt.get("tags", {}).items()}

    video_stream = None
    audio_streams = []
    subtitle_streams = []

    for s in probe_data.get("streams", []):
        ctype = s.get("codec_type")
        if ctype == "video" and video_stream is None:
            video_stream = s
        elif ctype == "audio":
            audio_streams.append(s)
        elif ctype == "subtitle":
            subtitle_streams.append(s)

    # Resolution & Codec
    width = video_stream.get("width") if video_stream else None
    height = video_stream.get("height") if video_stream else None
    vcodec = (video_stream.get("codec_name") if video_stream else "").upper()
    res_label = _get_resolution_label(width, height)

    # Duration
    duration_sec = 0.0
    try:
        if fmt.get("duration"):
            duration_sec = float(fmt["duration"])
        elif video_stream and video_stream.get("duration"):
            duration_sec = float(video_stream["duration"])
    except (ValueError, TypeError):
        pass
    duration_str = _format_duration(duration_sec) if duration_sec > 0 else "Unknown"

    # Title & Year
    title = tags.get("title") or filename_clues["title"] or Path(filename).stem
    year = tags.get("year") or tags.get("date") or filename_clues["year"]
    if year and len(year) >= 4:
        match = re.search(r"\b(19\d\d|20\d\d)\b", year)
        if match:
            year = match.group(1)

    # Audio languages
    audio_langs = []
    for a in audio_streams:
        alang = a.get("tags", {}).get("language") or ""
        acodec = a.get("codec_name", "").upper()
        achannels = "Stereo" if a.get("channels") == 2 else ("5.1 Surround" if a.get("channels") == 6 else f"{a.get('channels')}ch" if a.get("channels") else "")
        desc = f"{acodec} {achannels}".strip()
        if alang:
            desc = f"{alang.upper()} ({desc})"
        audio_langs.append(desc)

    language = filename_clues["language"]
    if not language and audio_langs:
        language = audio_langs[0].split()[0]

    # Chapters
    chapters = []
    for ch in probe_data.get("chapters", []):
        ch_title = ch.get("tags", {}).get("title") or ""
        ch_start = _format_duration(float(ch.get("start_time", 0)))
        if ch_title:
            chapters.append(f"[{ch_start}] {ch_title}")

    # 1. DEEP SUBTITLE EXTRACTION: Extract full dialogue across all tracks
    subtitle_chunks = _extract_all_subtitles(path, subtitle_streams)

    # 2. DEEP SCENE ANALYSIS: Sample keyframes across video for scene descriptions & OCR
    scenes = _analyze_video_scenes(path, duration_sec)

    # Check companion .meta.json
    sidecar_meta = Path(f"{file_path}.meta.json")
    sidecar_notes = ""
    if sidecar_meta.exists():
        try:
            with open(sidecar_meta, "r", encoding="utf-8") as f:
                sdata = json.load(f)
                if isinstance(sdata, dict):
                    sidecar_notes = sdata.get("description") or sdata.get("summary") or ""
                    if not title and sdata.get("title"):
                        title = sdata["title"]
                    if not year and sdata.get("year"):
                        year = str(sdata["year"])
        except Exception:
            pass

    # Build structured document text
    text_lines = [
        f"Video / Movie Title: {title}",
        f"Filename: {filename}",
        f"Format: {ext.upper()} Video ({vcodec})",
    ]
    if year:
        text_lines.append(f"Release Year / Date: {year}")
    if language:
        text_lines.append(f"Language: {language}")
    if res_label:
        text_lines.append(f"Resolution: {res_label}")
    if filename_clues["quality"]:
        text_lines.append(f"Quality & Source: {', '.join(filename_clues['quality'])}")
    if duration_str != "Unknown":
        text_lines.append(f"Duration: {duration_str}")
    if audio_langs:
        text_lines.append(f"Audio Tracks: {', '.join(audio_langs)}")
    if chapters:
        text_lines.append(f"Chapters / Scene List:\n" + "\n".join(chapters))
    if sidecar_notes:
        text_lines.append(f"Companion Notes: {sidecar_notes}")

    # Append scenes
    if scenes:
        scene_lines = []
        for s in scenes:
            line = f"Scene at {s['timestamp']}:"
            if s["description"]:
                line += f" {s['description']}"
            if s["ocr_text"]:
                line += f" | On-screen text: {s['ocr_text']}"
            scene_lines.append(line)
        text_lines.append("\nVisual Scene Analysis & Keyframes:\n" + "\n".join(scene_lines))

    # Append full subtitle sample/dialogue
    if subtitle_chunks:
        sample_subs = "\n\n".join([c["content"] for c in subtitle_chunks[:5]])
        text_lines.append(f"\nDialogue & Subtitles Script:\n{sample_subs}")

    extracted_text = "\n".join(text_lines).strip()

    # Build summary
    summary_parts = [f"[Video - {ext.upper()}] \"{title}\""]
    if year:
        summary_parts.append(f"({year})")
    if language:
        summary_parts.append(f"| Language: {language}")
    if res_label:
        summary_parts.append(f"| {res_label.split('(')[0].strip()}")
    if duration_str != "Unknown":
        summary_parts.append(f"({duration_str})")
    summary = " ".join(summary_parts)

    # Entities
    entities: List[Dict[str, str]] = []
    if title:
        entities.append({"entity_type": "topic", "entity_value": title})
    if year:
        entities.append({"entity_type": "time", "entity_value": str(year)})
    if language:
        entities.append({"entity_type": "topic", "entity_value": language})
    if res_label:
        entities.append({"entity_type": "technology", "entity_value": res_label.split()[0]})
    for q in filename_clues["quality"]:
        entities.append({"entity_type": "technology", "entity_value": q})
    entities.append({"entity_type": "technology", "entity_value": ext.upper()})
    entities.append({"entity_type": "topic", "entity_value": "Video"})
    entities.append({"entity_type": "topic", "entity_value": "Movie"})

    # Assemble Semantic Chunks embedded via EmbeddingGemma
    chunks: List[Dict[str, Any]] = [
        {
            "content": (
                f"Video / Movie Title: {title}\n"
                f"Release Year: {year or 'N/A'}\n"
                f"Language: {language or 'N/A'}\n"
                f"Resolution: {res_label or 'N/A'}\n"
                f"Duration: {duration_str}\n"
                f"Quality: {', '.join(filename_clues['quality']) or 'Standard'}\n"
                f"Audio: {', '.join(audio_langs) or 'Default'}"
            ),
            "page_number": 1,
            "slide_number": None,
        }
    ]

    # Add scene analysis chunks
    chunk_page = 2
    for s in scenes:
        content_parts = [f"Video Scene at {s['timestamp']} ({title}):"]
        if s["description"]:
            content_parts.append(s["description"])
        if s["ocr_text"]:
            content_parts.append(f"On-screen text: {s['ocr_text']}")
        chunks.append({
            "content": "\n".join(content_parts),
            "page_number": chunk_page,
            "slide_number": None,
        })
        chunk_page += 1

    # Add subtitle dialogue chunks
    for sc in subtitle_chunks:
        chunks.append({
            "content": f"Movie Dialogue ({title}):\n" + sc["content"],
            "page_number": chunk_page,
            "slide_number": None,
        })
        chunk_page += 1

    return {
        "extracted_text": extracted_text,
        "summary": summary,
        "chunks": chunks,
        "entities": entities,
        "metadata": {
            "title": title,
            "year": year,
            "language": language,
            "resolution": res_label,
            "codec": vcodec,
            "duration_seconds": duration_sec,
            "duration_formatted": duration_str,
            "audio_tracks": audio_langs,
            "chapters": chapters,
            "subtitles_extracted": len(subtitle_chunks) > 0,
            "scenes_analyzed": len(scenes),
        }
    }
