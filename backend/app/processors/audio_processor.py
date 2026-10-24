import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger("refind.processors.audio")


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
        try:
            from app.services.media_metadata import probe_local_media
            return probe_local_media(file_path)
        except Exception as error:
            logger.debug("Audio metadata unavailable for %s: %s", file_path.name, error)
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


def _clean_token(text: str) -> str:
    return re.sub(r"[_\.\-\+]+", " ", text).strip()


def _parse_filename_movie_clues(filename: str) -> Dict[str, str]:
    """
    Parses filename patterns common in film soundtracks and songs:
    e.g. 'Khalifa - Song Name', 'Song Name (From Khalifa)', 'Song (Movie)', 'Movie - Track'
    """
    stem = Path(filename).stem
    clues = {"movie": "", "title": "", "artist": ""}

    # Pattern: "Song Name (From Movie Name)" or "Song Name [From Movie Name]"
    m = re.search(r"^(.*?)\s*[\(\[]\s*(?:from|movie|ost)\s+([^\]\)]+)[\)\]]", stem, re.IGNORECASE)
    if m:
        clues["title"] = _clean_token(m.group(1))
        clues["movie"] = _clean_token(m.group(2))
        return clues

    # Pattern: "Movie - Song" or "Artist - Song"
    if " - " in stem:
        parts = stem.split(" - ", 1)
        part1 = _clean_token(parts[0])
        part2 = _clean_token(parts[1])
        # If part2 has 'song' or track words, part1 is movie
        clues["title"] = part2
        clues["movie"] = part1
        return clues

    # Pattern: "Song (Movie)"
    m2 = re.search(r"^(.*?)\s*[\(\[]([^\]\)]+)[\)\]]$", stem)
    if m2:
        clues["title"] = _clean_token(m2.group(1))
        clues["movie"] = _clean_token(m2.group(2))
        return clues

    clues["title"] = _clean_token(stem)
    return clues


def _analyze_audio_with_gemini(file_path: Path) -> Optional[str]:
    """
    Extracts a 60-second lightweight audio snippet and analyzes it with Gemini
    to extract lyrics/speech transcript, musical style, mood, and movie/artist context.
    """
    if settings.LOCAL_FAST_MODE or not settings.GEMINI_API_KEY:
        return None

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return None

    temp_dir = tempfile.mkdtemp(prefix="refind_audio_snip_")
    snippet_path = Path(temp_dir) / "snippet.mp3"
    try:
        # Extract a 60-second snippet (at 96kbps mono, ~700KB)
        cmd = [
            ffmpeg_bin, "-y",
            "-ss", "0",
            "-i", str(file_path),
            "-t", "60",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "libmp3lame",
            "-b:a", "64k",
            str(snippet_path)
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=15)
        if res.returncode != 0 or not snippet_path.exists():
            return None

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.GEMINI_API_KEY, http_options={"timeout": 15000})
        with open(snippet_path, "rb") as f:
            audio_bytes = f.read()

        part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")
        prompt = """Analyze this audio recording or song in depth for a digital memory search system:
1. Audio Category: (Film song / movie soundtrack, independent music, speech / podcast, interview, voice memo, etc.)
2. If this is a song or music:
   - Identify or describe the movie / film / album context if recognizable.
   - Transcribe any lyrics or words sung or spoken (in English script and native language if applicable).
   - Describe the musical style, mood, tempo, and instruments (e.g., acoustic guitar, synth, energetic drums, violin, flute, piano).
   - Describe vocals (male, female, choir, solo, duet, instrumental).
3. If this is speech or spoken audio:
   - Transcribe the dialogue/speech content.
   - Summarize the topics discussed and key statements.

Provide clear, detailed text with headings:
[Audio Content & Lyrics Analysis]
- Movie / Album Context:
- Lyrics / Speech Transcript:
- Musical Mood & Instruments:
- Vocal & Audio Style:
"""
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[part, prompt]
        )
        if response and response.text:
            return response.text.strip()
    except Exception as exc:
        logger.debug("Gemini audio analysis skipped or failed on %s: %s", file_path.name, exc)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return None


def process_audio(file_path: str) -> Dict[str, Any]:
    """
    Performs deep inner data indexing of audio files:
    - Extracts full metadata tags (Movie/Album, Song Title, Singers, Composers, Lyricists, Year, Genre).
    - Parses filename patterns for movie and song associations.
    - Analyzes audio content, speech, and lyrics via multimodal AI when enabled.
    - Merges companion lyrics/transcripts (.lrc, .srt, .txt, .meta.json).
    - Produces rich structured text and semantic chunks embedded with EmbeddingGemma.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    from app.services.media_preflight import inspect_media, require_media_permission, transcript_chunks
    plan = inspect_media(path)
    require_media_permission(plan)

    filename = path.name
    stat = path.stat()
    file_size = stat.st_size
    ext = path.suffix.lower().lstrip(".")

    filename_clues = _parse_filename_movie_clues(filename)
    probe_data = _probe_media(path) or {}
    fmt = probe_data.get("format", {})
    tags = {k.lower(): str(v) for k, v in fmt.get("tags", {}).items()}

    audio_stream = None
    for s in probe_data.get("streams", []):
        if s.get("codec_type") == "audio":
            audio_stream = s
            break

    # Extract all metadata tags
    title = (
        tags.get("title")
        or tags.get("track_title")
        or tags.get("song_name")
        or filename_clues["title"]
        or _clean_token(path.stem)
    )

    # In film music / soundtracks, album tag IS the movie name!
    movie_name = (
        tags.get("movie")
        or tags.get("film")
        or tags.get("album")
        or filename_clues["movie"]
        or ""
    )
    album = tags.get("album") or movie_name or ""

    artist = (
        tags.get("artist")
        or tags.get("album_artist")
        or tags.get("performer")
        or tags.get("singer")
        or tags.get("lead_performer")
        or filename_clues.get("artist")
        or ""
    )
    composer = tags.get("composer") or tags.get("music_director") or ""
    lyricist = tags.get("lyricist") or tags.get("writer") or tags.get("author") or ""
    genre = tags.get("genre") or ""

    # Release Year / Date
    year = tags.get("date") or tags.get("year") or tags.get("original_year") or tags.get("creation_time") or ""
    if year and len(year) >= 4:
        match = re.search(r"\b(19\d\d|20\d\d)\b", year)
        if match:
            year = match.group(1)
    if not year:
        match = re.search(r"\b(19\d\d|20\d\d)\b", filename)
        if match:
            year = match.group(1)

    duration_sec = 0.0
    try:
        if fmt.get("duration"):
            duration_sec = float(fmt["duration"])
        elif audio_stream and audio_stream.get("duration"):
            duration_sec = float(audio_stream["duration"])
    except (ValueError, TypeError):
        pass

    duration_str = _format_duration(duration_sec) if duration_sec > 0 else "Unknown"
    codec = (audio_stream.get("codec_name") if audio_stream else ext).upper()
    sample_rate = audio_stream.get("sample_rate") if audio_stream else None
    channels = audio_stream.get("channels") if audio_stream else None
    bit_rate = fmt.get("bit_rate") or (audio_stream.get("bit_rate") if audio_stream else None)

    channel_desc = "Stereo" if channels == 2 else ("Mono" if channels == 1 else f"{channels} channels" if channels else "")
    bitrate_desc = f"{int(bit_rate) // 1000} kbps" if bit_rate and str(bit_rate).isdigit() else ""

    # Embedded lyrics / comments in ID3 tags
    embedded_lyrics = (
        tags.get("lyrics")
        or tags.get("unsyncedlyrics")
        or tags.get("unsynced_lyrics")
        or tags.get("comment")
        or ""
    )

    # Companion sidecar check (.meta.json, .lrc, .srt, .txt)
    sidecar_text = ""
    companion_meta = Path(f"{file_path}.meta.json")
    if companion_meta.exists():
        try:
            with open(companion_meta, "r", encoding="utf-8") as f:
                sdata = json.load(f)
                if isinstance(sdata, dict):
                    sidecar_text = sdata.get("transcript") or sdata.get("lyrics") or sdata.get("description") or ""
                    if not movie_name and sdata.get("movie"):
                        movie_name = sdata["movie"]
                    if not artist and sdata.get("artist"):
                        artist = sdata["artist"]
                    if not album and sdata.get("album"):
                        album = sdata["album"]
        except Exception as e:
            logger.debug("Failed reading audio sidecar %s: %s", companion_meta.name, e)

    for companion_ext in [".lrc", ".srt", ".txt"]:
        comp_file = path.with_suffix(companion_ext)
        if comp_file.exists() and comp_file != path:
            try:
                comp_content = comp_file.read_text(encoding="utf-8", errors="replace").strip()
                if comp_content:
                    sidecar_text += f"\n\nLyrics / Subtitles ({companion_ext}):\n{comp_content}"
            except Exception:
                pass

    # The local path listens to every segment; Gemini remains a legacy option.
    audio_chunks = []
    if settings.LOCAL_FAST_MODE and plan.strategy == "direct":
        from app.services.audio_service import build_audio_chunks
        audio_chunks = build_audio_chunks(path)
        duration_sec = audio_chunks[-1]["_embedding_provenance"]["end_seconds"]
        duration_str = _format_duration(duration_sec)

    # Deep AI Audio Content & Lyrics Analysis via Gemini
    ai_analysis = _analyze_audio_with_gemini(path) if plan.strategy == "direct" else None

    # Build comprehensive structured text
    text_lines = [
        f"Audio Track: {title}",
        f"Filename: {filename}",
        f"Format: {codec} Audio ({ext.upper()})",
    ]
    if movie_name:
        text_lines.append(f"Movie / Film / Album: {movie_name}")
    elif album:
        text_lines.append(f"Album / Collection: {album}")
    if artist:
        text_lines.append(f"Singer / Artist: {artist}")
    if composer:
        text_lines.append(f"Music Director / Composer: {composer}")
    if lyricist:
        text_lines.append(f"Lyricist: {lyricist}")
    if genre:
        text_lines.append(f"Genre / Musical Style: {genre}")
    if year:
        text_lines.append(f"Release Year / Date: {year}")

    specs = []
    if duration_str != "Unknown":
        specs.append(f"Duration: {duration_str}")
    if channel_desc:
        specs.append(channel_desc)
    if sample_rate:
        specs.append(f"{sample_rate} Hz")
    if bitrate_desc:
        specs.append(bitrate_desc)
    if specs:
        text_lines.append(f"Technical Specifications: {', '.join(specs)}")

    if embedded_lyrics:
        text_lines.append(f"Embedded Lyrics / Metadata Notes:\n{embedded_lyrics}")
    if sidecar_text:
        text_lines.append(f"Companion Transcript & Lyrics:\n{sidecar_text}")
    if ai_analysis:
        text_lines.append(f"\nDeep Audio Content & Scene Analysis:\n{ai_analysis}")
    if plan.transcript:
        text_lines.append(f"Transcript ({', '.join(plan.sources)}):\n{plan.transcript}")

    extracted_text = "\n".join(text_lines).strip()

    # Build human-readable summary
    summary_parts = [f"[Audio - {ext.upper()}] \"{title}\""]
    if movie_name:
        summary_parts.append(f"from '{movie_name}'")
    if artist:
        summary_parts.append(f"by {artist}")
    if duration_str != "Unknown":
        summary_parts.append(f"({duration_str})")
    summary = " ".join(summary_parts)
    if genre:
        summary += f" | {genre}"

    # Structured entities
    entities: List[Dict[str, str]] = []
    if movie_name:
        entities.append({"entity_type": "topic", "entity_value": movie_name})
    if artist:
        entities.append({"entity_type": "person", "entity_value": artist})
    if composer:
        entities.append({"entity_type": "person", "entity_value": composer})
    if year:
        entities.append({"entity_type": "time", "entity_value": str(year)})
    if genre:
        entities.append({"entity_type": "topic", "entity_value": genre})
    entities.append({"entity_type": "technology", "entity_value": codec})
    entities.append({"entity_type": "topic", "entity_value": "Audio"})
    if genre or movie_name:
        entities.append({"entity_type": "topic", "entity_value": "Song"})

    # Semantic Chunks embedded via EmbeddingGemma
    chunks: List[Dict[str, Any]] = [
        {
            "content": (
                f"Audio Track: {title}\n"
                f"Movie / Film: {movie_name or 'N/A'}\n"
                f"Artist / Singer: {artist or 'N/A'}\n"
                f"Composer: {composer or 'N/A'}\n"
                f"Genre: {genre or 'Unknown'}\n"
                f"Year: {year or 'N/A'}\n"
                f"Duration: {duration_str}"
            ),
            "page_number": 1,
            "slide_number": None,
        }
    ]

    # Additional chunk for lyrics / speech transcript
    full_transcript = (
        (embedded_lyrics + "\n" if embedded_lyrics else "") +
        (sidecar_text + "\n" if sidecar_text else "") +
        (ai_analysis if ai_analysis else "")
    ).strip()

    if plan.transcript:
        chunks.extend(transcript_chunks(plan))
    elif full_transcript:
        # Split into ~700 char chunks with paragraph preservation
        paragraphs = [p.strip() for p in full_transcript.split("\n\n") if p.strip()]
        current_chunk = []
        chunk_idx = 2
        for p in paragraphs:
            current_chunk.append(p)
            if len("\n\n".join(current_chunk)) > 700:
                chunks.append({
                    "content": f"Audio Content & Lyrics ({title}):\n" + "\n\n".join(current_chunk),
                    "page_number": chunk_idx,
                    "slide_number": None,
                })
                current_chunk = []
                chunk_idx += 1
        if current_chunk:
            chunks.append({
                "content": f"Audio Content & Lyrics ({title}):\n" + "\n\n".join(current_chunk),
                "page_number": chunk_idx,
                "slide_number": None,
            })

    chunks.extend(audio_chunks)

    return {
        "extracted_text": extracted_text,
        "summary": summary,
        "chunks": chunks,
        "entities": entities,
        "metadata": {
            "title": title,
            "movie": movie_name,
            "album": album,
            "artist": artist,
            "composer": composer,
            "lyricist": lyricist,
            "genre": genre,
            "year": year,
            "duration_seconds": duration_sec,
            "duration_formatted": duration_str,
            "codec": codec,
            "sample_rate": sample_rate,
            "channels": channels,
            "bit_rate": bit_rate,
            "audio_segments": len(audio_chunks),
            "content_indexing": "direct_audio" if audio_chunks else "text_only",
            "text_sources": list(plan.sources),
        }
    }
