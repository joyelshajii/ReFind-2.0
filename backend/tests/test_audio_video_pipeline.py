import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.processors.factory import get_file_type, process_file, SUPPORTED_EXTENSIONS
from app.processors.audio_processor import process_audio
from app.processors.video_processor import process_video
from app.services.query_parser import parse_query
from app.db.base import Base
from app.db.models import Document, Chunk
from app.services import indexer, hybrid_search
import app.db.base as db_base


@pytest.fixture
def media_samples(tmp_path):
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        pytest.skip("ffmpeg not available for media testing")

    # 1. Create a sample movie song with Album = 'Khalifa' (Movie name), Artist = 'Anirudh', Genre = 'Acoustic'
    audio_path = tmp_path / "khalifa_song_track1.wav"
    subprocess.run(
        [
            ffmpeg_bin, "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", "1",
            "-metadata", "title=Dheera Song",
            "-metadata", "album=Khalifa",
            "-metadata", "artist=Anirudh",
            "-metadata", "genre=Film Soundtrack",
            "-metadata", "date=2026",
            str(audio_path)
        ],
        capture_output=True,
        check=True
    )

    # 2. Create sample SRT subtitle file with dialogue
    srt_file = tmp_path / "dialogue.srt"
    srt_file.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nWhere is the confidential blueprint?\n\n"
        "2\n00:00:03,500 --> 00:00:05,000\nWe are deploying the new PostgreSQL cluster for Khalifa.\n",
        encoding="utf-8"
    )

    # 3. Create a video with embedded subtitles and chapters
    video_path = tmp_path / "Khalifa The Ruler (2026) Malayalam HQ HDRip - 1080p - HE.mkv"
    subprocess.run(
        [
            ffmpeg_bin, "-y",
            "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=10",
            "-i", str(srt_file),
            "-c:v", "h264",
            "-c:s", "srt",
            "-t", "5",
            "-metadata", "title=Khalifa The Ruler",
            "-metadata", "year=2026",
            str(video_path)
        ],
        capture_output=True,
        check=True
    )

    return {
        "audio": audio_path,
        "video": video_path,
    }


def test_factory_supports_media_extensions():
    assert get_file_type("movie.mkv") == "mkv"
    assert get_file_type("clip.mp4") == "mp4"
    assert get_file_type("song.mp3") == "mp3"
    assert get_file_type("track.wav") == "wav"
    assert get_file_type("audio.flac") == "flac"


def test_audio_processor_extracts_movie_and_track_data(media_samples):
    audio_path = media_samples["audio"]
    result = process_audio(str(audio_path))

    assert "Dheera Song" in result["extracted_text"]
    assert "Khalifa" in result["extracted_text"]
    assert "Anirudh" in result["extracted_text"]
    assert "Film Soundtrack" in result["extracted_text"]
    assert "2026" in result["extracted_text"]
    assert result["metadata"]["movie"] == "Khalifa"
    assert len(result["chunks"]) >= 1


def test_video_processor_extracts_embedded_subtitles_and_scenes(media_samples):
    video_path = media_samples["video"]
    result = process_video(str(video_path))

    assert "Khalifa The Ruler" in result["extracted_text"]
    assert "1080p Full HD" in result["extracted_text"]
    # Verify embedded subtitles were extracted into dialogue chunks!
    assert "PostgreSQL cluster for Khalifa" in result["extracted_text"]
    assert "confidential blueprint" in result["extracted_text"]
    assert len(result["chunks"]) >= 2


def test_query_parser_media_clues():
    q1 = parse_query("the song from Khalifa")
    assert q1.file_type == "audio"
    assert "Khalifa" in q1.people or "Khalifa" in q1.keywords or "Khalifa" in q1.topics

    q2 = parse_query("find the movie scene with the blueprint")
    assert q2.file_type == "video"
    assert "blueprint" in q2.keywords

    q3 = parse_query("find the Malayalam movie from 2026")
    assert q3.file_type == "video"
    assert "2026" in q3.time_clues


def test_index_and_search_dialogue_and_movie_song(media_samples, monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(indexer, "SessionLocal", sessions)
    monkeypatch.setattr(db_base, "SessionLocal", sessions)

    with sessions() as db:
        # Index audio & video
        doc_audio = indexer.index_single_file(str(media_samples["audio"]), db)
        doc_video = indexer.index_single_file(str(media_samples["video"]), db)

        assert doc_audio.id is not None
        assert doc_video.id is not None

        # 1. Search for song from movie: "the song from Khalifa"
        q_song = "the song from Khalifa"
        clues_song = parse_query(q_song)
        res_song = hybrid_search.search_documents(q_song, clues_song, db, top_k=5)
        assert len(res_song) > 0
        assert res_song[0].file_type == "wav"
        assert "khalifa_song" in res_song[0].filename.lower()

        # 2. Search for exact dialogue in the movie subtitles: "Where is the confidential blueprint"
        q_dialogue = "Where is the confidential blueprint in the movie"
        clues_dialogue = parse_query(q_dialogue)
        res_dialogue = hybrid_search.search_documents(q_dialogue, clues_dialogue, db, top_k=5)
        assert len(res_dialogue) > 0
        assert "Khalifa The Ruler" in res_dialogue[0].filename
        assert res_dialogue[0].file_type == "mkv"
        # Verify subtitle dialogue matched
        assert any("confidential blueprint" in ev.matched_text for ev in res_dialogue[0].evidence) or "confidential blueprint" in res_dialogue[0].snippet

    engine.dispose()
