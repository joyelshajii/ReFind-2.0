import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base import Base
from app.db.models import Document, Chunk, MediaIndexJob
from app.processors.audio_processor import process_audio
from app.processors.video_processor import process_video
from app.services import media_preflight as preflight, media_jobs, indexer
from app.services.upload_service import save_and_index_uploads


@pytest.fixture
def media_env(tmp_path, monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(media_jobs, "SessionLocal", sessions)
    monkeypatch.setattr(indexer, "SessionLocal", sessions)
    monkeypatch.setattr(settings, "LOCAL_FAST_MODE", True)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(preflight, "probe_local_media", lambda path: {
        "format": {"duration": 3600, "tags": {"title": "Movie title", "comment": "A comment is not a transcript"}},
        "streams": [{"codec_type": "video", "codec_name": "h264"}],
    })
    monkeypatch.setattr("app.processors.audio_processor._probe_media", lambda path: None)
    monkeypatch.setattr("app.processors.video_processor._probe_media", lambda path: None)
    monkeypatch.setattr(indexer, "embed_texts_with_provenance", lambda texts, task: [
        ([1.0] + [0.0] * 767, indexer.expected_embedding_provenance(task)) for _ in texts])
    preflight._cache.clear()
    indexer.indexing_status.is_indexing = False
    yield tmp_path, sessions
    indexer.indexing_status.is_indexing = False
    preflight._cache.clear()
    engine.dispose()


def no_inference(*args, **kwargs):
    pytest.fail("Raw media inference must not run on this path")


@pytest.mark.parametrize("kind", ["mp4", "mp3"])
def test_existing_transcript_skips_all_media_inference(media_env, monkeypatch, kind):
    root, sessions = media_env
    source = root / ("recording." + kind)
    source.write_bytes(b"fixture")
    source.with_suffix(".en.srt").write_text("1\n00:00:01,000 --> 00:00:04,000\nThe spare key is in the flower pot.\n", encoding="utf-8")
    monkeypatch.setattr("app.services.audio_service.build_audio_chunks", no_inference)
    monkeypatch.setattr("app.services.video_service.build_video_chunks", no_inference)
    monkeypatch.setattr("app.processors.video_processor._analyze_video_scenes", no_inference)
    monkeypatch.setattr("app.processors.audio_processor._analyze_audio_with_gemini", no_inference)
    with sessions() as db:
        doc = indexer.index_single_file(str(source), db)
        assert "flower pot" in doc.extracted_text
        assert json.loads(doc.processing_signature)["strategy"] == "text"
        assert indexer._document_embeddings_match_configuration(doc)
        monkeypatch.setattr(indexer, "process_file", no_inference)
        assert indexer.index_single_file(str(source), db).id == doc.id


@pytest.mark.parametrize("content", ["", "1\n00:00:01,000 --> 00:00:04,000\n", "WEBVTT", "not a subtitle file"])
def test_empty_or_invalid_subtitles_still_require_confirmation(media_env, content):
    root, _ = media_env
    source = root / "movie.mp4"
    source.write_bytes(b"fixture")
    source.with_suffix(".srt").write_text(content)
    plan = preflight.inspect_media(source)
    assert plan.strategy == "direct"
    assert plan.requires_confirmation
    with pytest.raises(preflight.MediaConfirmationRequired):
        process_video(str(source))


def test_embedded_subtitles_and_lyrics_are_used(media_env, monkeypatch):
    root, _ = media_env
    source = root / "movie.mkv"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(preflight, "probe_local_media", lambda path: {
        "format": {"duration": 7200}, "streams": [{"codec_type": "subtitle"}]})
    monkeypatch.setattr(preflight, "read_local_subtitles", lambda path: iter([("ENG", 65, "The confidential blueprint.")]))
    assert "01:05" in preflight.inspect_media(source).transcript
    assert not preflight.inspect_media(source).requires_confirmation
    preflight._cache.clear()
    monkeypatch.setattr(preflight, "read_local_subtitles", lambda path: iter([]))
    assert preflight.inspect_media(source).requires_confirmation
    preflight._cache.clear()
    monkeypatch.setattr(preflight, "probe_local_media", lambda path: {
        "format": {"duration": 7200, "tags": {"lyrics-eng": "These are the actual song lyrics."}}, "streams": []})
    assert preflight.inspect_media(source).strategy == "text"


@pytest.mark.parametrize("duration,confirm", [(899, False), (900, False), (901, True), (0, True)])
def test_duration_boundary_and_unknown_duration(media_env, monkeypatch, duration, confirm):
    root, _ = media_env
    source = root / "recording.mp3"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(preflight, "probe_local_media", lambda path: {"format": {"duration": duration}, "streams": []})
    assert preflight.inspect_media(source).requires_confirmation is confirm


def test_uploads_save_later_selected_companion_before_processing(media_env, monkeypatch):
    _, sessions = media_env
    monkeypatch.setattr("app.services.video_service.build_video_chunks", no_inference)
    files = [UploadFile(filename="movie.mp4", file=io.BytesIO(b"fixture")),
             UploadFile(filename="movie.en.srt", file=io.BytesIO(b"1\n00:00:01,000 --> 00:00:02,000\nconfidential blueprint"))]
    with sessions() as db:
        result = save_and_index_uploads(files, db)
        assert result["status"] == "success"
        assert result["companions_saved"] == 1
        assert result["deferred_files"] == []
        assert "confidential blueprint" in db.query(Document).one().extracted_text


def test_long_upload_is_saved_and_deferred_without_inference(media_env, monkeypatch):
    root, sessions = media_env
    monkeypatch.setattr("app.services.audio_service.build_audio_chunks", no_inference)
    with sessions() as db:
        result = save_and_index_uploads([UploadFile(filename="voice.mp3", file=io.BytesIO(b"fixture"))], db)
        assert result["status"] == "awaiting_confirmation"
        assert len(result["deferred_files"]) == 1
        assert db.query(Document).count() == 0
    assert (root / "voice.mp3").exists()
    assert media_jobs.list_media_jobs()[0]["status"] == "awaiting"


def test_durable_schedule_runs_only_when_due_and_once(media_env, monkeypatch):
    root, sessions = media_env
    source = root / "voice.mp3"
    source.write_bytes(b"fixture")
    plan = preflight.inspect_media(source)
    job = media_jobs.defer_media(plan)
    calls = []
    def process(path, db):
        preflight.require_media_permission(preflight.inspect_media(path))
        calls.append(path)
    monkeypatch.setattr(indexer, "index_single_file", process)
    media_jobs.run_next_media_job()
    assert calls == []
    for date in [None, datetime.now(), datetime.now(timezone.utc) - timedelta(minutes=1)]:
        with pytest.raises(ValueError):
            media_jobs.decide_media_job(job["id"], "schedule", date)
    scheduled = media_jobs.decide_media_job(job["id"], "schedule", datetime.now(timezone.utc) + timedelta(hours=1))
    assert scheduled["scheduled_at"].endswith("+00:00")
    media_jobs.recover_media_jobs()
    media_jobs.run_next_media_job()
    assert calls == []
    with sessions() as db:
        db.get(MediaIndexJob, job["id"]).scheduled_at = datetime.utcnow() - timedelta(seconds=1)
        db.commit()
    media_jobs.run_next_media_job()
    media_jobs.run_next_media_job()
    assert calls == [str(source)]
    assert media_jobs.list_media_jobs() == []


def test_changed_file_revokes_approval_and_skip_does_not_run(media_env, monkeypatch):
    root, sessions = media_env
    source = root / "voice.mp3"
    source.write_bytes(b"fixture")
    job = media_jobs.defer_media(preflight.inspect_media(source))
    media_jobs.decide_media_job(job["id"], "continue")
    source.write_bytes(b"changed content")
    monkeypatch.setattr(indexer, "index_single_file", no_inference)
    media_jobs.run_next_media_job()
    assert media_jobs.list_media_jobs()[0]["status"] == "awaiting"
    media_jobs.decide_media_job(job["id"], "skip")
    media_jobs.run_next_media_job()
    assert media_jobs.list_media_jobs() == []
    assert source.exists()


def test_long_transcript_chunks_cover_the_end(media_env):
    root, _ = media_env
    source = root / "voice.mp3"
    source.write_bytes(b"fixture")
    source.with_suffix(".txt").write_text("word " * 4000 + "final unique sentence")
    plan = preflight.inspect_media(source)
    chunks = preflight.transcript_chunks(plan)
    assert len(chunks) > 10
    assert max(len(chunk["content"]) for chunk in chunks) <= 820
    assert "final unique sentence" in chunks[-1]["content"]


def test_uploading_transcript_requires_selecting_recording_again(media_env, monkeypatch):
    root, sessions = media_env
    source = root / "voice.mp3"
    source.write_bytes(b"fixture")
    media_jobs.defer_media(preflight.inspect_media(source))
    monkeypatch.setattr("app.services.audio_service.build_audio_chunks", no_inference)
    with sessions() as db:
        response = save_and_index_uploads([
            UploadFile(filename="voice.txt", file=io.BytesIO(b"We discussed the confidential blueprint."))
        ], db)
        assert response["status"] == "success"
        assert [item["filename"] for item in response["files"]] == ["voice.txt"]
        assert db.query(Document).filter_by(filepath=str(source)).first() is None
        assert media_jobs.list_media_jobs()[0]["status"] == "awaiting"
        response = save_and_index_uploads([
            UploadFile(filename="voice.mp3", file=io.BytesIO(b"fixture"))
        ], db)
        assert [item["filename"] for item in response["files"]] == ["voice.mp3"]
    assert media_jobs.list_media_jobs() == []


def test_media_job_api_validation_and_busy_upload(media_env):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.index import router
    from app.db.base import get_db
    root, sessions = media_env
    source = root / "voice.mp3"
    source.write_bytes(b"fixture")
    job = media_jobs.defer_media(preflight.inspect_media(source))
    app = FastAPI()
    app.include_router(router)
    def get_test_db():
        with sessions() as db:
            yield db
    app.dependency_overrides[get_db] = get_test_db
    with TestClient(app) as client:
        assert client.get("/api/index/media-jobs").json()[0]["status"] == "awaiting"
        url = f"/api/index/media-jobs/{job['id']}"
        assert client.post(url, json={"action": "invalid"}).status_code == 422
        assert client.post(url, json={"action": "schedule"}).status_code == 400
        assert client.post(url, json={"action": "continue"}).json()["status"] == "queued"
        assert indexer.try_reserve_indexing_run()
        try:
            assert client.post("/api/index/upload", files={"files": ("file.txt", b"hello")}).status_code == 409
        finally:
            indexer.release_indexing_run()
        assert client.post(url, json={"action": "skip"}).json()["status"] == "skipped"


def test_folder_defers_long_media_without_marking_failure(media_env, monkeypatch):
    root, sessions = media_env
    source = root / "movie.mp4"
    source.write_bytes(b"fixture")
    monkeypatch.setattr("app.services.video_service.build_video_chunks", no_inference)
    indexer.index_folder_background(str(root))
    assert indexer.indexing_status.current_stage == "Awaiting confirmation"
    assert indexer.indexing_status.error_message is None
    assert media_jobs.list_media_jobs()[0]["status"] == "awaiting"
    with sessions() as db:
        assert db.query(Document).count() == 0


def test_subtitle_decoder_preserves_unicode_and_commas(monkeypatch):
    import av
    from types import SimpleNamespace
    from app.services.media_metadata import read_local_subtitles
    class Subtitle:
        ass = '1,0,Default,,0,0,0,,{\\i1}♪ Hello, മലയാളം\\Nsecond line ♪'.encode()
        @property
        def dialogue(self):
            raise ValueError("byte must be in range(0, 256)")
    stream = SimpleNamespace(codec_context=SimpleNamespace(name="srt"), metadata={"language": "eng"})
    packet = SimpleNamespace(pts=65000, time_base=0.001, stream=stream, decode=lambda: [Subtitle()])
    class Container:
        streams = SimpleNamespace(subtitles=[stream])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def demux(self, streams): return iter([packet])
    monkeypatch.setattr(av, "open", lambda path: Container())
    assert list(read_local_subtitles("movie.mkv")) == [("ENG", 65, "♪ Hello, മലയാളം second line ♪")]


def test_existing_direct_index_is_reused_without_new_approval(media_env, monkeypatch):
    root, sessions = media_env
    source = root / "voice.mp3"
    source.write_bytes(b"fixture")
    provenance = indexer.expected_embedding_provenance("retrieval.passage")
    old_signature = {"file_type": "mp3", "chunk_size": settings.CHUNK_SIZE,
                     "chunk_overlap": settings.CHUNK_OVERLAP, "local_fast_mode": True,
                     "audio_processor": 1, "segment_seconds": 25, "overlap_seconds": 5,
                     "audio_embedding_config": provenance["config_id"]}
    with sessions() as db:
        doc = Document(filename=source.name, filepath=str(source), file_type="mp3",
                       file_hash=indexer.compute_file_hash(str(source)),
                       processing_signature=json.dumps(old_signature))
        doc.set_embedding([1.0] + [0.0] * 767)
        doc.set_embedding_provenance(provenance)
        chunk = Chunk(content="Audio 00:00–00:25", chunk_index=0)
        chunk.set_embedding([1.0] + [0.0] * 767)
        chunk.set_embedding_provenance({**provenance, "task": "retrieval.audio"})
        doc.chunks.append(chunk)
        db.add(doc)
        db.commit()
        monkeypatch.setattr(indexer, "process_file", no_inference)
        assert indexer.index_single_file(str(source), db).id == doc.id
    assert media_jobs.list_media_jobs() == []
