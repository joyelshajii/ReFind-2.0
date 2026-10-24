import numpy as np
import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.base import Base
from app.db.models import Document
from app.processors import video_processor
from app.services import embedding_service, hybrid_search, indexer, video_service
from app.services.query_parser import parse_query


def write_video(path, seconds=15, audio=False):
    import av
    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=10)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        sound = container.add_stream("pcm_s16le", rate=16000) if audio else None
        if sound:
            sound.layout = "mono"
        for index in range(seconds * 10):
            pixels = np.full((48, 64, 3), (index % 200), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in video.encode(frame):
                container.mux(packet)
            if sound:
                frame = av.AudioFrame.from_ndarray(np.ones((1, 1600), dtype=np.int16), format="s16", layout="mono")
                frame.sample_rate = 16000
                for packet in sound.encode(frame):
                    container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if sound:
            for packet in sound.encode():
                container.mux(packet)


def test_video_sampler_covers_final_segment_without_subtitles(tmp_path):
    source = tmp_path / "clip.mkv"
    write_video(source)
    segments = list(video_service.iter_video_segments(source))
    assert [(start, end, len(frames)) for start, end, frames in segments] == [
        (0, 8, 8), (6, 14, 8), (12, 15, 3),
    ]
    from app.services.media_metadata import probe_local_media, read_local_subtitles
    metadata = probe_local_media(source)
    assert metadata["streams"][0]["width"] == 64
    assert metadata["format"]["duration"] == pytest.approx(15)
    assert list(read_local_subtitles(source)) == []


@pytest.mark.parametrize("with_audio", [False, True])
@pytest.mark.parametrize("folder", [False, True])
def test_subtitle_free_video_searches_visuals_and_speech(tmp_path, monkeypatch, with_audio, folder):
    source = tmp_path / "clip.mkv"
    write_video(source, seconds=2, audio=with_audio)
    monkeypatch.setattr(settings, "LOCAL_FAST_MODE", True)
    monkeypatch.setattr(settings, "SEARCH_INLINE_EXPLANATIONS", False)
    monkeypatch.setattr(settings, "JINA_API_KEY", "")
    monkeypatch.setattr(embedding_service, "_embeddinggemma_runtime_config", lambda: ("cpu", None, "float32"))
    video_vec = [1.0, 0.0, 0.0] + [0.0] * 765
    audio_vec = [0.0, 1.0, 0.0] + [0.0] * 765
    text_vec = [0.0, 0.0, 1.0] + [0.0] * 765
    monkeypatch.setattr(embedding_service, "embed_video_with_provenance", lambda frames: (
        video_vec, embedding_service.expected_embedding_provenance("retrieval.video")))
    monkeypatch.setattr(embedding_service, "embed_audio_with_provenance", lambda samples: (
        audio_vec, embedding_service.expected_embedding_provenance("retrieval.audio")))
    monkeypatch.setattr(indexer, "embed_texts_with_provenance", lambda texts, task: [
        (text_vec, embedding_service.expected_embedding_provenance(task)) for _ in texts])
    monkeypatch.setattr(video_processor, "_probe_media", lambda path: None)

    def no_remote_scenes(*args):
        pytest.fail("Local video processing must not call remote scene analysis")

    monkeypatch.setattr(video_processor, "_analyze_video_scenes", no_remote_scenes)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        if folder:
            metrics = {key: 0 for key in (
                "failed_files", "hash_seconds", "extraction_seconds", "embedding_seconds",
                "database_seconds", "indexed_files", "skipped_files",
            )}
            indexer._index_folder_local_fast(db, [{
                "filepath": str(source), "filename": source.name, "file_type": "mkv",
                "file_size": source.stat().st_size,
            }], metrics)
            assert metrics["failed_files"] == 0
            doc = db.query(Document).one()
        else:
            doc = indexer.index_single_file(str(source), db)
        tasks = {chunk.get_embedding_provenance()["task"] for chunk in doc.chunks}
        assert "retrieval.video" in tasks
        assert ("retrieval.audio" in tasks) == with_audio
        assert "Subtitles Script" not in doc.extracted_text
        queries = [("a dog running in the video", video_vec, "Video")]
        if with_audio:
            queries.append(('video where someone says "confidential blueprint"', audio_vec, "Audio"))
        for query, vector, label in queries:
            monkeypatch.setattr(hybrid_search, "get_embedding_with_provenance", lambda *a, **kw: (
                vector, embedding_service.expected_embedding_provenance("retrieval.query")))
            results = hybrid_search.search_documents(query, parse_query(query), db)
            assert results[0].document_id == doc.id
            assert results[0].page_or_slide.startswith(label + " ")
        doc.processing_signature = ""
        db.commit()
        assert not indexer._processing_state_matches(doc, "mkv", indexer._processing_signature(str(source), "mkv"))
    engine.dispose()


def test_video_encoder_api(monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_FAST_MODE", True)
    monkeypatch.setattr(embedding_service, "_embeddinggemma_runtime_config", lambda: ("cpu", None, "float32"))

    class Model:
        def encode(self, inputs, **kwargs):
            assert len(inputs["video"]) == 2
            assert kwargs["processing_kwargs"]["video"]["add_timestamps"] is False
            return np.ones(768, dtype=np.float32) / np.sqrt(768)

    monkeypatch.setattr(embedding_service, "_get_embeddinggemma_model", lambda: Model())
    _, provenance = embedding_service.embed_video_with_provenance([Image.new("RGB", (32, 32))] * 2)
    assert embedding_service.embedding_signature_is_compatible(
        provenance, embedding_service.expected_embedding_provenance("retrieval.query"))
