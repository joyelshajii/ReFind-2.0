"""Regression coverage for full-recording audio retrieval without transcripts."""
import wave
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.base import Base
from app.db.models import Document
from app.processors import audio_processor
from app.services import audio_service, embedding_service, hybrid_search, indexer
from app.services.query_parser import parse_query


def write_audio(path, seconds):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        # Nonzero stereo PCM, exercising downmixing and 8 -> 16 kHz resampling.
        stream.writeframes(np.full((int(seconds * 8000), 2), 1600, dtype="<i2").tobytes())


@pytest.mark.parametrize("seconds,expected", [
    (1, [(0, 1)]), (25, [(0, 25)]),
    (45, [(0, 25), (20, 45)]),
    (66, [(0, 25), (20, 45), (40, 65), (60, 66)]),
])
def test_decoder_covers_entire_recording_with_overlap(tmp_path, seconds, expected):
    path = tmp_path / "recording.wav"
    write_audio(path, seconds)
    segments = list(audio_service.iter_audio_segments(path))
    assert [(start, end) for start, end, _ in segments] == expected
    for start, end, samples in segments:
        assert samples.dtype == np.float32
        assert samples.shape == (int((end - start) * 16000),)
        assert np.isfinite(samples).all()
        assert np.max(samples) > 0


@pytest.fixture
def local_audio(monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_FAST_MODE", True)
    monkeypatch.setattr(settings, "JINA_API_KEY", "configured-but-must-not-rerank-audio")
    monkeypatch.setattr(settings, "SEARCH_INLINE_EXPLANATIONS", False)
    monkeypatch.setattr(embedding_service, "_embeddinggemma_runtime_config", lambda: ("cpu", None, "float32"))
    audio_vector = [1.0] + [0.0] * 767
    text_vector = [0.0, 1.0] + [0.0] * 766
    calls = []

    def embed_audio(samples):
        calls.append(len(samples))
        return audio_vector, embedding_service.expected_embedding_provenance("retrieval.audio")

    def embed_texts(texts, task="retrieval.passage"):
        assert not any(text.startswith("Audio 00:") for text in texts)
        return [(text_vector, embedding_service.expected_embedding_provenance(task)) for _ in texts]

    monkeypatch.setattr(embedding_service, "embed_audio_with_provenance", embed_audio)
    monkeypatch.setattr(indexer, "embed_texts_with_provenance", embed_texts)
    monkeypatch.setattr(hybrid_search, "get_embedding_with_provenance", lambda *a, **k: (
        audio_vector, embedding_service.expected_embedding_provenance("retrieval.query")))
    monkeypatch.setattr(audio_processor, "_probe_media", lambda path: None)
    monkeypatch.setattr(audio_processor, "_analyze_audio_with_gemini", lambda path: None)

    def forbidden(*args, **kwargs):
        pytest.fail("Audio matches must not be reranked using timestamp labels")

    monkeypatch.setattr(hybrid_search.reranker_service, "rerank", forbidden)
    return calls


@pytest.mark.parametrize("folder", [False, True])
def test_audio_vectors_survive_upload_and_folder_indexing(local_audio, tmp_path, folder):
    source = tmp_path / "memo.wav"
    write_audio(source, 46)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        if folder:
            metrics = {key: 0 for key in (
                "failed_files", "hash_seconds", "extraction_seconds", "embedding_seconds",
                "database_seconds", "indexed_files", "skipped_files",
            )}
            indexer._index_folder_local_fast(db, [{
                "filepath": str(source), "filename": source.name, "file_type": "wav",
                "file_size": source.stat().st_size,
            }], metrics)
            assert metrics["failed_files"] == 0
            doc = db.query(Document).one()
        else:
            doc = indexer.index_single_file(str(source), db)
        chunks = [ch for ch in doc.chunks
                  if ch.get_embedding_provenance().get("task") == "retrieval.audio"]
        assert len(chunks) == 3
        assert chunks[-1].get_embedding_provenance()["end_seconds"] == 46
        assert chunks[-1].get_embedding()[0] == 1
        assert "blueprint" not in doc.extracted_text.lower()
        for query in ['blueprint', 'audio where I said confidential blueprint', '"confidential blueprint"']:
            results = hybrid_search.search_documents(query, parse_query(query), db)
            assert results[0].document_id == doc.id
            assert results[0].filepath == str(source.resolve())
            assert results[0].page_or_slide == "Audio 00:00–00:25"
            assert any(ev.clue_type == "audio" and not ev.verified for ev in results[0].evidence)
        count = len(local_audio)
        assert indexer.index_single_file(str(source), db).id == doc.id
        assert len(local_audio) == count

        # An old metadata-only signature must force audio extraction even though
        # the file hash and text embedding space have not changed.
        doc.processing_signature = '{"file_type":"wav","chunk_size":800,"chunk_overlap":150}'
        db.commit()
        assert indexer.index_single_file(str(source), db).id == doc.id
        assert len(local_audio) == count + 3

        # Vector refresh must also listen again, not embed timestamp strings.
        doc.chunks[-1].set_embedding([])
        db.commit()
        indexer.index_single_file(str(source), db)
        assert len(local_audio) == count + 6
    engine.dispose()


def test_failed_audio_decode_preserves_previous_index(local_audio, tmp_path, monkeypatch):
    source = tmp_path / "memo.wav"
    write_audio(source, 1)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        doc = indexer.index_single_file(str(source), db)
        previous_hash = doc.file_hash
        source.write_bytes(b"broken audio")
        monkeypatch.setattr(settings, "INDEX_HASH_METADATA_FAST_PATH_SECONDS", 0)
        with pytest.raises(RuntimeError, match="Media inspection failed"):
            indexer.index_single_file(str(source), db)
        db.refresh(doc)
        assert doc.file_hash == previous_hash
        assert any(ch.get_embedding_provenance().get("task") == "retrieval.audio" for ch in doc.chunks)
    engine.dispose()


def test_audio_encoder_api_and_provenance(monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_FAST_MODE", True)
    monkeypatch.setattr(embedding_service, "_embeddinggemma_runtime_config", lambda: ("cpu", None, "float32"))

    class Model:
        def encode(self, inputs, **kwargs):
            assert inputs["audio"].shape == (16000,)
            assert kwargs["processing_kwargs"]["audio"]["sampling_rate"] == 16000
            return np.ones(768, dtype=np.float32) / np.sqrt(768)

    monkeypatch.setattr(embedding_service, "_get_embeddinggemma_model", lambda: Model())
    vector, provenance = embedding_service.embed_audio_with_provenance(np.ones(16000))
    assert len(vector) == 768
    assert provenance["task"] == "retrieval.audio"
    assert embedding_service.embedding_signature_is_compatible(
        provenance, embedding_service.expected_embedding_provenance("retrieval.query"))
