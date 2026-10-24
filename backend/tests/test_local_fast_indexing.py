from __future__ import annotations

import hashlib
import sys
import time
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.base import Base
from app.db.models import Chunk, Document
from app.services import embedding_service, indexer


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = []
        self.fail_large_batch_once = False
        self._failed = False

    @staticmethod
    def _vector(value: str) -> np.ndarray:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        vector = np.zeros(768, dtype=np.float32)
        for index, byte in enumerate(digest):
            vector[index] = byte / 255.0
        return vector

    def encode(self, inputs, prompt_name=None, **kwargs):
        self.calls.append((inputs, prompt_name, kwargs))
        if self.fail_large_batch_once and isinstance(inputs, list) and len(inputs) > 1 and not self._failed:
            self._failed = True
            raise RuntimeError("CUDA out of memory")
        if isinstance(inputs, dict):
            image = inputs["image"]
            return self._vector(f"image:{image.size}:{image.mode}:{image.tobytes()}")
        if isinstance(inputs, list):
            return np.stack([self._vector(str(value)) for value in inputs])
        return self._vector(str(inputs))


@pytest.fixture
def local_fast(monkeypatch):
    model = FakeEmbeddingModel()
    monkeypatch.setattr(settings, "LOCAL_FAST_MODE", True)
    monkeypatch.setattr(settings, "LOCAL_EMBEDDING_BATCH_SIZE", 4)
    monkeypatch.setattr(settings, "INDEX_EMBEDDING_FILE_BATCH_SIZE", 4)
    monkeypatch.setattr(settings, "INDEX_EXTRACTION_WORKERS", 2)
    monkeypatch.setattr(embedding_service, "_embeddinggemma_runtime_config", lambda: ("cpu", None, "float32"))
    monkeypatch.setattr(embedding_service, "_local_model", model)
    monkeypatch.setattr(
        embedding_service,
        "_local_model_config",
        (settings.LOCAL_EMBEDDING_MODEL_ID, settings.LOCAL_EMBEDDING_MODEL_REVISION, "cpu", "float32"),
    )
    embedding_service._embeddinggemma_cache.clear()
    yield model
    embedding_service._embeddinggemma_cache.clear()


def _db_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False)(), engine


def _metrics():
    return {
        "failed_files": 0, "skipped_images": 0, "indexed_images": 0,
        "indexed_files": 0, "skipped_files": 0, "hash_seconds": 0.0,
        "extraction_seconds": 0.0, "image_extraction_seconds": 0.0,
        "decode_metadata_seconds": 0.0, "vision_preprocess_seconds": 0.0,
        "vision_seconds": 0.0, "ocr_seconds": 0.0, "image_embedding_seconds": 0.0,
        "embedding_seconds": 0.0, "database_seconds": 0.0,
        "image_index_seconds": 0.0, "image_bytes": 0,
    }


@pytest.mark.parametrize(
    ("extension", "save_format"),
    [
        ("png", "PNG"), ("webp", "WEBP"), ("gif", "GIF"),
        ("bmp", "BMP"), ("tiff", "TIFF"), ("heic", "HEIF"),
    ],
)
def test_supported_image_keeps_source_format_after_heif_opener_retry(
    local_fast, monkeypatch, tmp_path, extension, save_format
):
    from app.api.documents import list_documents
    from app.processors import image_processor

    class FakeOcr:
        def extract_text_from_image(self, image):
            assert image.mode == "RGB"
            return ["SERIAL-2026"]

    monkeypatch.setattr("app.services.ocr_service.ocr_service", FakeOcr())
    source = tmp_path / f"source.{extension}"
    Image.new("RGB", (32, 24), color="white").save(source, format=save_format)
    if extension == "heic":
        monkeypatch.setattr(image_processor, "_heif_registered", False)
        monkeypatch.setattr(
            image_processor, "_heif_registration_error", ModuleNotFoundError("pillow_heif")
        )
    original_bytes = source.read_bytes()
    observed_formats = []
    original_encode = local_fast.encode

    def encode(inputs, *args, **kwargs):
        if isinstance(inputs, dict):
            observed_formats.append(inputs["image"].format)
        return original_encode(inputs, *args, **kwargs)

    monkeypatch.setattr(local_fast, "encode", encode)
    session, engine = _db_session()
    try:
        indexed = indexer.index_single_file(str(source), session)
        assert indexed.file_type == indexer.get_file_type(str(source))
        assert indexed.filename == source.name
        assert indexed.filepath == str(source.resolve())
        assert indexed.file_hash == hashlib.sha256(original_bytes).hexdigest()
        assert "SERIAL-2026" in indexed.extracted_text
        assert f"Format: {save_format}" in indexed.extracted_text
        assert observed_formats == [None if extension == "gif" else save_format]
        if extension == "heic":
            assert image_processor._heif_registered
            assert image_processor._heif_registration_error is None
        assert source.read_bytes() == original_bytes
        assert not source.with_suffix(".jpg").exists()

        original_id = indexed.id
        again = indexer.index_single_file(str(source), session)
        assert again.id == original_id
        assert observed_formats == [None if extension == "gif" else save_format]

        filtered = list_documents(
            search=None, file_type=extension, limit=100, offset=0, db=session
        )
        assert [result.id for result in filtered] == [original_id]
    finally:
        session.close()
        engine.dispose()


def test_local_fast_embeddings_use_documented_prompts_and_skip_remote(local_fast, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("remote embedding provider must not be called")

    monkeypatch.setattr(embedding_service, "_embed_gemini", unexpected)
    monkeypatch.setattr(embedding_service, "_embed_jina", unexpected)
    query, query_provenance = embedding_service.get_embedding_with_provenance(
        "find the project plan", task="retrieval.query"
    )
    passage, passage_provenance = embedding_service.get_embedding_with_provenance(
        "The project plan is ready.", task="retrieval.passage"
    )

    assert len(query) == len(passage) == 768
    assert query_provenance["provider"] == passage_provenance["provider"] == "embeddinggemma"
    assert local_fast.calls[0][1] == "SearchQuery"
    assert local_fast.calls[1][1] == "Document"
    assert embedding_service.embedding_signature_is_compatible(passage_provenance, query_provenance)


def test_embeddinggemma_loads_once_from_local_cache(monkeypatch):
    model = FakeEmbeddingModel()
    constructors = []

    def constructor(*args, **kwargs):
        constructors.append((args, kwargs))
        return model

    import torch

    monkeypatch.setattr(settings, "LOCAL_FAST_MODE", True)
    monkeypatch.setattr(embedding_service, "_local_model", None)
    monkeypatch.setattr(embedding_service, "_local_model_config", None)
    monkeypatch.setattr(embedding_service, "_embeddinggemma_runtime_config", lambda: ("cpu", torch.float32, "float32"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))

    first = embedding_service._get_embeddinggemma_model()
    second = embedding_service._get_embeddinggemma_model()

    assert first is second is model
    assert len(constructors) == 1
    assert constructors[0][1]["local_files_only"] is True
    assert constructors[0][1]["revision"] == settings.LOCAL_EMBEDDING_MODEL_REVISION
    assert constructors[0][1]["model_kwargs"]["torch_dtype"] is torch.float32


def test_batch_mapping_modified_file_and_unchanged_skip(local_fast, monkeypatch, tmp_path):
    files = [tmp_path / "alpha.txt", tmp_path / "beta.txt"]
    files[0].write_text("alpha source", encoding="utf-8")
    files[1].write_text("beta source", encoding="utf-8")
    calls = []

    def process(path):
        text = Path(path).read_text(encoding="utf-8")
        calls.append(Path(path).name)
        return {
            "extracted_text": f"document::{text}",
            "chunks": [{"content": f"chunk::{text}", "page_number": 7}],
        }

    monkeypatch.setattr(indexer, "process_file", process)
    session, engine = _db_session()
    try:
        file_infos = indexer.LocalFileSource().scan(str(tmp_path))
        metrics = _metrics()
        indexer._index_folder_local_fast(session, file_infos, metrics)

        docs = {doc.filename: doc for doc in session.query(Document).all()}
        assert set(docs) == {"alpha.txt", "beta.txt"}
        for name, source in (("alpha.txt", "alpha source"), ("beta.txt", "beta source")):
            doc = docs[name]
            assert doc.get_embedding() == pytest.approx(FakeEmbeddingModel._vector(f"document::{source}"))
            chunk = session.query(Chunk).filter_by(document_id=doc.id).one()
            assert chunk.content == f"chunk::{source}"
            assert chunk.get_embedding() == pytest.approx(FakeEmbeddingModel._vector(f"chunk::{source}"))

        ids = {name: doc.id for name, doc in docs.items()}
        calls_before = len(calls)
        second_metrics = _metrics()
        indexer._index_folder_local_fast(session, file_infos, second_metrics)
        assert len(calls) == calls_before
        assert second_metrics["skipped_files"] == 2

        files[0].write_text("alpha source modified", encoding="utf-8")
        modified_infos = indexer.LocalFileSource().scan(str(tmp_path))
        third_metrics = _metrics()
        indexer._index_folder_local_fast(session, modified_infos, third_metrics)
        updated = session.query(Document).filter_by(filename="alpha.txt").one()
        assert updated.id == ids["alpha.txt"]
        assert updated.extracted_text == "document::alpha source modified"
        assert third_metrics["indexed_files"] == 1
    finally:
        session.close()
        engine.dispose()


def test_failed_extraction_can_be_retried(local_fast, monkeypatch, tmp_path):
    file_path = tmp_path / "retry.txt"
    file_path.write_text("retry me", encoding="utf-8")
    attempts = 0

    def process(path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary read failure")
        return {"extracted_text": "retry succeeded", "chunks": [{"content": "retry succeeded"}]}

    monkeypatch.setattr(indexer, "process_file", process)
    session, engine = _db_session()
    try:
        info = indexer.LocalFileSource().scan(str(tmp_path))
        first = _metrics()
        indexer._index_folder_local_fast(session, info, first)
        assert first["failed_files"] == 1
        assert session.query(Document).count() == 0

        second = _metrics()
        indexer._index_folder_local_fast(session, info, second)
        assert second["failed_files"] == 0
        assert session.query(Document).one().extracted_text == "retry succeeded"
    finally:
        session.close()
        engine.dispose()


def test_text_extraction_workers_are_bounded(local_fast, monkeypatch, tmp_path):
    for index in range(4):
        (tmp_path / f"worker-{index}.txt").write_text(f"worker {index}", encoding="utf-8")
    lock = threading.Lock()
    active = 0
    peak_active = 0

    def process(path):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.04)
        text = Path(path).read_text(encoding="utf-8")
        with lock:
            active -= 1
        return {"extracted_text": text, "chunks": [{"content": text}]}

    monkeypatch.setattr(indexer, "process_file", process)
    session, engine = _db_session()
    try:
        metrics = _metrics()
        indexer._index_folder_local_fast(session, indexer.LocalFileSource().scan(str(tmp_path)), metrics)
        assert metrics["indexed_files"] == 4
        assert peak_active == 2
    finally:
        session.close()
        engine.dispose()


def test_cuda_oom_retries_text_batch_in_smaller_groups(local_fast):
    local_fast.fail_large_batch_once = True
    values = [f"text {index}" for index in range(4)]
    result = embedding_service.embed_texts_with_provenance(values)

    assert len(result) == 4
    assert all(len(vector) == 768 for vector, _ in result)
    assert [len(call[0]) for call in local_fast.calls] == [4, 2, 2]


def test_embedding_cache_keys_include_task_model_revision_and_precision(local_fast, monkeypatch):
    document_key = embedding_service._embeddinggemma_cache_key("same text", "retrieval.passage")
    query_key = embedding_service._embeddinggemma_cache_key("same text", "retrieval.query")
    monkeypatch.setattr(settings, "LOCAL_EMBEDDING_MODEL_REVISION", "different-revision")
    revision_key = embedding_service._embeddinggemma_cache_key("same text", "retrieval.passage")
    monkeypatch.setattr(embedding_service, "_embeddinggemma_runtime_config", lambda: ("cpu", None, "different-precision"))
    precision_key = embedding_service._embeddinggemma_cache_key("same text", "retrieval.passage")

    assert len({document_key, query_key, revision_key, precision_key}) == 4


def test_identical_text_reuses_vectors_only_for_same_task_configuration(local_fast):
    first = embedding_service.embed_texts_with_provenance(["duplicate passage", "duplicate passage"])
    assert len(local_fast.calls) == 1
    assert local_fast.calls[0][0] == ["duplicate passage"]
    assert first[0][0] == first[1][0]

    embedding_service.embed_texts_with_provenance(["duplicate passage"])
    assert len(local_fast.calls) == 1

    embedding_service.embed_texts_with_provenance(["duplicate passage"], task="retrieval.query")
    assert len(local_fast.calls) == 2
    assert local_fast.calls[-1][1] == "SearchQuery"


def test_local_fast_ocr_uses_bundled_engine_without_paddle_or_network(monkeypatch, tmp_path):
    from app.services.ocr_service import OCRService

    class FakeRapidOCR:
        def __init__(self):
            self.calls = 0

        def __call__(self, _image):
            self.calls += 1
            return ([[None, "OFFLINE OCR 81", 0.99]], None)

    rapid = FakeRapidOCR()
    service = OCRService()
    monkeypatch.setattr(settings, "LOCAL_FAST_MODE", True)
    monkeypatch.setattr(service, "_get_rapidocr", lambda: rapid)
    monkeypatch.setattr(service, "_get_paddleocr", lambda: pytest.fail("PaddleOCR may download model weights"))
    image_path = tmp_path / "offline.png"
    Image.new("RGB", (20, 20), "white").save(image_path)

    first = service.extract_text_from_image(str(image_path))
    second = service.extract_text_from_image(str(image_path))

    assert first == second == ["OFFLINE OCR 81"]
    assert rapid.calls == 1


def test_scanned_pdf_uses_local_ocr_in_local_fast_mode(monkeypatch):
    import io
    from app.services.ocr_service import OCRService

    class Pixmap:
        def tobytes(self, _format):
            buffer = io.BytesIO()
            Image.new("RGB", (24, 24), "white").save(buffer, format="PNG")
            return buffer.getvalue()

    class FakeRapidOCR:
        def __call__(self, _image):
            return ([[None, "SCANNED PDF TEXT", 0.99]], None)

    service = OCRService()
    monkeypatch.setattr(settings, "LOCAL_FAST_MODE", True)
    monkeypatch.setattr(service, "_get_rapidocr", lambda: FakeRapidOCR())
    monkeypatch.setattr(service, "_get_paddleocr", lambda: pytest.fail("PaddleOCR may download model weights"))

    assert service.extract_text_from_scanned_pdf_page(Pixmap()) == "SCANNED PDF TEXT"


def test_pdf_processor_ocr_is_only_used_for_pages_with_insufficient_text(monkeypatch, tmp_path):
    import fitz
    from app.processors.pdf_processor import process_pdf
    from app.services.ocr_service import ocr_service

    class FakeRapidOCR:
        def __init__(self):
            self.calls = 0

        def __call__(self, _image):
            self.calls += 1
            return ([[None, "SCANNED PAGE OCR TEXT IS AVAILABLE", 0.99]], None)

    fake_ocr = FakeRapidOCR()
    monkeypatch.setattr(settings, "LOCAL_FAST_MODE", True)
    monkeypatch.setattr(ocr_service, "_get_rapidocr", lambda: fake_ocr)

    scanned_pdf = tmp_path / "scanned.pdf"
    document = fitz.open()
    document.new_page()
    document.save(scanned_pdf)
    document.close()
    extracted = process_pdf(str(scanned_pdf))
    assert "SCANNED PAGE OCR TEXT IS AVAILABLE" in extracted["extracted_text"]
    assert fake_ocr.calls == 1

    text_pdf = tmp_path / "digital.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "This PDF page already has sufficient searchable text content.")
    document.save(text_pdf)
    document.close()
    before = fake_ocr.calls
    text_result = process_pdf(str(text_pdf))
    assert "sufficient searchable text" in text_result["extracted_text"]
    assert fake_ocr.calls == before


def test_image_local_fast_keeps_ocr_and_never_calls_gemini(local_fast, monkeypatch, tmp_path):
    from app.processors import image_processor
    from app.services import ocr_service

    image_path = tmp_path / "readable.png"
    Image.new("RGB", (32, 24), "white").save(image_path)
    monkeypatch.setattr(settings, "LOCAL_FAST_MODE", True)
    monkeypatch.setattr(image_processor, "_analyze_image_payload_with_gemini", lambda *_: pytest.fail("Gemini API called"))
    monkeypatch.setattr(ocr_service.ocr_service, "extract_text_from_image", lambda _image: ["OCR EXACT 42"])

    result = image_processor.process_image(str(image_path))

    assert "OCR EXACT 42" in result["extracted_text"]
    assert len(result["image_embedding"]) == 768
    assert result["image_embedding_provenance"]["provider"] == "embeddinggemma"
    assert local_fast.calls[-1][0]["image"].mode == "RGB"

