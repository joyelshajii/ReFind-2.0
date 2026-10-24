from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from io import BytesIO

from PIL import Image
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi import HTTPException, UploadFile

from app.processors import image_processor
from app.services import embedding_service
from app.services.ocr_service import OCRService
from app.db.base import Base
from app.db.models import Document, Chunk
from app.services import indexer
from app.services import hybrid_search
from app.schemas.search import QueryClues
from app.services.indexer import _processing_signature
from app.api import index as index_api
from app.core.config import settings


@pytest.fixture(autouse=True)
def force_legacy_remote_pipeline_mode(monkeypatch):
    """Image pipeline tests verify PaddleOCR, Gemini vision, and remote fallback semantics."""
    monkeypatch.setattr(settings, "LOCAL_FAST_MODE", False)
    monkeypatch.setattr(indexer.settings, "LOCAL_FAST_MODE", False)
    monkeypatch.setattr(embedding_service.settings, "LOCAL_FAST_MODE", False)
    monkeypatch.setattr(image_processor.settings, "LOCAL_FAST_MODE", False)


@pytest.fixture
def isolated_index_db(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(indexer, "SessionLocal", sessions)
    monkeypatch.setattr(indexer.settings, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(indexer.settings, "EMBEDDING_DIMENSION", 2)
    monkeypatch.setattr(indexer.settings, "IMAGE_VISION_ENABLED", False)
    monkeypatch.setattr(indexer.settings, "GEMINI_API_KEY", "")
    yield sessions
    engine.dispose()


def _fake_image_processor(path: str):
    content = Path(path).read_bytes().decode("utf-8", errors="replace")
    return {
        "extracted_text": content,
        "chunks": [{"content": content, "page_number": 1, "slide_number": None}],
        "entities": [],
        "timings": {},
    }


def _fake_embedding_with_provenance(text):
    return [1.0, 0.0], embedding_service._embedding_provenance(
        "local", "local-hash-ngram-v1", 2, "retrieval.passage"
    )


def test_ocr_reuses_decoded_pixels_and_cache(monkeypatch):
    service = OCRService()
    calls = []

    class FakeOcr:
        def ocr(self, image_array, cls=True):
            calls.append(image_array.shape)
            return [[[[], ("Invoice 0427", 0.99)]]]

    monkeypatch.setattr(service, "_get_paddleocr", lambda: FakeOcr())
    first = Image.new("RGB", (32, 24), color=(255, 255, 255))
    second = first.copy()

    assert service.extract_text_from_image(first) == ["Invoice 0427"]
    assert service.extract_text_from_image(second) == ["Invoice 0427"]
    assert calls == [(24, 32, 3)]


def test_ocr_cache_key_includes_engine_profile(monkeypatch):
    service = OCRService()
    calls = []

    class FakeOcr:
        def ocr(self, image_array, cls=True):
            calls.append(image_array.shape)
            return [[[[], ("Profile-aware OCR", 0.99)]]]

    monkeypatch.setattr(service, "_get_paddleocr", lambda: FakeOcr())
    image = Image.new("RGB", (30, 20), color="white")
    monkeypatch.setattr("app.services.ocr_service._OCR_PROFILE", "profile-v1")
    service.extract_text_from_image(image)
    monkeypatch.setattr("app.services.ocr_service._OCR_PROFILE", "profile-v2")
    service.extract_text_from_image(image)
    assert len(calls) == 2


def test_embedding_cache_keys_provider_model_and_text(monkeypatch):
    embedding_service._embedding_cache.clear()
    embedding_service._local_embedding_cache.clear()
    monkeypatch.setattr(embedding_service.settings, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(embedding_service.settings, "GEMINI_API_KEY", "unit-test-key")
    monkeypatch.setattr(embedding_service.settings, "GEMINI_EMBEDDING_MODEL", "test-embedding-model")
    calls = []

    def fake_embed(text, dim):
        calls.append((text, dim))
        return [0.25] * dim

    monkeypatch.setattr(embedding_service, "_embed_gemini", fake_embed)
    first = embedding_service.get_embedding("same image description")
    second = embedding_service.get_embedding("same image description")
    monkeypatch.setattr(embedding_service.settings, "GEMINI_EMBEDDING_MODEL", "test-embedding-model-v2")
    third = embedding_service.get_embedding("same image description")

    assert first == second
    assert third == first
    assert calls == [
        ("same image description", embedding_service.DIMENSION),
        ("same image description", embedding_service.DIMENSION),
    ]


def test_embedding_provenance_tracks_remote_fallback_and_recovery(monkeypatch):
    embedding_service._embedding_cache.clear()
    embedding_service._local_embedding_cache.clear()
    embedding_service._embedding_key_locks.clear()
    monkeypatch.setattr(embedding_service.settings, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(embedding_service.settings, "GEMINI_API_KEY", "unit-test-key")
    monkeypatch.setattr(embedding_service.settings, "EMBEDDING_DIMENSION", 4)
    monkeypatch.setattr(embedding_service.settings, "GEMINI_EMBEDDING_MODEL", "gemini-test")
    calls = []

    def intermittent_gemini(text, dim):
        calls.append(text)
        return None if len(calls) == 1 else [0.25] * dim

    monkeypatch.setattr(embedding_service, "_embed_gemini", intermittent_gemini)
    fallback_vector, fallback = embedding_service.get_embedding_with_provenance("retry me")
    remote_vector, remote = embedding_service.get_embedding_with_provenance("retry me")

    assert len(fallback_vector) == len(remote_vector) == 4
    assert fallback["provider"] == "local"
    assert fallback["model"] == "local-hash-ngram-v1"
    assert remote["provider"] == "gemini"
    assert remote["model"] == "gemini-test"
    assert remote["dimension"] == 4
    assert remote["task"] == "retrieval.passage"
    assert len(calls) == 2
    assert not embedding_service._embedding_key_locks


def test_concurrent_identical_embedding_requests_share_one_remote_call(monkeypatch):
    embedding_service._embedding_cache.clear()
    embedding_service._local_embedding_cache.clear()
    embedding_service._embedding_key_locks.clear()
    monkeypatch.setattr(embedding_service.settings, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(embedding_service.settings, "GEMINI_API_KEY", "unit-test-key")
    monkeypatch.setattr(embedding_service.settings, "EMBEDDING_DIMENSION", 4)
    started = threading.Event()
    finish = threading.Event()
    calls = []

    def slow_remote(text, dim):
        calls.append(text)
        started.set()
        assert finish.wait(timeout=5)
        return [0.5] * dim

    monkeypatch.setattr(embedding_service, "_embed_gemini", slow_remote)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(
            embedding_service.get_embedding_with_provenance, "same request"
        ) for _ in range(4)]
        assert started.wait(timeout=5)
        finish.set()
        results = [future.result(timeout=5) for future in futures]

    assert calls == ["same request"]
    assert all(vector == results[0][0] and provenance == results[0][1]
               for vector, provenance in results)
    assert not embedding_service._embedding_key_locks


def test_embedding_provenance_records_jina_task_and_rejects_incompatible_vectors(monkeypatch):
    monkeypatch.setattr(embedding_service.settings, "EMBEDDING_PROVIDER", "jina")
    monkeypatch.setattr(embedding_service.settings, "JINA_API_KEY", "unit-test-key")
    monkeypatch.setattr(embedding_service.settings, "JINA_EMBEDDING_MODEL", "jina-test")
    monkeypatch.setattr(embedding_service.settings, "JINA_EMBEDDING_DIMENSION", 4)
    monkeypatch.setattr(embedding_service, "_embed_jina", lambda text, task: [0.5] * 4)
    _, passage = embedding_service.get_embedding_with_provenance("retrievable", "retrieval.passage")
    _, query = embedding_service.get_embedding_with_provenance("retrievable", "retrieval.query")
    assert passage["provider"] == "jina"
    assert passage["task"] == "retrieval.passage"
    assert embedding_service.embedding_signature_is_compatible(passage, query)

    other_model = dict(query, model="another-model")
    other_dimension = dict(query, dimension=5)
    other_config = dict(query, config_id="different")
    assert not embedding_service.embedding_signature_is_compatible(passage, other_model)
    assert not embedding_service.embedding_signature_is_compatible(passage, other_dimension)
    assert not embedding_service.embedding_signature_is_compatible(passage, other_config)
    other_provider = dict(query, provider="gemini")
    assert not embedding_service.embedding_signature_is_compatible(passage, other_provider)
    assert not embedding_service.embedding_signature_is_compatible("", query)
    assert not embedding_service.embedding_signature_is_compatible("not-json", query)


def test_search_keeps_keyword_results_when_legacy_vector_provenance_is_unknown(
    isolated_index_db, tmp_path, monkeypatch
):
    file_path = tmp_path / "invoice.png"
    file_path.write_bytes(b"fixture")
    monkeypatch.setattr(hybrid_search, "get_embedding_with_provenance", lambda query, task: (
        [1.0, 0.0], embedding_service._embedding_provenance(
            "local", "local-hash-ngram-v1", 2, task
        )
    ))
    monkeypatch.setattr(hybrid_search.settings, "LLM_PROVIDER", "local")
    monkeypatch.setattr(hybrid_search.settings, "JINA_API_KEY", "")
    monkeypatch.setattr(hybrid_search.settings, "WEIGHT_SEMANTIC", 1.0)
    monkeypatch.setattr(hybrid_search.settings, "WEIGHT_RERANKER", 0.0)
    monkeypatch.setattr(hybrid_search.settings, "WEIGHT_KEYWORD", 0.0)
    monkeypatch.setattr(hybrid_search.settings, "WEIGHT_METADATA", 0.0)
    monkeypatch.setattr(hybrid_search.settings, "WEIGHT_FILENAME", 0.0)
    with isolated_index_db() as db:
        legacy = Document(
            filename=file_path.name,
            filepath=str(file_path.resolve()),
            file_type="png",
            extracted_text="Visible text: Invoice 0427",
            embedding_json="[1.0, 0.0]",
            embedding_provenance="",
        )
        db.add(legacy)
        db.commit()

        results = hybrid_search.search_documents(
            "Invoice 0427",
            QueryClues(raw_query="Invoice 0427", keywords=["Invoice 0427"], file_type="image"),
            db,
            top_k=3,
        )

    assert len(results) == 1
    assert results[0].filename == "invoice.png"
    assert results[0].score == 0.05  # Legacy vector was not compared; keyword retrieval still found it.


def test_search_rejects_recorded_incompatible_embedding_model(monkeypatch):
    monkeypatch.setattr(embedding_service.settings, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(embedding_service.settings, "EMBEDDING_DIMENSION", 768)
    monkeypatch.setattr(embedding_service.settings, "GEMINI_EMBEDDING_MODEL", "gemini-test-v1")
    signature = embedding_service._embedding_provenance(
        "gemini", "gemini-test-v1", 768, "retrieval.passage"
    )
    query = embedding_service._embedding_provenance(
        "gemini", "gemini-test-v1", 768, "retrieval.query"
    )
    assert embedding_service.embedding_signature_is_compatible(signature, query)

    monkeypatch.setattr(embedding_service.settings, "GEMINI_EMBEDDING_MODEL", "gemini-test-v2")
    query_after_model_change = embedding_service._embedding_provenance(
        "gemini", "gemini-test-v2", 768, "retrieval.query"
    )
    assert not embedding_service.embedding_signature_is_compatible(signature, query_after_model_change)
    assert not embedding_service.embedding_signature_is_compatible("", query_after_model_change)


def test_jina_embedding_cache_honors_retrieval_task_and_configuration(monkeypatch):
    embedding_service._embedding_cache.clear()
    embedding_service._local_embedding_cache.clear()
    monkeypatch.setattr(embedding_service.settings, "EMBEDDING_PROVIDER", "jina")
    monkeypatch.setattr(embedding_service.settings, "JINA_API_KEY", "unit-test-jina-key")
    monkeypatch.setattr(embedding_service.settings, "JINA_EMBEDDING_MODEL", "test-jina-model")
    monkeypatch.setattr(embedding_service.settings, "JINA_EMBEDDING_DIMENSION", 4)
    calls = []

    def fake_embed(text, task):
        calls.append((text, task, embedding_service.settings.JINA_EMBEDDING_MODEL))
        return [float(len(calls))] * embedding_service.settings.JINA_EMBEDDING_DIMENSION

    monkeypatch.setattr(embedding_service, "_embed_jina", fake_embed)
    passage = embedding_service.get_embedding("same query text", task="retrieval.passage")
    assert embedding_service.get_embedding("same query text", task="retrieval.passage") == passage
    query = embedding_service.get_embedding("same query text", task="retrieval.query")
    monkeypatch.setattr(embedding_service.settings, "JINA_EMBEDDING_MODEL", "test-jina-model-v2")
    other_model = embedding_service.get_embedding("same query text", task="retrieval.query")
    monkeypatch.setattr(embedding_service.settings, "JINA_EMBEDDING_DIMENSION", 8)
    other_dimension = embedding_service.get_embedding("same query text", task="retrieval.query")

    assert len(passage) == len(query) == len(other_model) == 4
    assert len(other_dimension) == 8
    assert calls == [
        ("same query text", "retrieval.passage", "test-jina-model"),
        ("same query text", "retrieval.query", "test-jina-model"),
        ("same query text", "retrieval.query", "test-jina-model-v2"),
        ("same query text", "retrieval.query", "test-jina-model-v2"),
    ]


def test_image_index_skip_retry_and_modified_file(isolated_index_db, tmp_path, monkeypatch):
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"first image content")
    process_calls = []
    should_fail = True

    def process(path):
        nonlocal should_fail
        process_calls.append(Path(path).read_bytes())
        if should_fail:
            raise RuntimeError("temporary image decode failure")
        return _fake_image_processor(path)

    monkeypatch.setattr(indexer, "process_file", process)
    monkeypatch.setattr(indexer, "get_embedding_with_provenance", _fake_embedding_with_provenance)

    indexer.index_folder_background(str(tmp_path))
    assert indexer.indexing_status.current_stage == "Error"
    with isolated_index_db() as db:
        assert db.query(Document).count() == 0

    should_fail = False
    indexer.index_folder_background(str(tmp_path))
    with isolated_index_db() as db:
        first_doc = db.query(Document).one()
        original_id = first_doc.id
        assert first_doc.extracted_text == "first image content"
        assert first_doc.file_hash
        assert first_doc.processing_signature

    calls_after_retry = len(process_calls)
    indexer.index_folder_background(str(tmp_path))
    assert len(process_calls) == calls_after_retry

    image_path.write_bytes(b"modified image content")
    indexer.index_folder_background(str(tmp_path))
    with isolated_index_db() as db:
        modified_doc = db.query(Document).one()
        assert modified_doc.id != original_id
        assert modified_doc.extracted_text == "modified image content"
        assert modified_doc.file_hash != ""
    assert len(process_calls) == calls_after_retry + 1


def test_indexed_fallback_vector_is_rebuilt_after_provider_recovers(
    isolated_index_db, tmp_path, monkeypatch
):
    file_path = tmp_path / "recovery.txt"
    file_path.write_text("same indexed content", encoding="utf-8")
    monkeypatch.setattr(indexer.settings, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(indexer.settings, "GEMINI_API_KEY", "unit-test-key")
    monkeypatch.setattr(indexer.settings, "EMBEDDING_DIMENSION", 2)
    process_calls = []

    def process(path):
        process_calls.append(path)
        return {"extracted_text": Path(path).read_text(), "chunks": [], "entities": []}

    monkeypatch.setattr(indexer, "process_file", process)
    local = embedding_service._embedding_provenance(
        "local", "local-hash-ngram-v1", 2, "retrieval.passage"
    )
    remote = embedding_service._embedding_provenance(
        "gemini", indexer.settings.GEMINI_EMBEDDING_MODEL, 2, "retrieval.passage"
    )
    provenance_results = iter([([1.0, 0.0], local), ([0.0, 1.0], remote)])
    monkeypatch.setattr(
        indexer,
        "embed_texts_with_provenance",
        lambda texts, task="retrieval.passage": [next(provenance_results) for _ in texts],
    )

    with isolated_index_db() as db:
        first = indexer.index_single_file(str(file_path), db)
        assert first.get_embedding_provenance()["provider"] == "local"
        recovered = indexer.index_single_file(str(file_path), db)
        assert recovered.id == first.id
        assert recovered.get_embedding_provenance()["provider"] == "gemini"
        assert len(process_calls) == 1


def test_overlapping_folder_index_runs_do_not_repeat_work(isolated_index_db, tmp_path, monkeypatch):
    image_path = tmp_path / "one.png"
    image_path.write_bytes(b"one image")
    started = threading.Event()
    finish = threading.Event()
    calls = []

    def slow_process(path):
        calls.append(path)
        started.set()
        assert finish.wait(timeout=5)
        return _fake_image_processor(path)

    monkeypatch.setattr(indexer, "process_file", slow_process)
    monkeypatch.setattr(indexer, "get_embedding_with_provenance", _fake_embedding_with_provenance)

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_run = executor.submit(indexer.index_folder_background, str(tmp_path))
        assert started.wait(timeout=5)
        indexer.index_folder_background(str(tmp_path))
        finish.set()
        first_run.result(timeout=5)

    assert calls == [str(image_path.resolve())]
    with isolated_index_db() as db:
        assert db.query(Document).count() == 1


def test_upload_waits_for_folder_scan_before_replacing_file(isolated_index_db, tmp_path, monkeypatch):
    image_path = tmp_path / "race.png"
    image_path.write_bytes(b"folder version")
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(indexer, "get_embedding_with_provenance", _fake_embedding_with_provenance)
    process_started = threading.Event()
    upload_started = threading.Event()
    release_folder = threading.Event()

    def slow_process(path):
        process_started.set()
        assert release_folder.wait(timeout=5)
        return _fake_image_processor(path)

    monkeypatch.setattr(indexer, "process_file", slow_process)

    def upload():
        upload_started.set()
        with isolated_index_db() as db:
            return index_api.upload_and_index_files(
                [UploadFile(filename="race.png", file=BytesIO(b"uploaded version"))], db
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        folder_future = executor.submit(indexer.index_folder_background, str(tmp_path))
        assert process_started.wait(timeout=5)
        upload_future = executor.submit(upload)
        assert upload_started.wait(timeout=5)
        with pytest.raises(HTTPException) as busy:
            upload_future.result(timeout=5)
        assert busy.value.status_code == 409
        assert image_path.read_bytes() == b"folder version"
        release_folder.set()
        folder_future.result(timeout=10)
        result = upload()

    assert result["status"] == "success"
    with isolated_index_db() as db:
        assert db.query(Document).count() == 1
        document = db.query(Document).one()
        assert document.extracted_text == "uploaded version"
        assert db.query(Chunk).count() == 1
        assert db.query(Chunk).one().document_id == document.id


def test_indexing_different_paths_remains_concurrent(tmp_path, monkeypatch):
    db_path = tmp_path / "parallel-index.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(indexer, "get_embedding_with_provenance", _fake_embedding_with_provenance)
    paths = [tmp_path / "first.txt", tmp_path / "second.txt"]
    for path in paths:
        path.write_text(path.stem, encoding="utf-8")
    both_processing = threading.Barrier(2)

    def process(path):
        both_processing.wait(timeout=5)
        return {"extracted_text": Path(path).read_text(), "chunks": [], "entities": []}

    monkeypatch.setattr(indexer, "process_file", process)

    def run(path):
        with sessions() as db:
            return indexer.index_single_file(str(path), db)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(run, paths))
        assert {doc.filename for doc in results} == {"first.txt", "second.txt"}
        with sessions() as db:
            assert db.query(Document).count() == 2
    finally:
        engine.dispose()


def test_legacy_image_is_reprocessed_and_database_survives_failed_commit(
    isolated_index_db, tmp_path, monkeypatch
):
    image_path = tmp_path / "legacy.png"
    image_path.write_bytes(b"new image text")
    monkeypatch.setattr(indexer, "process_file", _fake_image_processor)
    monkeypatch.setattr(indexer, "get_embedding_with_provenance", _fake_embedding_with_provenance)
    resolved = str(image_path.resolve())

    with isolated_index_db() as db:
        legacy = Document(
            filename=image_path.name,
            filepath=resolved,
            file_type="png",
            file_hash=indexer.compute_file_hash(resolved),
            processing_signature="",
            extracted_text="old indexed text",
            embedding_json="[1.0, 0.0]",
        )
        db.add(legacy)
        db.commit()
        legacy_id = legacy.id

        original_commit = db.commit
        should_fail = True

        def fail_once():
            nonlocal should_fail
            if should_fail:
                should_fail = False
                raise RuntimeError("simulated database commit failure")
            original_commit()

        monkeypatch.setattr(db, "commit", fail_once)
        with pytest.raises(RuntimeError, match="simulated database commit failure"):
            indexer.index_single_file(resolved, db)
        db.rollback()
        surviving = db.query(Document).one()
        assert surviving.id == legacy_id
        assert surviving.extracted_text == "old indexed text"

        monkeypatch.setattr(db, "commit", original_commit)
        refreshed = indexer.index_single_file(resolved, db)
        assert refreshed.id == legacy_id
        assert refreshed.extracted_text == "new image text"
        assert refreshed.processing_signature


def test_identical_gemini_vision_payload_is_analyzed_once(monkeypatch):
    image_processor._vision_result_cache.clear()
    calls = []

    def fake_analysis(payload, model):
        calls.append((payload, model))
        return {"visual_description": "a lake", "detected_text": [], "entities": []}

    monkeypatch.setattr(image_processor, "_analyze_image_payload_with_gemini", fake_analysis)
    first = image_processor._cached_gemini_vision("same-payload", "test-model", b"identical bytes")
    second = image_processor._cached_gemini_vision("same-payload", "test-model", b"identical bytes")

    assert first == second
    assert len(calls) == 1


def test_gemini_image_resize_and_vision_results_are_reused(monkeypatch):
    image_processor._vision_result_cache.clear()
    image_processor._resized_image_cache.clear()
    monkeypatch.setattr(image_processor.settings, "IMAGE_VISION_ENABLED", True)
    monkeypatch.setattr(image_processor.settings, "GEMINI_API_KEY", "unit-test-key")
    calls = []

    def fake_analysis(payload, model):
        calls.append((payload, model))
        return {"visual_description": "a red canoe", "detected_text": [], "entities": []}

    monkeypatch.setattr(image_processor, "_analyze_image_payload_with_gemini", fake_analysis)
    first = Image.new("RGB", (128, 96), color=(10, 20, 30))
    second = first.copy()

    result_one, _ = image_processor._analyze_image_with_gemini(first)
    result_two, _ = image_processor._analyze_image_with_gemini(second)
    assert result_one == result_two
    assert len(calls) == 1
    assert len(image_processor._resized_image_cache) == 1


def test_vision_cache_distinguishes_image_and_model(monkeypatch):
    image_processor._vision_result_cache.clear()
    image_processor._resized_image_cache.clear()
    monkeypatch.setattr(image_processor.settings, "IMAGE_VISION_ENABLED", True)
    monkeypatch.setattr(image_processor.settings, "GEMINI_API_KEY", "unit-test-key")
    monkeypatch.setattr(image_processor.settings, "GEMINI_MODEL", "vision-model-a")
    calls = []

    def fake_analysis(payload, model):
        calls.append((payload, model))
        return {"visual_description": model, "detected_text": [], "entities": []}

    monkeypatch.setattr(image_processor, "_analyze_image_payload_with_gemini", fake_analysis)
    first = Image.new("RGB", (20, 20), color="red")
    different_image = Image.new("RGB", (20, 20), color="blue")
    image_processor._analyze_image_with_gemini(first)
    image_processor._analyze_image_with_gemini(different_image)
    monkeypatch.setattr(image_processor.settings, "GEMINI_MODEL", "vision-model-b")
    image_processor._analyze_image_with_gemini(first)

    assert len(calls) == 3
    assert {model for _, model in calls} == {"vision-model-a", "vision-model-b"}


def test_image_processing_signature_tracks_companion_metadata(tmp_path, monkeypatch):
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"same-image-bytes")
    monkeypatch.setattr(image_processor.settings, "IMAGE_VISION_ENABLED", False)
    before = _processing_signature(str(image_path), "png")

    sidecar = Path(f"{image_path}.meta.json")
    sidecar.write_text('{"visual_description":"first"}', encoding="utf-8")
    with_sidecar = _processing_signature(str(image_path), "png")
    sidecar.write_text('{"visual_description":"updated"}', encoding="utf-8")
    after_edit = _processing_signature(str(image_path), "png")

    assert before != with_sidecar
    assert with_sidecar != after_edit


def test_image_processor_keeps_local_ocr_text_when_vision_is_off(tmp_path, monkeypatch):
    image_path = tmp_path / "diagram.png"
    Image.new("RGB", (48, 32), color="white").save(image_path)
    monkeypatch.setattr(image_processor.settings, "IMAGE_VISION_ENABLED", False)

    class FakeOcrService:
        def extract_text_from_image(self, image):
            assert isinstance(image, Image.Image)
            return ["MQTT-0427", "Battery telemetry"]

    monkeypatch.setattr(
        "app.services.ocr_service.ocr_service", FakeOcrService()
    )
    result = image_processor.process_image(str(image_path))

    assert "MQTT-0427" in result["extracted_text"]
    assert result["ocr_text"] == "MQTT-0427\nBattery telemetry"
    assert "Visual Description:" not in result["extracted_text"]


@pytest.mark.parametrize(
    ("vision_result", "local_lines", "expected_lines"),
    [
        (
            {"visual_description": "invoice", "detected_text": ["Invoice 0427", "Total 39.50"], "entities": []},
            ["Invoice 0427", "Total 39.50"],
            ["Invoice 0427", "Total 39.50"],
        ),
        (
            {"visual_description": "invoice", "detected_text": ["Invoice 0427"], "entities": []},
            ["Invoice 0427", "Total 39.50"],
            ["Invoice 0427", "Total 39.50"],
        ),
        (
            {"visual_description": "invoice", "detected_text": [], "entities": []},
            ["Invoice 0427", "Total 39.50"],
            ["Invoice 0427", "Total 39.50"],
        ),
        (None, ["Diagram node: MQTT-0427"], ["Diagram node: MQTT-0427"]),
    ],
    ids=["complete-gemini", "partial-gemini", "empty-gemini", "gemini-failure"],
)
def test_local_ocr_supplements_gemini_transcription(
    tmp_path, monkeypatch, vision_result, local_lines, expected_lines
):
    image_path = tmp_path / "ocr-policy.png"
    Image.new("RGB", (64, 40), color="white").save(image_path)
    monkeypatch.setattr(image_processor.settings, "IMAGE_VISION_ENABLED", True)
    monkeypatch.setattr(image_processor.settings, "GEMINI_API_KEY", "unit-test-key")
    monkeypatch.setattr(image_processor.settings, "IMAGE_OCR_POLICY", "supplement")
    timings = {
        "decode_metadata_seconds": 0.0,
        "vision_preprocess_seconds": 0.0,
        "vision_seconds": 0.0,
        "ocr_seconds": 0.0,
    }
    monkeypatch.setattr(image_processor, "_analyze_image_with_gemini", lambda image: (vision_result, timings))
    ocr_calls = []

    class FakeOcr:
        def extract_text_from_image(self, image):
            assert isinstance(image, Image.Image)
            ocr_calls.append(True)
            return local_lines

    monkeypatch.setattr("app.services.ocr_service.ocr_service", FakeOcr())
    result = image_processor.process_image(str(image_path))
    assert result["ocr_text"].splitlines() == expected_lines
    assert len(ocr_calls) == 1


def test_fallback_ocr_policy_skips_local_ocr_when_gemini_has_text(tmp_path, monkeypatch):
    image_path = tmp_path / "fast-ocr.png"
    Image.new("RGB", (64, 40), color="white").save(image_path)
    monkeypatch.setattr(image_processor.settings, "IMAGE_VISION_ENABLED", True)
    monkeypatch.setattr(image_processor.settings, "GEMINI_API_KEY", "unit-test-key")
    monkeypatch.setattr(image_processor.settings, "IMAGE_OCR_POLICY", "fallback")
    monkeypatch.setattr(
        image_processor,
        "_analyze_image_with_gemini",
        lambda image: ({"visual_description": "", "detected_text": ["Ticket 17"], "entities": []}, {}),
    )

    class UnexpectedOcr:
        def extract_text_from_image(self, image):
            raise AssertionError("fallback policy should not run local OCR when Gemini returned text")

    monkeypatch.setattr("app.services.ocr_service.ocr_service", UnexpectedOcr())
    result = image_processor.process_image(str(image_path))
    assert result["ocr_text"] == "Ticket 17"
