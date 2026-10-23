from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

from PIL import Image
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.processors import image_processor
from app.services import embedding_service
from app.services.ocr_service import OCRService
from app.db.base import Base
from app.db.models import Document
from app.services import indexer
from app.services.indexer import _processing_signature


@pytest.fixture
def isolated_index_db(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(indexer, "SessionLocal", sessions)
    monkeypatch.setattr(indexer.settings, "EMBEDDING_PROVIDER", "local")
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

    assert len(passage) == len(query) == len(other_model) == 4
    assert calls == [
        ("same query text", "retrieval.passage", "test-jina-model"),
        ("same query text", "retrieval.query", "test-jina-model"),
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
    monkeypatch.setattr(indexer, "get_embedding", lambda text: [1.0, 0.0])

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
    monkeypatch.setattr(indexer, "get_embedding", lambda text: [1.0, 0.0])

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_run = executor.submit(indexer.index_folder_background, str(tmp_path))
        assert started.wait(timeout=5)
        indexer.index_folder_background(str(tmp_path))
        finish.set()
        first_run.result(timeout=5)

    assert calls == [str(image_path.resolve())]
    with isolated_index_db() as db:
        assert db.query(Document).count() == 1


def test_legacy_image_is_reprocessed_and_database_survives_failed_commit(
    isolated_index_db, tmp_path, monkeypatch
):
    image_path = tmp_path / "legacy.png"
    image_path.write_bytes(b"new image text")
    monkeypatch.setattr(indexer, "process_file", _fake_image_processor)
    monkeypatch.setattr(indexer, "get_embedding", lambda text: [1.0, 0.0])
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
        assert refreshed.id != legacy_id
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
