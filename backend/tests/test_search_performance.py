import json
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.base import Base
from app.db.models import Document, Chunk
from app.services.embedding_service import cosine_similarity
from app.services.search_index import get_search_index, install_search_revision
from app.services.request_cache import RequestCache


PROVENANCE = {"provider": "embeddinggemma", "model": "fixture", "dimension": 2,
              "config_id": "test", "task": "retrieval.passage"}
QUERY = {**PROVENANCE, "task": "retrieval.query"}


@pytest.fixture
def corpus(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        install_search_revision(connection)
    engine._refind_search_revision = True
    session = sessionmaker(bind=engine)()
    source = tmp_path / "cars.txt"
    source.write_text("Automobiles carry passengers on roads.")
    doc = Document(filename=source.name, filepath=str(source), file_type="txt",
                   extracted_text=source.read_text())
    doc.set_embedding([0.6, 0.8])
    doc.set_embedding_provenance(PROVENANCE)
    chunk = Chunk(content=source.read_text(), chunk_index=0, page_number=7)
    chunk.set_embedding([1.0, 0.0])
    chunk.set_embedding_provenance(PROVENANCE)
    doc.chunks.append(chunk)
    session.add(doc)
    session.commit()
    yield session, engine, doc.id, chunk.id
    session.close()
    engine.dispose()


def test_exact_scores_and_best_passage_match_original_cosine(corpus):
    db, _, doc_id, _ = corpus
    index = get_search_index(db, QUERY)
    for query in ([1.0, 0.0], [0.2, 0.9], [-1.0, 0.0], [0.0, 0.0]):
        expected = max(cosine_similarity(query, vector) for vector in ([0.6, 0.8], [1, 0]))
        assert index.score(query)[doc_id][0] == pytest.approx(expected, abs=1e-6)
    assert index.score([1, 0])[doc_id][1] == "Page 7"
    assert "Automobiles" in index.score([1, 0])[doc_id][2]


def test_warm_search_reads_only_revision_and_external_changes_invalidate(corpus):
    db, engine, doc_id, chunk_id = corpus
    first = get_search_index(db, QUERY)
    statements = []
    def observe(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)
    event.listen(engine, "before_cursor_execute", observe)
    try:
        assert get_search_index(db, QUERY) is first
        assert len(statements) == 1
        assert "search_revision" in statements[0]
    finally:
        event.remove(engine, "before_cursor_execute", observe)
    db.rollback()
    with engine.begin() as connection:
        connection.execute(text("UPDATE chunks SET embedding_json = '[0,1]', content = 'Updated' WHERE id = :id"), {"id": chunk_id})
    changed = get_search_index(db, QUERY)
    assert changed is not first
    assert changed.score([1, 0])[doc_id][0] == pytest.approx(0.6)
    assert changed.documents[0].chunks[0].content == "Updated"


def test_rollback_never_publishes_uncommitted_snapshot(corpus):
    db, _, doc_id, _ = corpus
    original = get_search_index(db, QUERY)
    db.execute(text("UPDATE documents SET extracted_text='UNCOMMITTED' WHERE id=:id"), {"id": doc_id})
    assert get_search_index(db, QUERY).documents[0].extracted_text == "UNCOMMITTED"
    db.rollback()
    assert get_search_index(db, QUERY) is original
    db.execute(text("DELETE FROM documents WHERE id=:id"), {"id": doc_id})
    db.commit()
    assert not get_search_index(db, QUERY).documents


def test_model_change_does_not_reuse_incompatible_vectors(corpus):
    db, _, doc_id, _ = corpus
    get_search_index(db, QUERY)
    index = get_search_index(db, {**QUERY, "config_id": "different-model"})
    assert index.score([1, 0])[doc_id][0] == 0


def test_synonyms_are_retrieved_but_explicit_text_search_stays_literal(corpus, monkeypatch):
    from app.services import hybrid_search
    from app.services.query_parser import parse_query
    db, _, doc_id, _ = corpus
    monkeypatch.setattr(settings, "JINA_API_KEY", "")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "present-but-not-needed")
    monkeypatch.setattr(settings, "SEARCH_INLINE_EXPLANATIONS", False)
    monkeypatch.setattr(hybrid_search, "get_embedding_with_provenance", lambda *a, **kw: ([1, 0], QUERY))
    monkeypatch.setattr(hybrid_search.gemini_service, "explain_match", lambda **kw: pytest.fail("Search must not wait for explanations"))
    query = "Find information about vehicles and transportation"
    results = hybrid_search.search_documents(query, parse_query(query), db)
    assert results[0].document_id == doc_id
    assert "Semantic similarity" in results[0].matching_clues
    query = "Find the file containing the word vehicles"
    assert hybrid_search.search_documents(query, parse_query(query), db) == []
    query = 'Find "vehicles"'
    assert hybrid_search.search_documents(query, parse_query(query), db) == []


def test_provider_cache_deduplicates_concurrent_calls_and_copies_results():
    cache = RequestCache()
    calls = []
    def request():
        calls.append(1)
        time.sleep(0.03)
        return [{"score": 0.9}]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: cache.get_or_call("same", 30, request), range(4)))
    assert len(calls) == 1
    results[0][0]["score"] = 0
    assert cache.get_or_call("same", 30, request)[0]["score"] == 0.9


def test_pdf_ocr_consumes_pixels_without_png_roundtrip(monkeypatch):
    from app.services.ocr_service import OCRService
    service = OCRService()
    class Pixmap:
        width, height, n, alpha = 3, 2, 3, False
        samples = bytes([12, 34, 56]) * 6
        def tobytes(self, *_):
            pytest.fail("PNG encoding is redundant")
    def extract(image):
        assert image.size == (3, 2)
        assert image.getpixel((1, 1)) == (12, 34, 56)
        return ["Exact OCR 42"]
    monkeypatch.setattr(service, "extract_text_from_image", extract)
    assert service.extract_text_from_scanned_pdf_page(Pixmap()) == "Exact OCR 42"


def test_image_ocr_and_embedding_overlap_without_skipping_either(tmp_path, monkeypatch):
    from PIL import Image
    from app.processors import image_processor
    from app.services.ocr_service import ocr_service
    source = tmp_path / "parallel.png"
    Image.new("RGB", (24, 24), "white").save(source)
    barrier = threading.Barrier(2)
    def embed(image):
        barrier.wait(timeout=5)
        return [1, 0], PROVENANCE, 0.01
    def ocr(image):
        barrier.wait(timeout=5)
        return ["Invoice 0427"]
    monkeypatch.setattr(settings, "LOCAL_FAST_MODE", True)
    monkeypatch.setattr(image_processor, "_embed_image_timed", embed)
    monkeypatch.setattr(ocr_service, "extract_text_from_image", ocr)
    result = image_processor.process_image(str(source))
    assert result["image_embedding"] == [1, 0]
    assert result["ocr_text"] == "Invoice 0427"


def test_waiting_query_precedes_next_index_batch():
    from app.services import embedding_service as service
    order = []
    index_waiting = threading.Event()
    def indexing():
        index_waiting.set()
        with service._inference_slot():
            order.append("index")
    def query():
        with service._inference_slot(is_query=True):
            order.append("query")
    with ThreadPoolExecutor(max_workers=2) as pool:
        with service._inference_slot():
            indexed = pool.submit(indexing)
            assert index_waiting.wait(timeout=5)
            searched = pool.submit(query)
            with service._inference_condition:
                assert service._inference_condition.wait_for(lambda: service._waiting_queries > 0, timeout=5)
        searched.result(timeout=5)
        indexed.result(timeout=5)
    assert order == ["query", "index"]
