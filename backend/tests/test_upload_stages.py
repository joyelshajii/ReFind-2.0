import io
import tempfile
import shutil
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.db.base import Base, get_db
from app.db.models import Document
from app.services.indexer import indexing_status
from app.main import app

@pytest.fixture
def test_app():
    temp_dir = tempfile.mkdtemp(prefix="refind_upload_test_")
    db_path = str(Path(temp_dir) / "test.db")
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    yield client, temp_dir

    app.dependency_overrides.clear()
    engine.dispose()
    shutil.rmtree(temp_dir, ignore_errors=True)

def test_upload_and_index_stages(test_app, monkeypatch):
    client, temp_dir = test_app

    # Point uploads to temp directory
    upload_dir = Path(temp_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("app.core.config.settings.UPLOAD_DIR", str(upload_dir))

    # Mock embed_texts_with_provenance and process_file to return fast test data
    def mock_process_file(filepath):
        return {
            "extracted_text": "Sample uploaded file content for testing stages.",
            "chunks": [{"content": "Sample uploaded file content", "page_number": 1}],
            "entities": [{"entity_type": "technology", "entity_value": "Python"}],
        }

    def mock_embed(texts, task="retrieval.passage"):
        return [([0.1] * 768, {"config_id": "test"}) for _ in texts]

    monkeypatch.setattr("app.services.indexer.process_file", mock_process_file)
    monkeypatch.setattr("app.services.indexer.embed_texts_with_provenance", mock_embed)

    file_content = b"This is test document content for indexing stages."
    files = [
        ("files", ("test_sample.txt", io.BytesIO(file_content), "text/plain"))
    ]

    res = client.post("/api/index/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert len(data["files"]) == 1

    # Check that status reports Complete
    status_res = client.get("/api/index/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["current_stage"].lower() == "complete"
    assert status_data["total_files"] >= 1
    assert status_data["processed_files"] >= 1
    assert status_data["is_indexing"] is False
