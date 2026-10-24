import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.db.base import Base, get_db
from app.db.models import Document
from app.services.duplicate_service import (
    find_duplicates,
    delete_single_duplicate,
    resolve_duplicate_group,
    resolve_all_duplicates,
)
from app.main import app

@pytest.fixture
def db_session():
    temp_dir = tempfile.mkdtemp(prefix="refind_dup_test_")
    db_path = str(Path(temp_dir) / "test.db")
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session, temp_dir

    session.close()
    engine.dispose()
    shutil.rmtree(temp_dir, ignore_errors=True)

def test_exact_hash_duplicates(db_session):
    session, temp_dir = db_session
    now = datetime.utcnow()

    # Create 2 files on disk with same content
    file1 = Path(temp_dir) / "file1.txt"
    file2 = Path(temp_dir) / "file2.txt"
    content = b"Exact same content for both files."
    file1.write_bytes(content)
    file2.write_bytes(content)

    doc1 = Document(
        id="doc_old",
        filename="file1.txt",
        filepath=str(file1),
        file_type="txt",
        file_size=len(content),
        file_hash="dummy_hash_12345",
        modified_at=now - timedelta(days=5),
        created_at=now - timedelta(days=5),
    )
    doc2 = Document(
        id="doc_new",
        filename="file2.txt",
        filepath=str(file2),
        file_type="txt",
        file_size=len(content),
        file_hash="dummy_hash_12345",
        modified_at=now,
        created_at=now,
    )
    session.add_all([doc1, doc2])
    session.commit()

    res = find_duplicates(session)
    assert res.total_groups == 1
    assert res.total_duplicate_files == 1
    group = res.groups[0]
    assert group.match_type == "exact_hash"
    assert group.recommended_document_id == "doc_new"
    assert group.potential_savings_bytes == len(content)
    assert "file1.txt" in group.differences or "file2.txt" in group.differences

def test_exact_text_duplicates(db_session):
    session, temp_dir = db_session
    now = datetime.utcnow()

    file1 = Path(temp_dir) / "notes.txt"
    file2 = Path(temp_dir) / "notes.docx"
    file1.write_text("Unique text content here that has sufficient length for hashing test.")
    file2.write_text("Different raw bytes on disk")

    doc1 = Document(
        id="doc_txt",
        filename="notes.txt",
        filepath=str(file1),
        file_type="txt",
        file_size=120,
        file_hash="hash_x",
        extracted_text="Unique text content here that has sufficient length for hashing test.",
        modified_at=now - timedelta(days=2),
    )
    doc2 = Document(
        id="doc_docx",
        filename="notes.docx",
        filepath=str(file2),
        file_type="docx",
        file_size=250,
        file_hash="hash_y",
        extracted_text="Unique text content here that has sufficient length for hashing test.",
        modified_at=now,
    )
    session.add_all([doc1, doc2])
    session.commit()

    res = find_duplicates(session)
    assert res.total_groups == 1
    group = res.groups[0]
    assert group.match_type == "exact_text"
    assert group.recommended_document_id == "doc_docx"

def test_filename_variant_and_delete(db_session):
    session, temp_dir = db_session
    now = datetime.utcnow()

    file1 = Path(temp_dir) / "report.pdf"
    file2 = Path(temp_dir) / "report (1).pdf"
    file1.write_bytes(b"Report 1")
    file2.write_bytes(b"Report 2")

    doc1 = Document(
        id="doc_canonical",
        filename="report.pdf",
        filepath=str(file1),
        file_type="pdf",
        file_size=1000,
        file_hash="hash_a",
        extracted_text="Executive summary of the annual report for BitByBit company quarterly review.",
        modified_at=now,
        created_at=now,
    )
    doc2 = Document(
        id="doc_copy",
        filename="report (1).pdf",
        filepath=str(file2),
        file_type="pdf",
        file_size=1000,
        file_hash="hash_b",
        extracted_text="Executive summary of the annual report for BitByBit company quarterly review.",
        modified_at=now - timedelta(hours=2),
        created_at=now - timedelta(hours=2),
    )
    session.add_all([doc1, doc2])
    session.commit()

    res = find_duplicates(session)
    assert res.total_groups == 1
    group = res.groups[0]
    assert group.recommended_document_id == "doc_canonical"

    # Test resolving group
    resolve_res = resolve_duplicate_group(session, keep_document_id="doc_canonical", delete_from_disk=True)
    assert resolve_res.deleted_count == 1
    assert not file2.exists() # physical file deleted
    assert file1.exists() # kept file preserved
    assert session.query(Document).filter(Document.id == "doc_copy").first() is None
    assert session.query(Document).filter(Document.id == "doc_canonical").first() is not None

def test_delete_single_duplicate(db_session):
    session, temp_dir = db_session
    test_file = Path(temp_dir) / "to_delete.txt"
    test_file.write_text("temporary")

    doc = Document(
        id="doc_del",
        filename="to_delete.txt",
        filepath=str(test_file),
        file_type="txt",
        file_size=9,
        file_hash="hash_del",
    )
    session.add(doc)
    session.commit()

    del_res = delete_single_duplicate(session, "doc_del", delete_from_disk=True)
    assert del_res["status"] == "success"
    assert not test_file.exists()
    assert session.query(Document).filter(Document.id == "doc_del").first() is None

def test_duplicates_api_endpoints(db_session):
    session, temp_dir = db_session
    now = datetime.utcnow()

    fileA = Path(temp_dir) / "fileA.txt"
    fileB = Path(temp_dir) / "fileB.txt"
    fileA.write_text("Hello duplicate world")
    fileB.write_text("Hello duplicate world")

    docA = Document(
        id="docA",
        filename="fileA.txt",
        filepath=str(fileA),
        file_type="txt",
        file_size=21,
        file_hash="h123",
        modified_at=now - timedelta(days=1),
    )
    docB = Document(
        id="docB",
        filename="fileB.txt",
        filepath=str(fileB),
        file_type="txt",
        file_size=21,
        file_hash="h123",
        modified_at=now,
    )
    session.add_all([docA, docB])
    session.commit()

    def override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    # 1. GET /api/duplicates
    resp = client.get("/api/duplicates")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_groups"] == 1
    assert data["total_duplicate_files"] == 1
    group = data["groups"][0]
    assert group["recommended_document_id"] == "docB"

    # 2. POST /api/duplicates/resolve-group
    resolve_resp = client.post("/api/duplicates/resolve-group", json={
        "keep_document_id": "docB",
        "delete_from_disk": True,
    })
    assert resolve_resp.status_code == 200
    assert resolve_resp.json()["deleted_count"] == 1
    assert not fileA.exists()
    assert fileB.exists()

    app.dependency_overrides.clear()
