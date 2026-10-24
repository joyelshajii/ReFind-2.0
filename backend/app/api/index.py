from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.base import get_db, is_pgvector_active
from app.core.config import settings
from app.db.models import Document, Chunk
from app.schemas.index import IndexFolderRequest, IndexProgressStatus, IndexResponse, MediaJobDecision
from app.connectors.local_source import LocalFileSource
from app.processors.factory import SUPPORTED_EXTENSIONS
from app.services.indexer import (
    indexing_status, index_folder_background,
    try_reserve_indexing_run, release_indexing_run,
)

router = APIRouter(prefix="/api/index", tags=["Indexing"])


@router.get("/media-jobs")
def media_jobs():
    from app.services.media_jobs import list_media_jobs
    return list_media_jobs()


@router.post("/media-jobs/{job_id}")
def decide_media_job(job_id: str, payload: MediaJobDecision):
    from app.services.media_jobs import decide_media_job as decide
    try:
        return decide(job_id, payload.action, payload.scheduled_at)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=400, detail="The source file is no longer accessible.") from error

@router.post("", response_model=IndexResponse)
def trigger_index(
    payload: Optional[IndexFolderRequest] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Index only a directory explicitly supplied by the user."""
    global indexing_status
    if payload is None or not payload.folder_path.strip():
        raise HTTPException(status_code=400, detail="Choose a folder to index. A folder path is required.")
    target_dir = payload.folder_path.strip()
    target_path = Path(target_dir).expanduser()
    if not target_path.is_absolute():
        raise HTTPException(status_code=400, detail="Enter the full absolute path of the folder you want to index.")
    if indexing_status.is_indexing:
        return IndexResponse(
            status="busy",
            message="Indexing is already in progress.",
            files_found=indexing_status.total_files
        )

    if not target_path.exists():
        raise HTTPException(status_code=400, detail=f"Directory '{target_dir}' does not exist.")
    if not target_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path '{target_dir}' is not a directory.")

    discovered_files = LocalFileSource().scan(str(target_path))
    if not discovered_files:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"No supported files found in '{target_dir}'. Supported formats: {supported}.",
        )

    # Start background processing
    if not try_reserve_indexing_run():
        return IndexResponse(
            status="busy",
            message="Indexing is already in progress.",
            files_found=indexing_status.total_files,
        )
    background_tasks.add_task(index_folder_background, str(target_path.resolve()), True, discovered_files)

    return IndexResponse(
        status="started",
        message=f"Indexing initiated for '{target_dir}'",
        files_found=len(discovered_files)
    )

@router.get("/status", response_model=IndexProgressStatus)
def get_indexing_status(db: Session = Depends(get_db)):
    """
    Returns real-time progress status of the indexing pipeline.
    Always reflects real database numbers when indexing is idle or complete.
    """
    global indexing_status
    if not indexing_status.is_indexing:
        # Populate live counts directly from database
        counts = dict(db.query(Document.file_type, func.count(Document.id)).group_by(Document.file_type).all())
        total_docs = sum(counts.values())
        indexing_status.processed_files = total_docs
        indexing_status.total_files = total_docs

        # Breakdown by file format
        doc_count = sum(counts.get(kind, 0) for kind in ("pdf", "docx", "pptx", "txt"))
        img_count = sum(counts.get(kind, 0) for kind in ("png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic", "heif"))
        indexing_status.documents_count = doc_count
        indexing_status.images_count = img_count
        indexing_status.other_count = max(0, total_docs - doc_count - img_count)

        indexing_status.last_indexed_at = db.query(func.max(Document.indexed_at)).scalar()

    return indexing_status

@router.post("/upload")
def upload_and_index_files(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Save companions first, then inspect media before expensive processing."""
    from app.services.upload_service import save_and_index_uploads
    if not files:
        raise HTTPException(status_code=400, detail="Select at least one file to upload.")
    if not try_reserve_indexing_run():
        raise HTTPException(status_code=409, detail="Indexing is already in progress. Please try again when it finishes.")
    try:
        return save_and_index_uploads(files, db)
    finally:
        release_indexing_run()


@router.get("/stats")
def get_index_stats(db: Session = Depends(get_db)):
    total_docs = db.query(Document).count()
    total_chunks = db.query(Chunk).count()
    last_doc = db.query(Document).order_by(Document.indexed_at.desc()).first()

    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "last_indexed_at": last_doc.indexed_at if last_doc else None,
        "storage_source": "Local Files",
        "database_type": "PostgreSQL (pgvector)" if is_pgvector_active else "Local Vector Store (SQLite + Cosine)",
        "embedding_provider": "embeddinggemma" if settings.LOCAL_FAST_MODE else settings.EMBEDDING_PROVIDER,
        "embedding_dimension": settings.EMBEDDING_DIMENSION
    }
