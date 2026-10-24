import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.base import get_db, is_postgres_active, is_pgvector_active
from app.db.models import Document, Chunk
from app.schemas.index import IndexFolderRequest, IndexProgressStatus, IndexResponse
from app.connectors.local_source import LocalFileSource
from app.processors.factory import SUPPORTED_EXTENSIONS
from app.services.indexer import (
    indexing_status, index_folder_background, index_single_file, file_index_lock,
    try_reserve_indexing_run,
)
from app.core.config import settings

router = APIRouter(prefix="/api/index", tags=["Indexing"])
logger = logging.getLogger("refind.api.index")

@router.post("", response_model=IndexResponse)
def trigger_index(
    payload: Optional[IndexFolderRequest] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Triggers indexing of a local directory. If no folder is provided,
    indexes the default uploads storage folder (or demo_data fallback).
    """
    global indexing_status
    if indexing_status.is_indexing:
        return IndexResponse(
            status="busy",
            message="Indexing is already in progress.",
            files_found=indexing_status.total_files
        )

    # Determine default target directory: prefer storage/uploads if populated, else demo_data
    if payload and payload.folder_path and payload.folder_path.strip():
        target_dir = payload.folder_path.strip()
    else:
        upload_path = Path(settings.UPLOAD_DIR)
        demo_path = Path(settings.DEMO_DATA_DIR)
        # Check if uploads directory has files
        if upload_path.exists() and any(upload_path.iterdir()):
            target_dir = str(upload_path)
        else:
            target_dir = str(demo_path)

    target_path = Path(target_dir).expanduser()
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
    """
    Accepts direct drag-and-drop file uploads and indexes them immediately.
    """
    global indexing_status
    if not files:
        raise HTTPException(status_code=400, detail="Select at least one file to upload.")
    indexed = []
    failed = []
    upload_dir = Path(settings.UPLOAD_DIR)
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        logger.exception("Unable to prepare upload directory %s", upload_dir)
        raise HTTPException(status_code=503, detail="Upload storage is unavailable. Please try again later.") from error

    for file in files:
        original_filename = file.filename or ""
        safe_filename = Path(original_filename.replace("\\", "/")).name
        staged_path = None
        try:
            if not safe_filename or safe_filename in {".", ".."}:
                raise ValueError("The uploaded file must have a valid filename.")
            file_type = Path(safe_filename).suffix.lower()
            if file_type not in SUPPORTED_EXTENSIONS:
                supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
                raise ValueError(f"Unsupported file type. Supported extensions: {supported}.")

            dest_path = upload_dir / safe_filename
            with file_index_lock(str(dest_path.resolve())):
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=upload_dir, prefix=".refind-upload-",
                    suffix=".upload", delete=False
                ) as buffer:
                    staged_path = Path(buffer.name)
                    shutil.copyfileobj(file.file, buffer)
                os.replace(staged_path, dest_path)
                staged_path = None

                doc = index_single_file(str(dest_path.resolve()), db)
            indexed.append({
                "id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size": doc.file_size
            })
        except Exception as upload_err:
            logger.exception("Failed to index uploaded file %s", safe_filename or "(unnamed)")
            try:
                db.rollback()
            except Exception:
                logger.exception("Failed to roll back database session after upload failure")
            if isinstance(upload_err, (ValueError, RuntimeError)):
                error_message = str(upload_err)
            else:
                error_message = "Unable to save or index this file. Check the backend logs for details."
            failed.append({
                "filename": safe_filename or "(unnamed file)",
                "error": error_message,
            })
        finally:
            if staged_path is not None:
                try:
                    staged_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    logger.warning("Unable to remove staged upload %s: %s", staged_path, cleanup_error)
            try:
                file.file.close()
            except OSError as close_error:
                logger.warning("Unable to close uploaded file %s: %s", safe_filename or "(unnamed)", close_error)

    if not indexing_status.is_indexing:
        # A folder scan owns these progress counters while it is active.
        total_docs = db.query(Document).count()
        indexing_status.processed_files = total_docs
        indexing_status.total_files = total_docs
        docs = db.query(Document.file_type).all()
        doc_count = sum(1 for d in docs if d[0] in ("pdf", "docx", "pptx", "txt"))
        img_count = sum(1 for d in docs if d[0] in ("png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic", "heif"))
        indexing_status.documents_count = doc_count
        indexing_status.images_count = img_count
        indexing_status.other_count = max(0, total_docs - doc_count - img_count)
        indexing_status.error_message = (
            f"{len(failed)} of {len(files)} uploaded file(s) could not be indexed."
            if failed else None
        )
        indexing_status.current_stage = "Error" if failed and not indexed else "Complete"

    if failed and indexed:
        result_status = "partial"
        message = f"Indexed {len(indexed)} file(s); {len(failed)} file(s) failed."
    elif failed:
        result_status = "failed"
        message = f"Could not index any of the {len(files)} uploaded file(s)."
    else:
        result_status = "success"
        message = f"Successfully indexed {len(indexed)} file(s)."

    return {
        "status": result_status,
        "message": message,
        "files": indexed,
        "failed_files": failed,
    }

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
