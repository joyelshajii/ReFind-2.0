from pathlib import Path
import os
import subprocess
import sys
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session, load_only, selectinload
from sqlalchemy import or_, func
from app.db.base import get_db
from app.db.models import Document, Chunk, Entity
from app.schemas.document import DocumentSchema, DocumentDetailSchema
from app.schemas.search import ContextDetailResponse
from app.core.config import settings
from app.services.query_parser import parse_query
from app.services.context_explainer import generate_context_explanation
from app.services.index_removal import clear_index, remove_document

router = APIRouter(prefix="/api/documents", tags=["Documents"])

@router.get("", response_model=List[DocumentSchema])
def list_documents(
    search: Optional[str] = None,
    file_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    List all currently indexed files in the digital memory with search, filter, and pagination.
    """
    query = db.query(Document)
    if search:
        query = query.filter(Document.filename.ilike(f"%{search}%"))
    if file_type:
        requested_type = file_type.lower().lstrip(".")
        if requested_type == "audio":
            query = query.filter(Document.file_type.in_({"mp3", "wav", "m4a", "flac", "aac", "ogg", "wma", "opus"}))
        elif requested_type == "video":
            query = query.filter(Document.file_type.in_({"mp4", "mkv", "avi", "mov", "wmv", "webm", "flv", "m4v"}))
        elif requested_type == "image":
            query = query.filter(Document.file_type.in_({"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic", "heif"}))
        else:
            query = query.filter(or_(
                Document.file_type == requested_type,
                func.lower(Document.filepath).endswith(f".{requested_type}", autoescape=True),
            ))
    # Do not expose stale index rows for files that were moved or deleted.
    # Search already applies the same guard; the file browser must stay
    # consistent with search and download behavior.
    ids = []
    visible = 0
    for row in query.with_entities(Document.id, Document.filepath).order_by(Document.indexed_at.desc()).all():
        try:
            available = Path(row.filepath).is_file()
        except OSError:
            available = False
        if not available:
            continue
        if visible >= offset:
            ids.append(row.id)
        visible += 1
        if len(ids) >= limit:
            break
    if not ids:
        return []
    # The browser needs metadata and counts, never full text or vector blobs.
    rows = db.query(Document).filter(Document.id.in_(ids)).options(
        load_only(Document.id, Document.filename, Document.filepath, Document.file_type,
                  Document.file_size, Document.created_at, Document.modified_at,
                  Document.source, Document.summary, Document.indexed_at),
        selectinload(Document.entities),
    ).all()
    by_id = {row.id: row for row in rows}
    docs = [by_id[doc_id] for doc_id in ids if doc_id in by_id]
    chunk_counts = dict(db.query(Chunk.document_id, func.count(Chunk.id))
                        .filter(Chunk.document_id.in_(ids)).group_by(Chunk.document_id).all())

    results = []
    for doc in docs:
        doc_dict = {
            "id": doc.id,
            "filename": doc.filename,
            "filepath": doc.filepath,
            "file_type": doc.file_type,
            "file_size": doc.file_size or 0,
            "created_at": doc.created_at,
            "modified_at": doc.modified_at,
            "source": doc.source,
            "summary": doc.summary or "",
            "indexed_at": doc.indexed_at,
            "entities": doc.entities,
            "chunk_count": chunk_counts.get(doc.id, 0)
        }
        results.append(DocumentSchema(**doc_dict))

    return results

@router.delete("", status_code=status.HTTP_200_OK)
@router.delete("/", status_code=status.HTTP_200_OK)
def delete_all_documents(db: Session = Depends(get_db)):
    """Clear indexed data, queued jobs, and ReFind's saved upload copies."""
    try:
        removed_count, copies_deleted = clear_index(db)
    except RuntimeError as error:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(error)) from error
    except OSError as error:
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not delete all saved upload copies. Check file permissions and try again.") from error
    return {
        "status": "success",
        "removed_count": removed_count,
        "copies_deleted": copies_deleted,
        "message": (
            f"Removed {removed_count} indexed file"
            f"{'' if removed_count == 1 else 's'} from memory. "
            f"Deleted {copies_deleted} saved upload copies and cleared queued jobs. "
            "Source files outside ReFind upload storage were not deleted."
        ),
    }

@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Remove a document, its queued job, and its saved upload copy if present."""
    try:
        filename, copies_deleted = remove_document(db, document_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(error)) from error
    except OSError as error:
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not delete the saved upload copy. Check file permissions and try again.") from error
    return {
        "status": "success",
        "copies_deleted": copies_deleted,
        "message": f"Removed '{filename}' and {copies_deleted} saved upload copies. Source files outside ReFind upload storage were preserved."
    }

@router.get("/{document_id}", response_model=DocumentDetailSchema)
def get_document_detail(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    return DocumentDetailSchema(
        id=doc.id,
        filename=doc.filename,
        filepath=doc.filepath,
        file_type=doc.file_type,
        file_size=doc.file_size or 0,
        created_at=doc.created_at,
        modified_at=doc.modified_at,
        source=doc.source,
        extracted_text=doc.extracted_text or "",
        summary=doc.summary or "",
        indexed_at=doc.indexed_at,
        entities=doc.entities,
        chunks=doc.chunks,
        chunk_count=len(doc.chunks)
    )

@router.get("/{document_id}/context", response_model=ContextDetailResponse)
def get_document_context(
    document_id: str,
    q: str = Query("", description="Original search query to ground context against"),
    score: float = Query(0.94, description="Calculated match score"),
    score_label: str = Query("Strong match", description="Calculated match label"),
    db: Session = Depends(get_db)
):
    """
    Signature 'Show Context' feature.
    Returns side-by-side verified evidence comparing remembered clues to actual document content.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    clues = parse_query(q) if q else parse_query(doc.filename)
    return generate_context_explanation(doc, clues, score, score_label)

@router.get("/{document_id}/download")
def download_document(document_id: str, db: Session = Depends(get_db)):
    """
    Allows user to open/download the original indexed file.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_path = Path(doc.filepath)
    try:
        available = file_path.is_file()
    except OSError:
        available = False
    if not available:
        raise HTTPException(status_code=404, detail=f"File not found on disk: {doc.filepath}")

    return FileResponse(
        path=str(file_path),
        filename=doc.filename,
        media_type="application/octet-stream"
    )


@router.post("/{document_id}/open")
def open_document(document_id: str, request: Request, db: Session = Depends(get_db)):
    """Open an indexed original in the local machine's default application."""
    origin = request.headers.get("origin")
    if origin and origin not in settings.CORS_ORIGINS:
        raise HTTPException(status_code=403, detail="This page cannot open local files.")
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_path = Path(doc.filepath)
    try:
        available = file_path.is_file()
    except OSError:
        available = False
    if not available:
        raise HTTPException(status_code=404, detail="Original file is no longer available.")

    try:
        if sys.platform == "win32":
            os.startfile(str(file_path))
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(file_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not open the original file: {exc}") from exc

    return {"status": "opened"}
