from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.schemas.duplicate import (
    DuplicateOverviewResponse,
    ResolveGroupRequest,
    ResolveGroupResponse,
    ResolveAllResponse,
)
from app.services.duplicate_service import (
    find_duplicates,
    delete_single_duplicate,
    resolve_duplicate_group,
    resolve_all_duplicates,
)

router = APIRouter(prefix="/api/duplicates", tags=["Duplicates"])

@router.get("", response_model=DuplicateOverviewResponse)
def get_duplicates_overview(
    min_similarity: float = Query(0.85, ge=0.5, le=1.0, description="Minimum content similarity threshold"),
    db: Session = Depends(get_db)
):
    """
    Scans all indexed files to identify duplicate sets using:
    - Exact binary SHA-256 hash
    - Exact extracted text hash
    - Filename similarity + content match
    - Near-duplicate text similarity

    Recommends the best one to keep (the latest updated file),
    explains why they are duplicates, and details what is different.
    """
    try:
        return find_duplicates(db, min_content_similarity=min_similarity)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to scan for duplicate files: {e}",
        )

@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_duplicate(
    document_id: str,
    delete_from_disk: bool = Query(True, description="Whether to also delete the physical file from disk"),
    db: Session = Depends(get_db)
):
    """
    Deletes an individual duplicate file.
    By default deletes the physical file from disk AND cleans the index.
    """
    try:
        return delete_single_duplicate(db, document_id, delete_from_disk=delete_from_disk)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete file: {e}")

@router.post("/resolve-group", response_model=ResolveGroupResponse)
def resolve_group(
    request: ResolveGroupRequest,
    db: Session = Depends(get_db)
):
    """
    Keeps the designated best file in a duplicate group and deletes all other duplicate copies.
    """
    try:
        return resolve_duplicate_group(
            db=db,
            keep_document_id=request.keep_document_id,
            delete_document_ids=request.delete_document_ids,
            delete_from_disk=request.delete_from_disk,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to resolve duplicate group: {e}")

@router.post("/resolve-all", response_model=ResolveAllResponse)
def resolve_all(
    delete_from_disk: bool = Query(True, description="Whether to delete physical files from disk"),
    db: Session = Depends(get_db)
):
    """
    1-Click batch cleanup: keeps the recommended (latest updated) file in every duplicate
    group and deletes all redundant copies from disk and digital memory.
    """
    try:
        return resolve_all_duplicates(db, delete_from_disk=delete_from_disk)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to resolve duplicates: {e}")
