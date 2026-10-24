from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class DuplicateFileItem(BaseModel):
    id: str
    filename: str
    filepath: str
    file_type: str
    file_size: int = 0
    modified_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    is_recommended: bool = False
    recommendation_reason: str = ""
    exists_on_disk: bool = True

class DuplicateGroup(BaseModel):
    group_id: str
    match_type: str # exact_hash, exact_text, filename_and_content, similar_content
    match_label: str
    confidence: float = 1.0
    why_duplicate: str
    differences: str
    potential_savings_bytes: int = 0
    recommended_document_id: str
    files: List[DuplicateFileItem]

class DuplicateOverviewResponse(BaseModel):
    total_groups: int = 0
    total_duplicate_files: int = 0
    total_reclaimable_bytes: int = 0
    groups: List[DuplicateGroup] = []

class ResolveGroupRequest(BaseModel):
    keep_document_id: str
    delete_document_ids: Optional[List[str]] = None
    delete_from_disk: bool = True

class ResolveGroupResponse(BaseModel):
    status: str
    message: str
    kept_document_id: str
    deleted_count: int
    freed_bytes: int

class ResolveAllResponse(BaseModel):
    status: str
    message: str
    groups_resolved: int
    total_deleted: int
    freed_bytes: int
