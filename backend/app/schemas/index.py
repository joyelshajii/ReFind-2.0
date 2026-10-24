from datetime import datetime
from typing import Optional, List, Literal
from pydantic import BaseModel, ConfigDict, Field

class IndexFolderRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    folder_path: str = Field(min_length=1)


class MediaJobDecision(BaseModel):
    action: Literal["continue", "schedule", "skip"]
    scheduled_at: Optional[datetime] = None

class IndexProgressStatus(BaseModel):
    is_indexing: bool = False
    current_stage: str = "idle" # Reading -> Extracting -> Understanding -> Indexing -> Complete
    total_files: int = 0
    processed_files: int = 0
    documents_count: int = 0
    images_count: int = 0
    other_count: int = 0
    current_file: Optional[str] = None
    last_indexed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class IndexResponse(BaseModel):
    status: str
    message: str
    files_found: int
