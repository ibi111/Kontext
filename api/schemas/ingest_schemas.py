from pydantic import BaseModel
from typing import Optional


class IngestResponse(BaseModel):
    document_id: int
    filename: str
    status: str                      # processing | ready | failed
    chunks_indexed: Optional[int] = None
    chunks_dropped: Optional[int] = None
    message: str


class DocumentOut(BaseModel):
    id: int
    filename: str
    status: str
    created_at: str
