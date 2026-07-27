"""
POST /ingest  — upload a PDF and trigger the ingestion graph.
GET  /ingest  — list all ingested documents.
DELETE /ingest/{document_id} — remove a document.
"""

import os
import shutil
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from api.deps import get_db
from api.schemas.ingest_schemas import IngestResponse, DocumentOut
from db.postgres import Document
from graphs.ingestion.graph import ingestion_graph
from config import UPLOADS_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE_MB = 50


@router.post("", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a PDF. Saves it to disk, creates a Document row, then runs
    the ingestion graph (validate → parse → chunk → enrich → index).

    Returns immediately with document_id and status. The heavy work
    (Docling parsing, enrichment LLM calls) runs synchronously — for a
    production system you'd push this to a background task queue.
    """
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only PDF is supported."
        )

    # Read and size-check
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Max {MAX_FILE_SIZE_MB}MB."
        )

    # Save to uploads dir
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / file.filename
    with open(dest, "wb") as f:
        f.write(contents)

    # Create Document row
    doc = Document(filename=file.filename, status="processing")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    document_id = doc.id

    # Run ingestion graph
    try:
        result = ingestion_graph.invoke({
            "file_path": str(dest),
            "document_id": document_id,
        })

        return IngestResponse(
            document_id=document_id,
            filename=file.filename,
            status="ready",
            chunks_indexed=result.get("indexed"),
            chunks_dropped=result.get("chunks_dropped_as_junk"),
            message=f"Successfully indexed {result.get('indexed', 0)} chunks.",
        )

    except Exception as e:
        logger.error(f"Ingestion failed for document {document_id}: {e}")
        db.query(Document).filter_by(id=document_id).update({"status": "failed"})
        db.commit()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    """List all ingested documents."""
    docs = db.query(Document).order_by(Document.id.desc()).all()
    return [
        DocumentOut(
            id=d.id,
            filename=d.filename,
            status=d.status,
            created_at=d.created_at.isoformat(),
        )
        for d in docs
    ]


@router.delete("/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Remove a document from Postgres and Qdrant."""
    from db.qdrant import client
    from config import QDRANT_COLLECTION
    from db.postgres import Chunk

    doc = db.query(Document).filter_by(id=document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove from Qdrant
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )]
            )
        )
    except Exception as e:
        logger.warning(f"Qdrant delete failed for doc {document_id}: {e}")

    # Remove chunks and document from Postgres
    db.query(Chunk).filter_by(document_id=document_id).delete()
    db.delete(doc)
    db.commit()

    return {"message": f"Document {document_id} deleted.", "document_id": document_id}
