from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from app.models import Document_Data

app = FastAPI(title="AI Document Assistant")

# Temporary in-memory storage
documents: list[Document_Data] = []


def get_document_store() -> list[Document_Data]:
    """Dependency providing access to the document storage."""
    return documents


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_SIZE = MAX_UPLOAD_BYTES


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=Document_Data, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    doc_store: list[Document_Data] = Depends(get_document_store),
):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or not extension:
        raise HTTPException(
            status_code=400,
            detail="Filename or extension is missing",
        )

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Allowed types: PDF, TXT, DOCX",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="File is empty",
        )

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MB limit",
        )

    document = Document_Data(
        id=str(uuid4()),
        filename=filename,
        content_type=file.content_type or "application/octet-stream",
        size_bytes=len(content),
        uploaded_at=datetime.now(timezone.utc),
    )

    doc_store.append(document)
    return document


@app.get("/documents", response_model=list[Document_Data])
def list_documents(doc_store: list[Document_Data] = Depends(get_document_store)):
    return doc_store
