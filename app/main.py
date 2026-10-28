
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.models import Document_Data

app = FastAPI(title="AI Document Assistant")

# Temporary in-memory storage
documents: list[Document_Data] = []

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_SIZE = 10 * 1024 * 1024  # 10 MB


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", status_code=201)
async def upload_document(file: UploadFile = File(...)):
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

    content = await file.read(MAX_SIZE + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="File is empty",
        )

    if len(content) > MAX_SIZE:
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

    documents.append(document)
    return document


@app.get("/documents", response_model=list[Document_Data])
def list_documents():
    return documents
