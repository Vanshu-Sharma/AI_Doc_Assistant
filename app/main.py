from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlmodel import Session, delete, select

from app.database import engine, get_session, init_db
from app.models import Document, Document_Data


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Document Assistant", lifespan=lifespan)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_SIZE = MAX_UPLOAD_BYTES


class DocumentCollection:
    """Helper collection providing backward-compatibility for tests and inspections."""

    def clear(self) -> None:
        with Session(engine) as session:
            session.exec(delete(Document))
            session.commit()

    def __iter__(self):
        with Session(engine) as session:
            return iter(session.exec(select(Document)).all())

    def __len__(self) -> int:
        with Session(engine) as session:
            return len(session.exec(select(Document)).all())

    def append(self, document: Document) -> None:
        with Session(engine) as session:
            session.add(document)
            session.commit()


documents = DocumentCollection()


def get_document_store():
    """Dependency providing access to the document storage."""
    return documents


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=Document, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
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

    document = Document(
        id=str(uuid4()),
        filename=filename,
        content_type=file.content_type or "application/octet-stream",
        size_bytes=len(content),
        uploaded_at=datetime.now(timezone.utc),
    )

    session.add(document)
    session.commit()
    session.refresh(document)
    return document


@app.get("/documents", response_model=list[Document])
def list_documents(session: Session = Depends(get_session)):
    statement = select(Document)
    return session.exec(statement).all()
