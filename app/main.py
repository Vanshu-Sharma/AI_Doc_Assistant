from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import logging
import os
from pathlib import Path
import time
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
import httpx
from sqlmodel import Session, delete, select


from app.database import engine, get_session, init_db
from app.ingestion import ingest_document_pdf, search_similar_chunks
from app.llm import ExtractionValidationError, TokenUsage, extract_resume, generate
from app.models import Chunk, Document, DocumentStatus, Feedback, UsageLog
from app.pricing import calculate_cost
from app.schema import (
    AskRequest,
    ChunkResponse,
    ExtractRequest,
    FeedbackRequest,
    FeedbackResponse,
    ResumeData,
    SearchResult,
)
from app.tracing import create_trace_id, record_feedback_score, trace_generation

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Document Assistant", lifespan=lifespan)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_SIZE = MAX_UPLOAD_BYTES
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))


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


def save_usage_log(
    session: Session,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    latency_seconds: float,
    cost_usd: Decimal,
) -> UsageLog:
    """Persist LLM token usage, latency, and cost to PostgreSQL."""
    usage_log = UsageLog(
        id=str(uuid4()),
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_seconds=round(latency_seconds, 4),
        cost_usd=cost_usd,
        created_at=datetime.now(timezone.utc),
    )
    try:
        session.add(usage_log)
        session.commit()
        session.refresh(usage_log)
        logger.info(
            "Persisted usage log %s (provider=%s, model=%s, tokens=%d, cost=$%s)",
            usage_log.id,
            provider,
            model,
            total_tokens,
            cost_usd,
        )
        return usage_log
    except Exception as exc:
        session.rollback()
        logger.error("Failed to commit usage log to database: %s", exc)
        raise


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=Document, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
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

    # Check if a document with the same filename is already uploaded
    existing_by_name = session.exec(
        select(Document).where(
            Document.filename == filename,
            Document.status != DocumentStatus.FAILED,
        )
    ).first()
    if existing_by_name:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document '{filename}' is already uploaded.",
        )

    # Clean up any previously failed record with the same filename
    failed_doc = session.exec(
        select(Document).where(
            Document.filename == filename,
            Document.status == DocumentStatus.FAILED,
        )
    ).first()
    if failed_doc:
        session.delete(failed_doc)
        session.commit()

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    doc_id = str(uuid4())
    temp_file_path = UPLOAD_DIR / f".tmp_{doc_id}_{filename}"
    saved_file_path = UPLOAD_DIR / f"{doc_id}_{filename}"

    total_bytes = 0
    chunk_size = 1024 * 1024  # 1 MB chunks
    hasher = hashlib.sha256()

    try:
        with open(temp_file_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    break
                hasher.update(chunk)
                f.write(chunk)
    except Exception:
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise

    if total_bytes == 0:
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(
            status_code=400,
            detail="File is empty",
        )

    if total_bytes > MAX_UPLOAD_BYTES:
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MB limit",
        )

    # Check if a document with identical content is already uploaded
    new_file_hash = hasher.hexdigest()
    same_size_docs = session.exec(
        select(Document).where(
            Document.size_bytes == total_bytes,
            Document.status != DocumentStatus.FAILED,
        )
    ).all()
    for doc in same_size_docs:
        if doc.file_path and Path(doc.file_path).exists():
            existing_hasher = hashlib.sha256()
            with open(doc.file_path, "rb") as ef:
                while file_chunk := ef.read(chunk_size):
                    existing_hasher.update(file_chunk)
            if existing_hasher.hexdigest() == new_file_hash:
                if temp_file_path.exists():
                    temp_file_path.unlink()
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"A document with identical content is already uploaded as '{doc.filename}'.",
                )

    temp_file_path.rename(saved_file_path)

    initial_status = (
        DocumentStatus.PENDING if extension == ".pdf" else DocumentStatus.READY
    )

    document = Document(
        id=doc_id,
        filename=filename,
        content_type=file.content_type or "application/octet-stream",
        size_bytes=total_bytes,
        uploaded_at=datetime.now(timezone.utc),
        status=initial_status,
        file_path=str(saved_file_path),
    )

    session.add(document)
    session.commit()
    session.refresh(document)

    # Schedule background ingestion for PDFs
    if extension == ".pdf":
        background_tasks.add_task(
            ingest_document_pdf,
            document_id=document.id,
            file_path=str(saved_file_path),
        )

    return document


@app.get("/documents", response_model=list[Document])
def list_documents(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum documents to return"),
    offset: int = Query(default=0, ge=0, description="Number of documents to skip"),
    session: Session = Depends(get_session),
):
    statement = select(Document).offset(offset).limit(limit)
    return session.exec(statement).all()


@app.get("/documents/{document_id}", response_model=Document)
def get_document(
    document_id: str,
    session: Session = Depends(get_session),
):
    doc = session.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found.",
        )
    return doc


@app.get("/documents/{document_id}/chunks", response_model=list[ChunkResponse])
def get_document_chunks(
    document_id: str,
    session: Session = Depends(get_session),
):
    doc = session.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found.",
        )
    statement = (
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
    )
    return list(session.exec(statement).all())


@app.get(
    "/search",
    response_model=list[SearchResult],
    status_code=status.HTTP_200_OK,
    summary="Search Document Chunks",
    description=(
        "Find document text chunks that best match a question using pgvector cosine "
        "distance (<=>). Returns the top-k closest chunks, ordered by relevance."
    ),
    tags=["Search"],
)
def search_documents(
    q: str = Query(..., min_length=1, description="Search query string"),
    k: int = Query(default=5, ge=1, le=50, description="Maximum number of results to return"),
    session: Session = Depends(get_session),
) -> list[SearchResult]:
    """Search ingested chunks using pgvector cosine distance operator <=>."""
    cleaned_q = q.strip()
    if not cleaned_q:
        raise HTTPException(
            status_code=422,
            detail="The 'q' search query cannot be empty or whitespace.",
        )

    try:
        matches = search_similar_chunks(
            query_text=cleaned_q,
            k=k,
            session=session,
        )
    except ValueError as exc:
        logger.error("Search configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Search processing failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service error. Please try again later.",
        ) from exc

    return [
        SearchResult(
            text=chunk.text,
            filename=chunk.filename,
            page=chunk.page_number,
            page_number=chunk.page_number,
            score=score,
        )
        for chunk, score in matches
    ]


@app.post("/ask")
async def ask(
    request: AskRequest,
    session: Session = Depends(get_session),
):
    """Generate a streamed answer based strictly on uploaded documents using pgvector retrieval."""
    start_time = time.perf_counter()
    usage_info: TokenUsage | None = None
    trace_id = request.trace_id or create_trace_id()

    def on_usage(usage: TokenUsage) -> None:
        nonlocal usage_info
        usage_info = usage

    # 1. Check for conversational greetings (handle without document retrieval)
    if is_greeting(request.question):
        prompt = (
            "You are a friendly and helpful AI Document Assistant. "
            f"The user greeted you with: '{request.question.strip()}'. "
            "Provide a warm, natural, and helpful conversational greeting offering assistance with their documents."
        )
    else:
        # 2. Retrieve relevant document chunks using pgvector semantic search
        try:
            matches = search_similar_chunks(
                query_text=request.question,
                k=5,
                session=session,
            )
        except Exception as exc:
            logger.error("Document retrieval failed during /ask: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document retrieval service is temporarily unavailable.",
            ) from exc

        # 3. Build grounded RAG prompt from retrieved chunks
        prompt = build_rag_prompt(
            question=request.question,
            chunks=matches,
        )

    try:
        stream = generate(
            prompt=prompt,
            provider=request.provider,
            on_usage=on_usage,
        )
        # Fetch the first chunk before sending HTTP headers to fail fast on errors
        first_chunk = await anext(stream)
    except StopAsyncIteration:
        first_chunk = ""
    except TimeoutError as exc:
        logger.error("LLM request timed out: %s", exc)
        raise HTTPException(
            status_code=504,
            detail="LLM request timed out.",
        ) from exc
    except (ConnectionError, httpx.TransportError) as exc:
        logger.error("LLM connection error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Failed to connect to LLM provider.",
        ) from exc
    except ValueError as exc:
        logger.error("LLM configuration error: %s", exc)
        raise HTTPException(
            status_code=400 if "Unsupported provider" in str(exc) else 500,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status_code is None and hasattr(exc, "response"):
            status_code = getattr(exc.response, "status_code", None)

        try:
            status_code = int(status_code)
        except (TypeError, ValueError):
            status_code = None

        logger.error("LLM provider error (%s): %s", status_code, exc)

        exc_str = str(exc).lower()
        is_rate_limit = (
            status_code == 429
            or "rate_limit_exceeded" in exc_str
            or "rate limit" in exc_str
            or "tokens per minute" in exc_str
        )

        if is_rate_limit:
            raise HTTPException(
                status_code=429,
                detail="LLM provider rate limit exceeded.",
            ) from exc
        elif status_code == 413:
            raise HTTPException(
                status_code=413,
                detail="LLM request payload exceeds model context or rate limits.",
            ) from exc
        elif status_code and 500 <= status_code <= 599:
            raise HTTPException(
                status_code=502,
                detail="LLM provider service error.",
            ) from exc
        else:
            raise HTTPException(
                status_code=500,
                detail="An error occurred while generating the response.",
            ) from exc

    async def stream_response_generator():
        accumulated_chunks: list[str] = []
        try:
            if first_chunk:
                accumulated_chunks.append(first_chunk)
                yield first_chunk
            async for chunk in stream:
                accumulated_chunks.append(chunk)
                yield chunk
        except Exception as stream_exc:
            logger.error(
                "Error during response streaming: %s",
                stream_exc,
                exc_info=True,
            )
            # Incomplete streaming requests must not be persisted
            return

        # Request completed successfully
        latency_seconds = time.perf_counter() - start_time
        full_output = "".join(accumulated_chunks)
        model = "unknown"
        provider = request.provider
        cost_usd = Decimal("0")

        if usage_info is not None:
            provider = usage_info.provider or request.provider
            model = usage_info.model or "unknown"
            input_tokens = usage_info.input_tokens
            output_tokens = usage_info.output_tokens
            total_tokens = usage_info.total_tokens

            try:
                cost_usd = calculate_cost(
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            except ValueError as cost_err:
                logger.warning(
                    "Cost calculation not found for provider=%s model=%s: %s. Defaulting to 0.",
                    provider,
                    model,
                    cost_err,
                )
                cost_usd = Decimal("0")

            try:
                with Session(engine) as log_session:
                    save_usage_log(
                        session=log_session,
                        provider=provider,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        latency_seconds=latency_seconds,
                        cost_usd=cost_usd,
                    )
            except Exception as db_exc:
                logger.error(
                    "Database persistence failed for usage log: %s",
                    db_exc,
                    exc_info=True,
                )

        # Log generation trace to Langfuse
        try:
            trace_generation(
                trace_id=trace_id,
                name="ask",
                prompt=request.question,
                output=full_output,
                model=model,
                provider=provider,
                usage=usage_info,
                cost_usd=float(cost_usd) if cost_usd else 0.0,
                metadata={
                    "latency_seconds": round(latency_seconds, 4),
                },
            )
        except Exception as trace_exc:
            logger.warning("Failed to trace ask generation: %s", trace_exc)

    return StreamingResponse(
        stream_response_generator(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Trace-Id": trace_id},
    )


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit User Feedback",
    description="Submit thumbs up/down user feedback for an LLM response or trace ID.",
    tags=["Feedback"],
)
def submit_feedback(
    request: FeedbackRequest,
    session: Session = Depends(get_session),
) -> FeedbackResponse:
    """Submit thumbs up/down feedback for a trace and persist to database & Langfuse."""
    feedback_record = Feedback(
        trace_id=request.trace_id,
        feedback=request.feedback or "thumbs_up",
        score=request.score if request.score is not None else 1.0,
        comment=request.comment,
    )
    try:
        session.add(feedback_record)
        session.commit()
        session.refresh(feedback_record)
    except Exception as exc:
        session.rollback()
        logger.error("Failed to persist feedback to database: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save feedback to database.",
        ) from exc

    # Send score to Langfuse (non-blocking failure)
    try:
        record_feedback_score(
            trace_id=feedback_record.trace_id,
            score=feedback_record.score,
            comment=feedback_record.comment,
        )
    except Exception as exc:
        logger.warning("Failed to record score in Langfuse: %s", exc)

    return feedback_record


@app.get(
    "/feedback",
    response_model=list[FeedbackResponse],
    summary="List Feedback",
    description="List submitted feedback records, optionally filtered by trace ID.",
    tags=["Feedback"],
)
def list_feedback(
    trace_id: str | None = Query(default=None, description="Filter by trace ID"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max feedback items to return"),
    offset: int = Query(default=0, ge=0, description="Number of feedback items to skip"),
    session: Session = Depends(get_session),
) -> list[Feedback]:
    """Retrieve feedback records."""
    statement = select(Feedback)
    if trace_id:
        statement = statement.where(Feedback.trace_id == trace_id)
    statement = statement.offset(offset).limit(limit)
    return list(session.exec(statement).all())


@app.post("/extract", response_model=ResumeData, status_code=200)
async def extract(
    request: ExtractRequest,
    session: Session = Depends(get_session),
):
    """Extract structured resume data (name, email, phone, skills) from raw resume text."""
    start_time = time.perf_counter()
    logger.info("Extraction request started. Provider: %s", request.provider)

    usage_records: list[TokenUsage] = []

    def on_usage(usage: TokenUsage) -> None:
        usage_records.append(usage)
        logger.info(
            "Token usage captured (provider=%s, model=%s, input=%d, output=%d, total=%d)",
            usage.provider,
            usage.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
        )

    def _persist_usage_records(latency_seconds: float) -> None:
        for usage in usage_records:
            provider_name = usage.provider or request.provider
            model_name = usage.model or "unknown"
            try:
                cost_usd = calculate_cost(
                    provider=provider_name,
                    model=model_name,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
            except ValueError as cost_err:
                logger.warning(
                    "Cost calculation failed for provider=%s model=%s: %s. Defaulting to 0.",
                    provider_name,
                    model_name,
                    cost_err,
                )
                cost_usd = Decimal("0")
            try:
                save_usage_log(
                    session=session,
                    provider=provider_name,
                    model=model_name,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                    latency_seconds=latency_seconds,
                    cost_usd=cost_usd,
                )
                logger.info(
                    "Usage record persisted (provider=%s, model=%s).",
                    provider_name,
                    model_name,
                )
            except Exception as db_exc:
                logger.error(
                    "Database persistence failed for extraction usage log: %s",
                    db_exc,
                    exc_info=True,
                )

    try:
        result = await extract_resume(
            resume_text=request.resume_text,
            provider=request.provider,
            on_usage=on_usage,
        )
        latency_seconds = time.perf_counter() - start_time
        logger.info("Extraction completed successfully.")
        _persist_usage_records(latency_seconds)
        return result

    except ExtractionValidationError as exc:
        latency_seconds = time.perf_counter() - start_time
        _persist_usage_records(latency_seconds)
        logger.error("Resume extraction validation error: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except TimeoutError as exc:
        logger.error("LLM request timed out during extraction: %s", exc)
        raise HTTPException(
            status_code=504,
            detail="LLM request timed out.",
        ) from exc
    except (ConnectionError, httpx.TransportError) as exc:
        logger.error("LLM connection error during extraction: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Failed to connect to LLM provider.",
        ) from exc
    except ValueError as exc:
        logger.error("LLM configuration error during extraction: %s", exc)
        raise HTTPException(
            status_code=400 if "Unsupported provider" in str(exc) else 500,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status_code is None and hasattr(exc, "response"):
            status_code = getattr(exc.response, "status_code", None)

        try:
            status_code = int(status_code)
        except (TypeError, ValueError):
            status_code = None

        logger.error("LLM provider error during extraction (%s): %s", status_code, exc)

        exc_str = str(exc).lower()
        is_rate_limit = (
            status_code == 429
            or "rate_limit_exceeded" in exc_str
            or "rate limit" in exc_str
            or "tokens per minute" in exc_str
        )

        if is_rate_limit:
            raise HTTPException(
                status_code=429,
                detail="LLM provider rate limit exceeded.",
            ) from exc
        elif status_code == 413:
            raise HTTPException(
                status_code=413,
                detail="LLM request payload exceeds model context or rate limits.",
            ) from exc
        elif status_code and 500 <= status_code <= 599:
            raise HTTPException(
                status_code=502,
                detail="LLM provider service error.",
            ) from exc
        else:
            raise HTTPException(
                status_code=500,
                detail="An error occurred while extracting resume data.",
            ) from exc
