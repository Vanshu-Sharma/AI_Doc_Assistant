from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
import logging
from pathlib import Path
import time
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
import httpx
from sqlmodel import Session, delete, select

from app.database import engine, get_session, init_db
from app.llm import TokenUsage, generate
from app.models import Document, UsageLog
from app.pricing import calculate_cost
from app.schema import AskRequest

logger = logging.getLogger(__name__)


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


@app.post("/ask")
async def ask(
    request: AskRequest,
    session: Session = Depends(get_session),
):
    """Generate a streamed answer using the specified LLM provider and log token usage."""
    start_time = time.perf_counter()
    usage_info: TokenUsage | None = None

    def on_usage(usage: TokenUsage) -> None:
        nonlocal usage_info
        usage_info = usage

    try:
        stream = generate(
            prompt=request.question,
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

        if status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="LLM provider rate limit exceeded.",
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
        try:
            if first_chunk:
                yield first_chunk
            async for chunk in stream:
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
                save_usage_log(
                    session=session,
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

    return StreamingResponse(
        stream_response_generator(),
        media_type="text/plain; charset=utf-8",
    )
