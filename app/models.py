from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
import sqlalchemy as sa
from sqlmodel import Field, SQLModel


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(SQLModel, table=True):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    description: str | None = Field(default=None)
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING,
        sa_type=sa.Enum(
            DocumentStatus,
            values_callable=lambda obj: [e.value for e in obj],
            native_enum=False,
        ),
        index=True,
    )
    error_message: str | None = Field(default=None)
    file_path: str | None = Field(default=None)


class Chunk(SQLModel, table=True):
    __tablename__ = "chunk"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
        index=True,
    )
    document_id: str = Field(
        foreign_key="document.id",
        index=True,
        ondelete="CASCADE",
    )
    filename: str = Field(
        index=True,
    )
    page_number: int = Field(
        description="1-based page number from the PDF document",
    )
    chunk_index: int = Field(
        default=0,
        description="0-based sequence index of the chunk",
    )
    text: str = Field(
        description="Clean text content extracted for this chunk",
    )
    embedding: list[float] = Field(
        sa_column=Column(Vector(768), nullable=False),
        description="768-dimensional dense vector embedding",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class UsageLog(SQLModel, table=True):
    """Store LLM token usage, latency, and cost."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )

    provider: str
    model: str

    input_tokens: int
    output_tokens: int
    total_tokens: int

    latency_seconds: float

    cost_usd: Decimal = Field(
        max_digits=20,
        decimal_places=12,
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


# Backward compatibility alias
Document_Data = Document
