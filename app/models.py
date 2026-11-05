
from datetime import datetime, timezone
from decimal import Decimal
import uuid

from sqlmodel import Field, SQLModel


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
