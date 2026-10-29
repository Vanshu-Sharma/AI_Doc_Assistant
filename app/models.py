from datetime import datetime, timezone
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


# Backward compatibility alias
Document_Data = Document
