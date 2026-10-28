
from datetime import datetime

from pydantic import BaseModel


class Document_Data(BaseModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime
