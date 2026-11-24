from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator


class AskRequest(BaseModel):
    """Request schema for the LLM endpoint."""

    question: str = Field(
        min_length=1,
        max_length=10000,
        description="Question to send to the LLM.",
    )

    provider: Literal["gemini", "groq"] = Field(
        default="gemini",
        description="LLM provider to use.",
    )

    trace_id: str | None = Field(
        default=None,
        description="Optional Langfuse trace ID.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty or blank.")
        return v

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip().lower()
        return v


class UsageResponse(BaseModel):
    """Schema for LLM token usage and cost information."""

    model_config = ConfigDict(from_attributes=True)

    provider: str
    model: str

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    latency_seconds: float = Field(ge=0)
    cost_usd: Decimal = Field(ge=0)

    created_at: datetime


class ChunkResponse(BaseModel):
    """Schema for retrieved document chunk."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    created_at: datetime


class SearchResult(BaseModel):
    """Schema for document search result item."""

    model_config = ConfigDict(from_attributes=True)

    text: str = Field(description="Clean text content of the matching chunk")
    filename: str = Field(description="Source document filename")
    page: int = Field(description="1-based page number where chunk occurs")
    page_number: int | None = Field(
        default=None,
        description="Alias for page number",
    )
    score: float = Field(description="Cosine similarity score (higher is more relevant)")

    @model_validator(mode="after")
    def sync_page_number(self) -> "SearchResult":
        if self.page_number is None:
            self.page_number = self.page
        return self


class ResumeData(BaseModel):
    """Structured data extracted from a resume."""

    name: str | None = None
    email: EmailStr | None = None
    phone: str | None = None
    skills: list[str] = Field(default_factory=list)

    @field_validator("name", "phone", mode="before")
    @classmethod
    def clean_empty_strings(cls, v: str | None) -> str | None:
        if isinstance(v, str):
            v_stripped = v.strip()
            return v_stripped if v_stripped else None
        return v

    @field_validator("email", mode="before")
    @classmethod
    def clean_empty_email(cls, v: str | None) -> str | None:
        if isinstance(v, str):
            v_stripped = v.strip()
            return v_stripped if v_stripped else None
        return v

    @field_validator("skills", mode="before")
    @classmethod
    def clean_skills_list(cls, v: list | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                if item is not None:
                    item_str = str(item).strip()
                    if item_str:
                        cleaned.append(item_str)
            return cleaned
        return v


class ExtractRequest(BaseModel):
    """Request schema for resume information extraction."""

    resume_text: str = Field(
        min_length=1,
        description="Raw resume text to extract information from.",
    )

    provider: Literal["gemini", "groq"] = Field(
        default="gemini",
        description="LLM provider to use for extraction.",
    )

    @field_validator("resume_text")
    @classmethod
    def validate_resume_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Resume text cannot be empty or blank.")
        return v

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip().lower()
        return v


class FeedbackRequest(BaseModel):
    """Request schema for thumbs up/down user feedback."""

    trace_id: str = Field(
        min_length=1,
        description="Trace ID associated with the response being evaluated.",
    )
    feedback: str | None = Field(
        default=None,
        description="Feedback rating: 'thumbs_up' or 'thumbs_down' (also accepts 'up'/'down').",
    )
    value: int | float | str | None = Field(
        default=None,
        description="Numeric or string score (1 for thumbs up, 0 or -1 for thumbs down).",
    )
    score: int | float | str | None = Field(
        default=None,
        description="Alias for value.",
    )
    rating: str | None = Field(
        default=None,
        description="Alias for feedback.",
    )
    comment: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional comment or explanation for the rating.",
    )

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("trace_id cannot be empty or blank.")
        return v.strip()

    @model_validator(mode="after")
    def validate_and_normalize_feedback(self) -> "FeedbackRequest":
        raw_val = self.feedback if self.feedback is not None else (
            self.rating if self.rating is not None else (
                self.score if self.score is not None else self.value
            )
        )
        if raw_val is None:
            raise ValueError(
                "Feedback rating is required. Provide 'thumbs_up' or 'thumbs_down' (or value 1/0)."
            )

        if isinstance(raw_val, bool):
            normalized_score = 1.0 if raw_val else 0.0
            normalized_feedback = "thumbs_up" if raw_val else "thumbs_down"
        elif isinstance(raw_val, (int, float)):
            if raw_val > 0:
                normalized_score = 1.0
                normalized_feedback = "thumbs_up"
            else:
                normalized_score = 0.0
                normalized_feedback = "thumbs_down"
        elif isinstance(raw_val, str):
            clean = raw_val.strip().lower()
            if clean in {"thumbs_up", "thumbsup", "up", "+1", "1", "positive", "good", "true"}:
                normalized_score = 1.0
                normalized_feedback = "thumbs_up"
            elif clean in {"thumbs_down", "thumbsdown", "down", "-1", "0", "negative", "bad", "false"}:
                normalized_score = 0.0
                normalized_feedback = "thumbs_down"
            else:
                raise ValueError(
                    f"Invalid feedback rating: '{raw_val}'. Must be 'thumbs_up' or 'thumbs_down'."
                )
        else:
            raise ValueError(f"Unsupported feedback type: {type(raw_val)}")

        self.feedback = normalized_feedback
        self.score = normalized_score
        self.value = int(normalized_score)
        return self


class FeedbackResponse(BaseModel):
    """Response schema for submitted user feedback."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    trace_id: str
    feedback: str
    score: float
    comment: str | None = None
    created_at: datetime


