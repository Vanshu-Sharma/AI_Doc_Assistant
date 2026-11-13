from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


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

