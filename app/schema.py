from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
