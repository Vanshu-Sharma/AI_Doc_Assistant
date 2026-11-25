"""Langfuse tracing and observability helpers."""

import logging
import os
from typing import Any
import uuid

from langfuse import Langfuse

logger = logging.getLogger(__name__)

_client: Langfuse | None = None


def get_langfuse_client() -> Langfuse:
    """Return a singleton Langfuse client instance."""
    global _client
    if _client is None:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if public_key and secret_key:
            _client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
        else:
            _client = Langfuse()
    return _client


def create_trace_id() -> str:
    """Generate a unique trace ID."""
    try:
        client = get_langfuse_client()
        return client.create_trace_id()
    except Exception:
        return uuid.uuid4().hex


def trace_generation(
    trace_id: str,
    name: str,
    prompt: Any,
    output: Any,
    model: str | None = None,
    provider: str | None = None,
    usage: Any | None = None,
    cost_usd: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log an LLM generation trace to Langfuse."""
    try:
        client = get_langfuse_client()
        usage_details = None
        if usage is not None:
            usage_details = {
                "input": getattr(usage, "input_tokens", 0) or 0,
                "output": getattr(usage, "output_tokens", 0) or 0,
                "total": getattr(usage, "total_tokens", 0) or 0,
            }

        meta = {"provider": provider} if provider else {}
        if metadata:
            meta.update(metadata)

        client.start_observation(
            name=name,
            as_type="generation",
            input=prompt,
            output=output,
            model=model,
            usage_details=usage_details,
            cost_details={"total": cost_usd} if cost_usd is not None else None,
            metadata=meta,
            trace_context={"trace_id": trace_id} if trace_id else None,
        )
        client.flush()
    except Exception as exc:
        logger.warning("Failed to record Langfuse generation trace: %s", exc)


def record_feedback_score(
    trace_id: str,
    score: float,
    comment: str | None = None,
    name: str = "user_feedback",
) -> None:
    """Send user feedback score to Langfuse."""
    try:
        client = get_langfuse_client()
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=score,
            comment=comment,
        )
        client.flush()
    except Exception as exc:
        logger.warning("Failed to record Langfuse feedback score: %s", exc)
