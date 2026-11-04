import asyncio
import inspect
import logging
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from google import genai
from groq import AsyncGroq

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage metadata for an LLM response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str = ""
    provider: str = ""


MAX_RETRIES = 3
REQUEST_TIMEOUT = 30


def is_retryable_error(exc: Exception) -> bool:
    """Check whether an exception should trigger a retry."""
    if isinstance(
        exc,
        (TimeoutError, ConnectionError, httpx.TransportError),
    ):
        return True

    status_code = getattr(exc, "status_code", None)

    if status_code is None:
        status_code = getattr(exc, "code", None)

    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)

    try:
        status_code = int(status_code)
    except (TypeError, ValueError):
        return False

    return status_code == 429 or 500 <= status_code <= 599


async def _generate_once(
    prompt: str,
    provider: str,
    on_usage: Callable[[TokenUsage], None] | None = None,
) -> AsyncIterator[str]:
    """Generate a streamed response and capture token usage from the provider."""
    provider_norm = provider.lower().strip()
    usage: TokenUsage | None = None
    model: str | None = None

    if provider_norm == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        if not model:
            raise ValueError("GEMINI_MODEL is not configured.")

        client = genai.Client(api_key=api_key)

        stream = await client.aio.models.generate_content_stream(
            model=model,
            contents=prompt,
        )

        async for chunk in stream:
            metadata = getattr(chunk, "usage_metadata", None)

            if metadata is not None:
                input_tokens = getattr(metadata, "prompt_token_count", 0) or 0
                output_tokens = getattr(metadata, "candidates_token_count", 0) or 0
                total_tokens = getattr(metadata, "total_token_count", 0) or (
                    input_tokens + output_tokens
                )

                usage = TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    model=model,
                    provider=provider_norm,
                )

            text = None
            try:
                text = getattr(chunk, "text", None)
            except (AttributeError, ValueError):
                text = None

            if text:
                yield text

    elif provider_norm == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("GROQ_MODEL")

        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        if not model:
            raise ValueError("GROQ_MODEL is not configured.")

        async with AsyncGroq(api_key=api_key) as client:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                extra_body={
                    "stream_options": {"include_usage": True}
                },
            )

            async for chunk in stream:
                groq_usage = getattr(chunk, "usage", None)
                if groq_usage is None and hasattr(chunk, "x_groq") and chunk.x_groq:
                    groq_usage = getattr(chunk.x_groq, "usage", None)

                if groq_usage is not None:
                    input_tokens = getattr(groq_usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(groq_usage, "completion_tokens", 0) or 0
                    total_tokens = getattr(groq_usage, "total_tokens", 0) or (
                        input_tokens + output_tokens
                    )

                    usage = TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        model=model,
                        provider=provider_norm,
                    )

                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        yield content

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            "Supported providers: gemini, groq."
        )

    # Fallback to zero tokens if metadata was not returned by provider
    if usage is None:
        usage = TokenUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            model=model or "unknown",
            provider=provider_norm,
        )

    # Report usage only after the stream completes successfully
    if on_usage is not None:
        on_usage(usage)


async def generate(
    prompt: str,
    provider: str,
    on_usage: Callable[[TokenUsage], None] | None = None,
) -> AsyncIterator[str]:
    """Generate a streamed response with timeout and retry support."""
    provider_norm = provider.lower().strip()

    for attempt in range(MAX_RETRIES):
        received_content = False

        try:
            async with asyncio.timeout(REQUEST_TIMEOUT):
                sig = inspect.signature(_generate_once)
                if on_usage is not None and (
                    len(sig.parameters) >= 3
                    or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                ):
                    stream = _generate_once(prompt, provider_norm, on_usage=on_usage)
                else:
                    stream = _generate_once(prompt, provider_norm)

                async for chunk in stream:
                    received_content = True
                    yield chunk

            return

        except Exception as exc:
            # Never retry after partial output has been delivered.
            if received_content:
                raise

            if not is_retryable_error(exc):
                raise

            if attempt == MAX_RETRIES - 1:
                logger.error(
                    "LLM request failed after %d attempts: %s",
                    MAX_RETRIES,
                    exc,
                )
                raise

            delay = 2**attempt

            logger.warning(
                "LLM request failed. Retrying in %d seconds. "
                "Attempt %d/%d. Error: %s",
                delay,
                attempt + 1,
                MAX_RETRIES,
                exc,
            )

            await asyncio.sleep(delay)
