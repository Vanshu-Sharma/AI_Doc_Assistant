import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import llm
from app.llm import TokenUsage, _generate_once, generate, is_retryable_error


def test_is_retryable_error():
    assert is_retryable_error(TimeoutError()) is True
    assert is_retryable_error(ConnectionError()) is True

    # Status code on exception object
    class CustomErr(Exception):
        def __init__(self, status_code):
            self.status_code = status_code

    assert is_retryable_error(CustomErr(429)) is True
    assert is_retryable_error(CustomErr(500)) is True
    assert is_retryable_error(CustomErr(503)) is True
    assert is_retryable_error(CustomErr(400)) is False
    assert is_retryable_error(CustomErr(401)) is False
    assert is_retryable_error(CustomErr(404)) is False

    # Status code on response object
    class ResponseErr(Exception):
        def __init__(self, status_code):
            self.response = MagicMock(status_code=status_code)

    assert is_retryable_error(ResponseErr(429)) is True
    assert is_retryable_error(ResponseErr(502)) is True
    assert is_retryable_error(ResponseErr(403)) is False


def test_unsupported_provider():
    async def run():
        async for _ in _generate_once("Hello", "unsupported_provider"):
            pass

    with pytest.raises(ValueError, match="Unsupported provider"):
        asyncio.run(run())


def test_missing_gemini_credentials(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3-flash-preview")

    async def run_missing_key():
        async for _ in _generate_once("Hello", "gemini"):
            pass

    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        asyncio.run(run_missing_key())

    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    async def run_missing_model():
        async for _ in _generate_once("Hello", "gemini"):
            pass

    with pytest.raises(ValueError, match="GEMINI_MODEL is not configured"):
        asyncio.run(run_missing_model())


def test_missing_groq_credentials(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-safeguard-20b")

    async def run_missing_key():
        async for _ in _generate_once("Hello", "groq"):
            pass

    with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
        asyncio.run(run_missing_key())

    monkeypatch.setenv("GROQ_API_KEY", "dummy_key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    async def run_missing_model():
        async for _ in _generate_once("Hello", "groq"):
            pass

    with pytest.raises(ValueError, match="GROQ_MODEL is not configured"):
        asyncio.run(run_missing_model())


def test_gemini_streaming_and_token_usage(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")

    chunk1 = MagicMock()
    chunk1.text = "Hello "
    chunk1.usage_metadata = None

    chunk2 = MagicMock()
    chunk2.text = "world!"
    usage_meta = MagicMock()
    usage_meta.prompt_token_count = 15
    usage_meta.candidates_token_count = 25
    usage_meta.total_token_count = 40
    chunk2.usage_metadata = usage_meta

    async def mock_gemini_stream(*args, **kwargs):
        async def _stream():
            yield chunk1
            yield chunk2
        return _stream()

    mock_client = MagicMock()
    mock_client.aio.models.generate_content_stream = mock_gemini_stream

    with patch.object(llm.genai, "Client", return_value=mock_client):
        recorded_usage = None

        def record_usage(usage: TokenUsage):
            nonlocal recorded_usage
            recorded_usage = usage

        async def run():
            chunks = []
            async for chunk in generate("Hi", "gemini", on_usage=record_usage):
                chunks.append(chunk)
            return chunks

        result = asyncio.run(run())

        assert result == ["Hello ", "world!"]
        assert recorded_usage is not None
        assert recorded_usage.input_tokens == 15
        assert recorded_usage.output_tokens == 25
        assert recorded_usage.total_tokens == 40
        assert recorded_usage.provider == "gemini"
        assert recorded_usage.model == "gemini-test-model"


def test_groq_streaming_and_token_usage(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
    monkeypatch.setenv("GROQ_MODEL", "groq-test-model")

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="Groq "))]
    chunk1.usage = None
    chunk1.x_groq = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content="stream!"))]
    chunk2.usage = None
    chunk2.x_groq = None

    chunk3 = MagicMock()
    chunk3.choices = []
    groq_usage = MagicMock()
    groq_usage.prompt_tokens = 10
    groq_usage.completion_tokens = 20
    groq_usage.total_tokens = 30
    chunk3.usage = groq_usage
    chunk3.x_groq = None

    async def mock_groq_stream(*args, **kwargs):
        yield chunk1
        yield chunk2
        yield chunk3

    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create = AsyncMock(side_effect=mock_groq_stream)

    class MockAsyncGroq:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return mock_client_instance

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch.object(llm, "AsyncGroq", MockAsyncGroq):
        recorded_usage = None

        def record_usage(usage: TokenUsage):
            nonlocal recorded_usage
            recorded_usage = usage

        async def run():
            chunks = []
            async for chunk in generate("Hi", "groq", on_usage=record_usage):
                chunks.append(chunk)
            return chunks

        result = asyncio.run(run())

        assert result == ["Groq ", "stream!"]
        assert recorded_usage is not None
        assert recorded_usage.input_tokens == 10
        assert recorded_usage.output_tokens == 20
        assert recorded_usage.total_tokens == 30
        assert recorded_usage.provider == "groq"
        assert recorded_usage.model == "groq-test-model"


def test_groq_x_groq_usage_fallback(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
    monkeypatch.setenv("GROQ_MODEL", "groq-test-model")

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="Fallback"))]
    chunk1.usage = None
    x_groq = MagicMock()
    x_groq.usage = MagicMock(prompt_tokens=8, completion_tokens=12, total_tokens=20)
    chunk1.x_groq = x_groq

    async def mock_groq_stream(*args, **kwargs):
        yield chunk1

    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create = AsyncMock(side_effect=mock_groq_stream)

    class MockAsyncGroq:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return mock_client_instance

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch.object(llm, "AsyncGroq", MockAsyncGroq):
        recorded_usage = None

        def record_usage(usage: TokenUsage):
            nonlocal recorded_usage
            recorded_usage = usage

        async def run():
            chunks = []
            async for chunk in generate("Hi", "groq", on_usage=record_usage):
                chunks.append(chunk)
            return chunks

        result = asyncio.run(run())

        assert result == ["Fallback"]
        assert recorded_usage is not None
        assert recorded_usage.input_tokens == 8
        assert recorded_usage.output_tokens == 12
        assert recorded_usage.total_tokens == 20
