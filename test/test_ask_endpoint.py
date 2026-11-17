from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, SQLModel, delete, select

from app.database import engine
from app.llm import TokenUsage
from app.main import app, save_usage_log
from app.models import UsageLog


@pytest.fixture(autouse=True)
def clean_usage_db():
    """Ensure every test starts with an empty usagelog table."""
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.exec(delete(UsageLog))
        session.commit()

    yield

    with Session(engine) as session:
        session.exec(delete(UsageLog))
        session.commit()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_ask_gemini_success_and_streaming(client):
    async def mock_generate(prompt, provider, on_usage=None):
        yield "Gemini "
        yield "answers "
        yield "your "
        yield "question."
        if on_usage:
            on_usage(
                TokenUsage(
                    input_tokens=100,
                    output_tokens=50,
                    total_tokens=150,
                    model="gemini-3-flash-preview",
                    provider="gemini",
                )
            )

    with patch("app.main.generate", side_effect=mock_generate):
        response = client.post(
            "/ask",
            json={"question": "What is Python?", "provider": "gemini"},
        )

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert response.text == "Gemini answers your question."

        with Session(engine) as session:
            logs = session.exec(select(UsageLog)).all()
            assert len(logs) == 1
            log = logs[0]
            assert log.provider == "gemini"
            assert log.model == "gemini-3-flash-preview"
            assert log.input_tokens == 100
            assert log.output_tokens == 50
            assert log.total_tokens == 150
            assert log.latency_seconds >= 0
            assert log.cost_usd > Decimal("0")
            assert log.created_at is not None


def test_ask_groq_success(client):
    async def mock_generate(prompt, provider, on_usage=None):
        yield "Groq "
        yield "is "
        yield "fast!"
        if on_usage:
            on_usage(
                TokenUsage(
                    input_tokens=200,
                    output_tokens=80,
                    total_tokens=280,
                    model="openai/gpt-oss-safeguard-20b",
                    provider="groq",
                )
            )

    with patch("app.main.generate", side_effect=mock_generate):
        response = client.post(
            "/ask",
            json={"question": "How fast is Groq?", "provider": "groq"},
        )

        assert response.status_code == 200
        assert response.text == "Groq is fast!"

        with Session(engine) as session:
            logs = session.exec(select(UsageLog)).all()
            assert len(logs) == 1
            log = logs[0]
            assert log.provider == "groq"
            assert log.model == "openai/gpt-oss-safeguard-20b"
            assert log.input_tokens == 200
            assert log.output_tokens == 80
            assert log.total_tokens == 280
            assert log.cost_usd > Decimal("0")


def test_ask_invalid_provider(client):
    response = client.post(
        "/ask",
        json={"question": "Hello", "provider": "unsupported_provider"},
    )

    assert response.status_code == 422
    with Session(engine) as session:
        logs = session.exec(select(UsageLog)).all()
        assert len(logs) == 0


def test_ask_empty_question(client):
    # Empty string
    response_empty = client.post(
        "/ask",
        json={"question": "", "provider": "gemini"},
    )
    assert response_empty.status_code == 422

    # Whitespace only
    response_whitespace = client.post(
        "/ask",
        json={"question": "     ", "provider": "gemini"},
    )
    assert response_whitespace.status_code == 422

    with Session(engine) as session:
        assert len(session.exec(select(UsageLog)).all()) == 0


def test_ask_streaming_response_chunks(client):
    expected_chunks = ["Chunk 1 ", "Chunk 2 ", "Chunk 3"]

    async def mock_generate(prompt, provider, on_usage=None):
        for c in expected_chunks:
            yield c
        if on_usage:
            on_usage(TokenUsage(10, 10, 20, model="gemini-3-flash-preview", provider="gemini"))

    with patch("app.main.generate", side_effect=mock_generate):
        with client.stream(
            "POST",
            "/ask",
            json={"question": "Stream test", "provider": "gemini"},
        ) as response:
            assert response.status_code == 200
            received_chunks = list(response.iter_text())
            assert received_chunks == expected_chunks


def test_ask_llm_provider_timeout_failure(client):
    async def mock_timeout(prompt, provider, on_usage=None):
        raise TimeoutError("Request timed out")
        yield "never"

    with patch("app.main.generate", side_effect=mock_timeout):
        response = client.post(
            "/ask",
            json={"question": "Will this timeout?", "provider": "gemini"},
        )

        assert response.status_code == 504
        assert response.json()["detail"] == "LLM request timed out."

        with Session(engine) as session:
            assert len(session.exec(select(UsageLog)).all()) == 0


def test_ask_llm_provider_rate_limit(client):
    class MockRateLimitError(Exception):
        status_code = 429

    async def mock_rate_limit(prompt, provider, on_usage=None):
        raise MockRateLimitError("Rate limit exceeded")
        yield "never"

    with patch("app.main.generate", side_effect=mock_rate_limit):
        response = client.post(
            "/ask",
            json={"question": "Rate limited?", "provider": "gemini"},
        )

        assert response.status_code == 429
        assert response.json()["detail"] == "LLM provider rate limit exceeded."

        with Session(engine) as session:
            assert len(session.exec(select(UsageLog)).all()) == 0


def test_ask_llm_provider_server_error(client):
    class Mock503Error(Exception):
        status_code = 503

    async def mock_server_error(prompt, provider, on_usage=None):
        raise Mock503Error("Service Unavailable")
        yield "never"

    with patch("app.main.generate", side_effect=mock_server_error):
        response = client.post(
            "/ask",
            json={"question": "Server error?", "provider": "groq"},
        )

        assert response.status_code == 502
        assert response.json()["detail"] == "LLM provider service error."

        with Session(engine) as session:
            assert len(session.exec(select(UsageLog)).all()) == 0


def test_ask_database_failure_handling(client, caplog):
    async def mock_generate(prompt, provider, on_usage=None):
        yield "Hello DB failure"
        if on_usage:
            on_usage(TokenUsage(10, 10, 20, model="gemini-3-flash-preview", provider="gemini"))

    with patch("app.main.generate", side_effect=mock_generate):
        with patch("app.main.save_usage_log", side_effect=SQLAlchemyError("Simulated DB failure")):
            response = client.post(
                "/ask",
                json={"question": "DB test", "provider": "gemini"},
            )

            # Response still completed streaming to the user
            assert response.status_code == 200
            assert response.text == "Hello DB failure"
            # Ensure the failure was logged
            assert any(
                "Database persistence failed" in record.message
                for record in caplog.records
            )


def test_save_usage_log_rollback():
    mock_session = MagicMock()
    mock_session.commit.side_effect = SQLAlchemyError("Commit failed")

    with pytest.raises(SQLAlchemyError):
        save_usage_log(
            session=mock_session,
            provider="gemini",
            model="gemini-3-flash-preview",
            input_tokens=10,
            output_tokens=10,
            total_tokens=20,
            latency_seconds=1.23,
            cost_usd=Decimal("0.001"),
        )

    mock_session.rollback.assert_called_once()


def test_no_usage_persistence_on_midstream_failure(client):
    async def mock_midstream_fail(prompt, provider, on_usage=None):
        yield "First chunk delivered. "
        raise RuntimeError("Midstream network disconnection")

    with patch("app.main.generate", side_effect=mock_midstream_fail):
        # The client will receive the first chunk and then the stream ends due to error
        response = client.post(
            "/ask",
            json={"question": "Fail halfway?", "provider": "gemini"},
        )

        assert response.status_code == 200
        assert response.text == "First chunk delivered. "

        # Since it failed mid-stream, incomplete usage must NOT be persisted
        with Session(engine) as session:
            assert len(session.exec(select(UsageLog)).all()) == 0
