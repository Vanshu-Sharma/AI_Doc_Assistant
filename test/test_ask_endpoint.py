from decimal import Decimal
from unittest.mock import MagicMock, patch
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, SQLModel, delete, select

from app.database import engine
from app.llm import TokenUsage, build_rag_prompt, is_greeting
from app.main import app, save_usage_log
from app.models import Chunk, Document, DocumentStatus, UsageLog


@pytest.fixture(autouse=True)
def clean_usage_db():
    """Ensure every test starts with an empty database."""
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.exec(delete(Chunk))
        session.exec(delete(Document))
        session.exec(delete(UsageLog))
        session.commit()

    yield

    with Session(engine) as session:
        session.exec(delete(Chunk))
        session.exec(delete(Document))
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


def test_ask_llm_provider_tpm_rate_limit_413(client):
    """Verify Groq TPM 413 rate limit error is mapped to HTTP 429."""
    class MockGroqTPMError(Exception):
        status_code = 413

    async def mock_tpm_limit(prompt, provider, on_usage=None):
        raise MockGroqTPMError(
            "Error code: 413 - {'error': {'message': 'Request too large for model on tokens per minute (TPM): Limit 2000', 'code': 'rate_limit_exceeded'}}"
        )
        yield "never"

    with patch("app.main.generate", side_effect=mock_tpm_limit):
        response = client.post(
            "/ask",
            json={"question": "Large query?", "provider": "groq"},
        )

        assert response.status_code == 429
        assert response.json()["detail"] == "LLM provider rate limit exceeded."


def test_ask_llm_provider_payload_too_large_413(client):
    """Verify standard 413 payload too large is mapped to HTTP 413."""
    class Mock413Error(Exception):
        status_code = 413

    async def mock_payload_large(prompt, provider, on_usage=None):
        raise Mock413Error("Payload too large")
        yield "never"

    with patch("app.main.generate", side_effect=mock_payload_large):
        response = client.post(
            "/ask",
            json={"question": "Huge payload?", "provider": "groq"},
        )

        assert response.status_code == 413
        assert response.json()["detail"] == "LLM request payload exceeds model context or rate limits."


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


def test_build_rag_prompt_with_chunks():
    chunks = [
        (
            MagicMock(
                filename="guide.pdf",
                page_number=2,
                text="API keys should be stored in .env only.",
            ),
            0.85,
        ),
        MagicMock(
            filename="notes.txt",
            page_number=5,
            text="Run alembic upgrade head to apply migrations.",
        ),
    ]

    prompt = build_rag_prompt(
        question="Where should API keys be kept?",
        chunks=chunks,
    )

    assert "Context from uploaded documents:" in prompt
    assert "[Document: guide.pdf | Page: 2]" in prompt
    assert "API keys should be stored in .env only." in prompt
    assert "[Document: notes.txt | Page: 5]" in prompt
    assert "Run alembic upgrade head to apply migrations." in prompt
    assert "User Question: Where should API keys be kept?" in prompt
    assert "I couldn't find this information in the provided documents." in prompt
    assert "Source: <filename>, Page <page_number>" in prompt


def test_build_rag_prompt_empty_chunks():
    for empty_chunks in [[], None]:
        prompt = build_rag_prompt(
            question="What is the capital of France?",
            chunks=empty_chunks,
        )
        assert "No relevant document excerpts were found." in prompt
        assert "User Question: What is the capital of France?" in prompt
        assert "I couldn't find this information in the provided documents." in prompt


def test_ask_grounded_in_documents_with_citations(client):
    doc_id = str(uuid.uuid4())
    mock_vec = [0.05] * 768

    with Session(engine) as session:
        session.exec(delete(Chunk))
        session.exec(delete(Document))
        session.commit()

        doc = Document(
            id=doc_id,
            filename="guide.pdf",
            content_type="application/pdf",
            size_bytes=2048,
            status=DocumentStatus.READY,
        )
        session.add(doc)

        chunk = Chunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            filename="guide.pdf",
            page_number=2,
            chunk_index=0,
            text="Put keys in .env only: GEMINI_API_KEY, GROQ_API_KEY.",
            embedding=mock_vec,
        )
        session.add(chunk)
        session.commit()

    captured_prompt = None

    async def mock_generate(prompt, provider, on_usage=None):
        nonlocal captured_prompt
        captured_prompt = prompt
        yield "Keys should be in .env.\n\n"
        yield "Source: guide.pdf, Page 2"
        if on_usage:
            on_usage(
                TokenUsage(
                    input_tokens=50,
                    output_tokens=20,
                    total_tokens=70,
                    model="gemini-3.1-flash-lite",
                    provider="gemini",
                )
            )

    with patch("app.ingestion.generate_embedding", return_value=mock_vec):
        with patch("app.main.generate", side_effect=mock_generate):
            response = client.post(
                "/ask",
                json={
                    "question": "Where should keys be stored?",
                    "provider": "gemini",
                },
            )

            assert response.status_code == 200
            assert "Source: guide.pdf, Page 2" in response.text
            assert "Keys should be in .env." in response.text

            # Check that prompt contains the retrieved chunk and instructions
            assert captured_prompt is not None
            assert "[Document: guide.pdf | Page: 2]" in captured_prompt
            assert "Put keys in .env only: GEMINI_API_KEY, GROQ_API_KEY." in captured_prompt
            assert "User Question: Where should keys be stored?" in captured_prompt


def test_ask_when_answer_not_in_documents_fallback(client):
    async def mock_generate(prompt, provider, on_usage=None):
        yield "I couldn't find this information in the provided documents."
        if on_usage:
            on_usage(
                TokenUsage(
                    input_tokens=30,
                    output_tokens=10,
                    total_tokens=40,
                    model="gemini-3.1-flash-lite",
                    provider="gemini",
                )
            )

    with patch("app.main.search_similar_chunks", return_value=[]):
        with patch("app.main.generate", side_effect=mock_generate):
            response = client.post(
                "/ask",
                json={
                    "question": "What is the speed of light in vacuum?",
                    "provider": "gemini",
                },
            )

            assert response.status_code == 200
            assert response.text == "I couldn't find this information in the provided documents."


def test_ask_retrieval_failure_raises_503(client):
    """Verify that retrieval infrastructure failures raise HTTP 503 instead of silent fallback."""
    with patch("app.main.search_similar_chunks", side_effect=RuntimeError("Vector DB unavailable")):
        response = client.post(
            "/ask",
            json={
                "question": "Explain Assignment 2",
                "provider": "gemini",
            },
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "Document retrieval service is temporarily unavailable."


def test_ask_retrieval_empty_succeeds_with_fallback(client):
    """Verify that when retrieval succeeds but finds no chunks, normal fallback occurs."""
    captured_prompt = None

    async def mock_generate(prompt, provider, on_usage=None):
        nonlocal captured_prompt
        captured_prompt = prompt
        yield "I couldn't find this information in the provided documents."

    with patch("app.main.search_similar_chunks", return_value=[]):
        with patch("app.main.generate", side_effect=mock_generate):
            response = client.post(
                "/ask",
                json={
                    "question": "What is quantum computing?",
                    "provider": "gemini",
                },
            )

            assert response.status_code == 200
            assert response.text == "I couldn't find this information in the provided documents."
            assert captured_prompt is not None
            assert "No relevant document excerpts were found." in captured_prompt


def test_ask_greeting_skips_retrieval_even_if_retrieval_fails(client):
    """Verify greetings succeed without calling retrieval even if retrieval would raise an error."""
    async def mock_generate(prompt, provider, on_usage=None):
        yield "Hello! How can I help you today?"

    with patch("app.main.search_similar_chunks", side_effect=RuntimeError("DB exploded")):
        with patch("app.main.generate", side_effect=mock_generate):
            response = client.post(
                "/ask",
                json={
                    "question": "Hello",
                    "provider": "gemini",
                },
            )

            assert response.status_code == 200
            assert "Hello!" in response.text


@pytest.mark.parametrize(
    "greeting_text",
    [
        "Hi",
        "Hello",
        "Good morning",
        "good morning!",
        "Hey",
        "Hey there",
        "Greetings",
        "Howdy",
    ],
)
def test_ask_greeting_handling_without_document_retrieval(client, greeting_text):
    """Verify greetings return natural conversational responses without triggering retrieval."""
    assert is_greeting(greeting_text) is True

    captured_prompt = None

    async def mock_generate(prompt, provider, on_usage=None):
        nonlocal captured_prompt
        captured_prompt = prompt
        yield "Hello! How can I assist you with your documents today?"
        if on_usage:
            on_usage(
                TokenUsage(
                    input_tokens=15,
                    output_tokens=12,
                    total_tokens=27,
                    model="gemini-3.1-flash-lite",
                    provider="gemini",
                )
            )

    with patch("app.main.search_similar_chunks") as mock_search:
        with patch("app.main.generate", side_effect=mock_generate):
            response = client.post(
                "/ask",
                json={"question": greeting_text, "provider": "gemini"},
            )

            # Verification: Document retrieval was NEVER triggered
            mock_search.assert_not_called()

            assert response.status_code == 200
            assert "Hello! How can I assist you" in response.text
            assert "couldn't find this information" not in response.text
            assert captured_prompt is not None
            assert "friendly and helpful AI Document Assistant" in captured_prompt


def test_ask_greeting_with_question_triggers_retrieval(client):
    """Verify that a greeting followed by a document question DOES trigger document retrieval."""
    query = "Hello, what is pgvector?"
    assert is_greeting(query) is False

    mock_search = MagicMock(return_value=[])

    async def mock_generate(prompt, provider, on_usage=None):
        yield "I couldn't find this information in the provided documents."

    with patch("app.main.search_similar_chunks", mock_search):
        with patch("app.main.generate", side_effect=mock_generate):
            response = client.post(
                "/ask",
                json={"question": query, "provider": "gemini"},
            )

            # Verification: Search WAS called because it is a substantive query
            mock_search.assert_called_once()
            assert response.status_code == 200


def test_ask_assignment_retrieval_and_continuation_chunks(client):
    """Verify that assignment-specific queries retrieve both the heading chunk and continuation chunks."""
    doc_id = str(uuid.uuid4())
    mock_vec = [0.05] * 768

    with Session(engine) as session:
        session.exec(delete(Chunk))
        session.exec(delete(Document))
        session.commit()

        doc = Document(
            id=doc_id,
            filename="syllabus.pdf",
            content_type="application/pdf",
            size_bytes=3000,
            status=DocumentStatus.READY,
        )
        session.add(doc)

        # Chunk 0: Syllabus overview (vector similarity might be high, but not the specific assignment)
        c0 = Chunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            filename="syllabus.pdf",
            page_number=1,
            chunk_index=0,
            text="Course Syllabus. This guide explains each assignment in the program in detail.",
            embedding=mock_vec,
        )
        # Chunk 1: Assignment 1
        c1 = Chunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            filename="syllabus.pdf",
            page_number=2,
            chunk_index=1,
            text="Assignment 1: Git and GitHub setup.\nLearn git workflow.",
            embedding=mock_vec,
        )
        # Chunk 2: Assignment 2 header and goal
        c2 = Chunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            filename="syllabus.pdf",
            page_number=3,
            chunk_index=2,
            text="Assignment 2: Document Storage Pipeline\nGoal: Build file upload and persistence.",
            embedding=mock_vec,
        )
        # Chunk 3: Assignment 2 continuation (What to build / steps) - does not have 'Assignment 2' string
        c3 = Chunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            filename="syllabus.pdf",
            page_number=3,
            chunk_index=3,
            text="What to Build:\n1. POST /documents: upload endpoint up to 10MB.\n2. Checklist: Upload returns 201.",
            embedding=mock_vec,
        )
        # Chunk 4: Assignment 8
        c4 = Chunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            filename="syllabus.pdf",
            page_number=5,
            chunk_index=4,
            text="Assignment 8: Model Race\nGoal: Compare ML models with LLMs.",
            embedding=mock_vec,
        )
        for chk in [c0, c1, c2, c3, c4]:
            session.add(chk)
        session.commit()

    captured_prompt = None

    async def mock_generate(prompt, provider, on_usage=None):
        nonlocal captured_prompt
        captured_prompt = prompt
        yield "Assignment 2 is the Document Storage Pipeline.\n\n"
        yield "Source: syllabus.pdf, Page 3"

    with patch("app.ingestion.generate_embedding", return_value=mock_vec):
        with patch("app.main.generate", side_effect=mock_generate):
            response = client.post(
                "/ask",
                json={"question": "Explain Assignment 2", "provider": "gemini"},
            )

            assert response.status_code == 200
            assert "Source: syllabus.pdf, Page 3" in response.text

            # Critical assertions: Both Chunk 2 (header) and Chunk 3 (continuation steps) are in prompt
            assert captured_prompt is not None
            assert "Assignment 2: Document Storage Pipeline" in captured_prompt
            assert "What to Build:" in captured_prompt
            assert "POST /documents: upload endpoint up to 10MB" in captured_prompt


def test_ask_global_relevance_filtering_preserved(client):
    """Verify that unrelated out-of-domain queries still trigger fallback without weakening filtering."""
    doc_id = str(uuid.uuid4())
    mock_vec = [0.05] * 768

    with Session(engine) as session:
        session.exec(delete(Chunk))
        session.exec(delete(Document))
        session.commit()

        doc = Document(
            id=doc_id,
            filename="syllabus.pdf",
            content_type="application/pdf",
            size_bytes=3000,
            status=DocumentStatus.READY,
        )
        session.add(doc)
        chk = Chunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            filename="syllabus.pdf",
            page_number=1,
            chunk_index=0,
            text="Assignment 8: Model Race\nCompare classification accuracy.",
            embedding=mock_vec,
        )
        session.add(chk)
        session.commit()

    captured_prompt = None

    async def mock_generate(prompt, provider, on_usage=None):
        nonlocal captured_prompt
        captured_prompt = prompt
        yield "I couldn't find this information in the provided documents."

    with patch("app.ingestion.generate_embedding", return_value=mock_vec):
        with patch("app.main.generate", side_effect=mock_generate):
            response = client.post(
                "/ask",
                json={"question": "Who won the 2022 FIFA World Cup?", "provider": "gemini"},
            )

            assert response.status_code == 200
            assert response.text == "I couldn't find this information in the provided documents."
            assert captured_prompt is not None
            assert "User Question: Who won the 2022 FIFA World Cup?" in captured_prompt


