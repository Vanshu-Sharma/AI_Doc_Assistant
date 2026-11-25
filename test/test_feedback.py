from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest
from sqlmodel import Session, SQLModel, delete, select

from app.database import engine
from app.llm import TokenUsage
from app.main import app
from app.models import Feedback
from app.tracing import (
    create_trace_id,
    get_langfuse_client,
    record_feedback_score,
    trace_generation,
)


@pytest.fixture(autouse=True)
def clean_feedback_db():
    """Ensure every test starts with an empty feedback table."""
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.exec(delete(Feedback))
        session.commit()

    yield

    with Session(engine) as session:
        session.exec(delete(Feedback))
        session.commit()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_submit_feedback_thumbs_up(client):
    response = client.post(
        "/feedback",
        json={
            "trace_id": "trace-12345",
            "feedback": "thumbs_up",
            "comment": "Great answer!",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["trace_id"] == "trace-12345"
    assert data["feedback"] == "thumbs_up"
    assert data["score"] == 1.0
    assert data["comment"] == "Great answer!"
    assert "id" in data
    assert "created_at" in data

    # Verify persisted in database
    with Session(engine) as session:
        records = session.exec(select(Feedback).where(Feedback.trace_id == "trace-12345")).all()
        assert len(records) == 1
        assert records[0].score == 1.0
        assert records[0].feedback == "thumbs_up"


def test_submit_feedback_thumbs_down(client):
    response = client.post(
        "/feedback",
        json={
            "trace_id": "trace-67890",
            "feedback": "thumbs_down",
            "comment": "Factually incorrect",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["trace_id"] == "trace-67890"
    assert data["feedback"] == "thumbs_down"
    assert data["score"] == 0.0
    assert data["comment"] == "Factually incorrect"

    with Session(engine) as session:
        records = session.exec(select(Feedback).where(Feedback.trace_id == "trace-67890")).all()
        assert len(records) == 1
        assert records[0].score == 0.0
        assert records[0].feedback == "thumbs_down"


def test_submit_feedback_aliases(client):
    # Test numeric value 1
    resp1 = client.post("/feedback", json={"trace_id": "trace-up", "value": 1})
    assert resp1.status_code == 200
    assert resp1.json()["feedback"] == "thumbs_up"
    assert resp1.json()["score"] == 1.0

    # Test string 'down'
    resp2 = client.post("/feedback", json={"trace_id": "trace-down", "rating": "down"})
    assert resp2.status_code == 200
    assert resp2.json()["feedback"] == "thumbs_down"
    assert resp2.json()["score"] == 0.0

    # Test score 0
    resp3 = client.post("/feedback", json={"trace_id": "trace-zero", "score": 0})
    assert resp3.status_code == 200
    assert resp3.json()["feedback"] == "thumbs_down"
    assert resp3.json()["score"] == 0.0


def test_submit_feedback_invalid_values(client):
    # Missing rating
    resp1 = client.post("/feedback", json={"trace_id": "trace-missing"})
    assert resp1.status_code == 422

    # Invalid rating string
    resp2 = client.post(
        "/feedback",
        json={"trace_id": "trace-invalid", "feedback": "so-so"},
    )
    assert resp2.status_code == 422

    # Missing trace_id
    resp3 = client.post("/feedback", json={"feedback": "thumbs_up"})
    assert resp3.status_code == 422

    # Blank trace_id
    resp4 = client.post("/feedback", json={"trace_id": "   ", "feedback": "thumbs_up"})
    assert resp4.status_code == 422


def test_list_feedback(client):
    client.post("/feedback", json={"trace_id": "trace-a", "feedback": "thumbs_up"})
    client.post("/feedback", json={"trace_id": "trace-b", "feedback": "thumbs_down"})

    # List all
    all_resp = client.get("/feedback")
    assert all_resp.status_code == 200
    all_data = all_resp.json()
    assert len(all_data) == 2

    # Filter by trace_id
    filter_resp = client.get("/feedback?trace_id=trace-a")
    assert filter_resp.status_code == 200
    filter_data = filter_resp.json()
    assert len(filter_data) == 1
    assert filter_data[0]["trace_id"] == "trace-a"


def test_ask_generates_x_trace_id_header(client):
    async def mock_generate(prompt, provider, on_usage=None):
        yield "Hello world"
        if on_usage:
            on_usage(
                TokenUsage(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    model="gemini-3-flash-preview",
                    provider="gemini",
                )
            )

    with patch("app.main.generate", side_effect=mock_generate):
        resp = client.post(
            "/ask",
            json={"question": "Test question", "provider": "gemini"},
        )
        assert resp.status_code == 200
        assert "X-Trace-Id" in resp.headers
        trace_id = resp.headers["X-Trace-Id"]
        assert len(trace_id) > 0

        # Submit feedback using the returned trace_id
        fb_resp = client.post(
            "/feedback",
            json={
                "trace_id": trace_id,
                "feedback": "thumbs_up",
                "comment": "Nice answer",
            },
        )
        assert fb_resp.status_code == 200
        assert fb_resp.json()["trace_id"] == trace_id


def test_ask_with_custom_trace_id(client):
    async def mock_generate(prompt, provider, on_usage=None):
        yield "Custom trace output"

    with patch("app.main.generate", side_effect=mock_generate):
        resp = client.post(
            "/ask",
            json={
                "question": "Test question",
                "provider": "gemini",
                "trace_id": "client-custom-trace-999",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-Trace-Id") == "client-custom-trace-999"


def test_tracing_module_helpers():
    # Test client instantiation
    client = get_langfuse_client()
    assert client is not None

    # Test trace ID generation
    tid = create_trace_id()
    assert isinstance(tid, str)
    assert len(tid) > 0

    # Test trace_generation handles valid and edge-case inputs without raising
    trace_generation(
        trace_id="test-trace-id",
        name="test-generation",
        prompt="User prompt",
        output="Model output",
        model="test-model",
        provider="gemini",
        usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
        cost_usd=0.001,
    )

    # Test record_feedback_score
    record_feedback_score(
        trace_id="test-trace-id",
        score=1.0,
        comment="test feedback",
    )


def test_langfuse_score_exception_does_not_break_feedback_endpoint(client):
    with patch("app.main.record_feedback_score", side_effect=RuntimeError("Langfuse unreachable")):
        response = client.post(
            "/feedback",
            json={
                "trace_id": "trace-resilience",
                "feedback": "thumbs_up",
            },
        )
        # Should still succeed and persist to database even if Langfuse is unreachable
        assert response.status_code == 200
        assert response.json()["trace_id"] == "trace-resilience"


def test_ask_traces_full_rag_prompt_to_langfuse(client):
    async def mock_generate(prompt, provider, on_usage=None):
        yield "Answer text"

    with patch("app.main.generate", side_effect=mock_generate), \
         patch("app.main.trace_generation") as mock_trace:
        resp = client.post(
            "/ask",
            json={"question": "What is Python?", "provider": "gemini"},
        )
        assert resp.status_code == 200
        assert mock_trace.called
        call_kwargs = mock_trace.call_args.kwargs
        # The prompt logged to Langfuse must be the full prompt including instructions/context
        assert "strictly grounded AI document assistant" in call_kwargs["prompt"]
        assert "Context from uploaded documents:" in call_kwargs["prompt"]
        assert "User Question: What is Python?" in call_kwargs["prompt"]
        assert call_kwargs["prompt"] != "What is Python?"


def test_list_feedback_ordering_deterministic(client):
    with Session(engine) as session:
        f1 = Feedback(
            trace_id="trace-older",
            feedback="thumbs_up",
            score=1.0,
            created_at=datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        )
        f2 = Feedback(
            trace_id="trace-newer",
            feedback="thumbs_down",
            score=0.0,
            created_at=datetime(2026, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )
        session.add(f1)
        session.add(f2)
        session.commit()

    resp = client.get("/feedback")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    # Newest feedback should appear first
    assert data[0]["trace_id"] == "trace-newer"
    assert data[1]["trace_id"] == "trace-older"

