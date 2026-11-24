"""Unit and integration tests for GET /search document search endpoint."""

from collections.abc import Generator
import uuid

from fastapi.testclient import TestClient
import pytest
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

import app.database as database_module
import app.ingestion as ingestion_module
from app.database import get_session
from app.main import app
from app.models import Chunk, Document, DocumentStatus

# In-memory SQLite engine for testing
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@pytest.fixture(name="session")
def session_fixture() -> Generator[Session, None, None]:
    """Provide an isolated database session."""
    SQLModel.metadata.create_all(test_engine)
    with Session(test_engine) as session:
        yield session
    SQLModel.metadata.drop_all(test_engine)


@pytest.fixture(name="client")
def client_fixture(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
    """Provide TestClient with overridden database session and mocked embeddings."""

    def get_session_override() -> Generator[Session, None, None]:
        yield session

    app.dependency_overrides[get_session] = get_session_override
    orig_engine = database_module.engine
    database_module.engine = test_engine

    # Default mock embedding: unit vector in 1st dimension
    mock_vec = [0.0] * 768
    mock_vec[0] = 1.0

    monkeypatch.setattr(
        ingestion_module,
        "generate_embedding",
        lambda *args, **kwargs: list(mock_vec),
    )

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()
    database_module.engine = orig_engine


def test_search_missing_query_returns_422(client: TestClient):
    """Verify GET /search without 'q' parameter returns HTTP 422."""
    response = client.get("/search")
    assert response.status_code == 422


def test_search_empty_query_returns_422(client: TestClient):
    """Verify GET /search with empty 'q' returns HTTP 422."""
    response = client.get("/search?q=")
    assert response.status_code == 422


def test_search_whitespace_query_returns_422(client: TestClient):
    """Verify GET /search with whitespace-only 'q' returns HTTP 422."""
    response = client.get("/search?q=%20%20%20")
    assert response.status_code == 422
    assert "cannot be empty or whitespace" in response.json()["detail"]


def test_search_no_documents_returns_empty_list(client: TestClient):
    """Verify search returns an empty list when no documents exist."""
    response = client.get("/search?q=deep+learning")
    assert response.status_code == 200
    assert response.json() == []


def test_search_only_pending_documents_returns_empty_list(
    client: TestClient, session: Session
):
    """Verify search ignores documents that are not READY."""
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename="pending.pdf",
        content_type="application/pdf",
        size_bytes=1000,
        status=DocumentStatus.PENDING,
    )
    session.add(doc)

    chk = Chunk(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        filename="pending.pdf",
        page_number=1,
        chunk_index=0,
        text="Pending document text about neural networks.",
        embedding=[1.0] + [0.0] * 767,
    )
    session.add(chk)
    session.commit()

    response = client.get("/search?q=neural+networks")
    assert response.status_code == 200
    assert response.json() == []


def test_search_successful_ranking_and_metadata(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
):
    """Verify search returns top results sorted closest first with complete metadata."""
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename="ai_guide.pdf",
        content_type="application/pdf",
        size_bytes=2048,
        status=DocumentStatus.READY,
    )
    session.add(doc)

    # Chunks with distinct directional vectors
    vec_close = [0.9] + [0.1] * 767
    vec_mid = [0.5] + [0.5] * 767
    vec_far = [0.0] + [0.9] * 767

    chk1 = Chunk(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        filename="ai_guide.pdf",
        page_number=1,
        chunk_index=0,
        text="High relevance chunk about transformer attention mechanisms.",
        embedding=vec_close,
    )
    chk2 = Chunk(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        filename="ai_guide.pdf",
        page_number=2,
        chunk_index=1,
        text="Medium relevance chunk discussing neural networks in general.",
        embedding=vec_mid,
    )
    chk3 = Chunk(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        filename="ai_guide.pdf",
        page_number=3,
        chunk_index=2,
        text="Low relevance chunk about data preprocessing pipelines.",
        embedding=vec_far,
    )
    session.add(chk1)
    session.add(chk2)
    session.add(chk3)
    session.commit()

    # Query vector matches vec_close most closely
    query_vec = [1.0] + [0.0] * 767
    monkeypatch.setattr(
        ingestion_module,
        "generate_embedding",
        lambda *args, **kwargs: query_vec,
    )

    response = client.get("/search?q=attention+mechanisms&k=3")
    assert response.status_code == 200

    results = response.json()
    assert len(results) == 3

    # Closest first: chk1 > chk2 > chk3
    assert results[0]["filename"] == "ai_guide.pdf"
    assert results[0]["page"] == 1
    assert "High relevance" in results[0]["text"]
    assert results[0]["score"] > results[1]["score"]
    assert results[1]["score"] > results[2]["score"]

    # Verify score is float
    for r in results:
        assert isinstance(r["score"], float)
        assert "text" in r
        assert "filename" in r
        assert "page" in r


def test_search_k_parameter_limits_count(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
):
    """Verify the 'k' parameter controls the number of results returned."""
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename="manual.pdf",
        content_type="application/pdf",
        size_bytes=4096,
        status=DocumentStatus.READY,
    )
    session.add(doc)

    for i in range(10):
        chk = Chunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            filename="manual.pdf",
            page_number=i + 1,
            chunk_index=i,
            text=f"Chunk content for section {i + 1}",
            embedding=[0.1 * (i + 1)] + [0.05] * 767,
        )
        session.add(chk)
    session.commit()

    # Request top 3
    response_k3 = client.get("/search?q=manual+section&k=3")
    assert response_k3.status_code == 200
    assert len(response_k3.json()) == 3

    # Request top 1
    response_k1 = client.get("/search?q=manual+section&k=1")
    assert response_k1.status_code == 200
    assert len(response_k1.json()) == 1


def test_search_embedding_failure_handled_gracefully(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
):
    """Verify embedding provider failure returns 503 without crashing the app."""
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename="valid.pdf",
        content_type="application/pdf",
        size_bytes=1000,
        status=DocumentStatus.READY,
    )
    session.add(doc)
    session.commit()

    def failing_embedding(*args, **kwargs):
        raise RuntimeError("Embedding API rate limit or network timeout")

    monkeypatch.setattr(ingestion_module, "generate_embedding", failing_embedding)

    response = client.get("/search?q=failing+query")
    assert response.status_code == 503
    assert "Search service error" in response.json()["detail"]


def test_search_assignment_query_strictly_respects_k(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
):
    """Verify that assignment-specific queries strictly respect the 'k' limit."""
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename="course_guide.pdf",
        content_type="application/pdf",
        size_bytes=4096,
        status=DocumentStatus.READY,
    )
    session.add(doc)

    c1 = Chunk(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        filename="course_guide.pdf",
        page_number=1,
        chunk_index=0,
        text="Assignment 2: Document Storage Pipeline\nGoal: Build file upload and persistence.",
        embedding=[0.9] + [0.1] * 767,
    )
    c2 = Chunk(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        filename="course_guide.pdf",
        page_number=2,
        chunk_index=1,
        text="What to Build:\n1. POST /documents: upload endpoint up to 10MB.\n2. Store file locally.",
        embedding=[0.85] + [0.1] * 767,
    )
    c3 = Chunk(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        filename="course_guide.pdf",
        page_number=3,
        chunk_index=2,
        text="Checklist for Assignment 2:\nVerify test coverage and error handling.",
        embedding=[0.8] + [0.1] * 767,
    )
    session.add(c1)
    session.add(c2)
    session.add(c3)
    session.commit()

    monkeypatch.setattr(
        ingestion_module,
        "generate_embedding",
        lambda *args, **kwargs: [1.0] + [0.0] * 767,
    )

    response_k1 = client.get("/search?q=Explain+Assignment+2&k=1")
    assert response_k1.status_code == 200
    assert len(response_k1.json()) == 1

    response_k2 = client.get("/search?q=Explain+Assignment+2&k=2")
    assert response_k2.status_code == 200
    assert len(response_k2.json()) == 2


def test_search_similar_chunks_max_candidates_bounds_queries(
    session: Session, monkeypatch: pytest.MonkeyPatch
):
    """Verify search_similar_chunks bounds candidate and continuation queries by max_candidates."""
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename="guide.pdf",
        content_type="application/pdf",
        size_bytes=2048,
        status=DocumentStatus.READY,
    )
    session.add(doc)

    c0 = Chunk(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        filename="guide.pdf",
        page_number=1,
        chunk_index=0,
        text="Assignment 2: Setup\nGoal: Build API.",
        embedding=[0.0] * 768,
    )
    c1 = Chunk(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        filename="guide.pdf",
        page_number=2,
        chunk_index=1,
        text="Implementation details for the endpoint.",
        embedding=[0.0] * 768,
    )
    session.add(c0)
    session.add(c1)
    session.commit()

    monkeypatch.setattr(
        ingestion_module,
        "generate_embedding",
        lambda *args, **kwargs: [1.0] + [0.0] * 767,
    )

    orig_exec = session.exec
    limits: list[int] = []

    def spied_exec(statement, *args, **kwargs):
        if hasattr(statement, "_limit") and statement._limit is not None:
            limits.append(statement._limit)
        return orig_exec(statement, *args, **kwargs)

    session.exec = spied_exec  # type: ignore[assignment]

    results = ingestion_module.search_similar_chunks(
        query_text="Explain Assignment 2",
        k=5,
        session=session,
        max_candidates=2,
    )

    # Both candidate query and continuation query applied .limit(2)
    assert limits == [2, 2]
    assert len(results) == 2
    assert results[0][0].id == c0.id
    assert results[1][0].id == c1.id


def test_search_similar_chunks_configurable_env_limit(
    session: Session, monkeypatch: pytest.MonkeyPatch
):
    """Verify MAX_RETRIEVAL_CANDIDATES environment variable bounds candidate queries."""
    monkeypatch.setenv("MAX_RETRIEVAL_CANDIDATES", "3")

    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename="guide.pdf",
        content_type="application/pdf",
        size_bytes=2048,
        status=DocumentStatus.READY,
    )
    session.add(doc)

    c0 = Chunk(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        filename="guide.pdf",
        page_number=1,
        chunk_index=0,
        text="Assignment 3: Data Pipeline\nBuild pipeline.",
        embedding=[0.0] * 768,
    )
    c1 = Chunk(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        filename="guide.pdf",
        page_number=2,
        chunk_index=1,
        text="Step 1: Ingest records.",
        embedding=[0.0] * 768,
    )
    session.add(c0)
    session.add(c1)
    session.commit()

    monkeypatch.setattr(
        ingestion_module,
        "generate_embedding",
        lambda *args, **kwargs: [1.0] + [0.0] * 767,
    )

    orig_exec = session.exec
    limits: list[int] = []

    def spied_exec(statement, *args, **kwargs):
        if hasattr(statement, "_limit") and statement._limit is not None:
            limits.append(statement._limit)
        return orig_exec(statement, *args, **kwargs)

    session.exec = spied_exec  # type: ignore[assignment]

    results = ingestion_module.search_similar_chunks(
        query_text="Explain Assignment 3",
        k=5,
        session=session,
    )

    # MAX_RETRIEVAL_CANDIDATES=3 was used as limit for candidate and continuation queries
    assert limits == [3, 3]
    assert len(results) == 2



