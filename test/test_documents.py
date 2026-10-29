import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.pool import StaticPool
from app.database import get_session
from app.main import MAX_UPLOAD_BYTES, app
from app.models import Document


@pytest.fixture
def test_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    SQLModel.metadata.create_all(engine)

    yield engine

    SQLModel.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def client(test_engine):
    def override_get_session():
        with Session(test_engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_session, None)
        
def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_valid_txt_file(client):
    file_content = b"This is a test document."

    response = client.post(
        "/documents",
        files={
            "file": (
                "sample.txt",
                file_content,
                "text/plain",
            )
        },
    )

    assert response.status_code == 201

    data = response.json()
    assert data["filename"] == "sample.txt"
    assert data["size_bytes"] == len(file_content)
    assert "id" in data
    assert "uploaded_at" in data


def test_list_documents(client):
    client.post(
        "/documents",
        files={
            "file": (
                "sample.txt",
                b"Hello world",
                "text/plain",
            )
        },
    )

    response = client.get("/documents")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["filename"] == "sample.txt"


def test_upload_empty_file(client):
    response = client.post(
        "/documents",
        files={
            "file": (
                "empty.txt",
                b"",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_upload_unsupported_file_type(client):
    response = client.post(
        "/documents",
        files={
            "file": (
                "sample.exe",
                b"not an allowed file",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 415


def test_upload_file_exact_10_mb(client):
    exact_content = b"a" * MAX_UPLOAD_BYTES

    response = client.post(
        "/documents",
        files={
            "file": (
                "exact_10mb.txt",
                exact_content,
                "text/plain",
            )
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "exact_10mb.txt"
    assert data["size_bytes"] == MAX_UPLOAD_BYTES


def test_upload_file_over_10_mb(client):
    large_content = b"a" * (MAX_UPLOAD_BYTES + 1)

    response = client.post(
        "/documents",
        files={
            "file": (
                "large.txt",
                large_content,
                "text/plain",
            )
        },
    )

    assert response.status_code == 413


def test_document_persisted_in_database(client, test_engine):
    content = b"Database persistence test content."
    response = client.post(
        "/documents",
        files={
            "file": (
                "persisted.txt",
                content,
                "text/plain",
            )
        },
    )

    assert response.status_code == 201
    doc_id = response.json()["id"]

    # Verify directly from database session that the document row exists
    with Session(test_engine) as session:
        doc = session.get(Document, doc_id)
        assert doc is not None
        assert doc.filename == "persisted.txt"
        assert doc.size_bytes == len(content)
        assert doc.content_type == "text/plain"
