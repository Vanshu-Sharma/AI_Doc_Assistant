import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, delete

from app.database import engine
from app.models import Document
from app.main import MAX_UPLOAD_BYTES, app, get_document_store
from app.models import Document_Data

@pytest.fixture(autouse=True)
def clean_database():
    """Ensure every test starts with an empty documents table."""

    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        session.exec(delete(Document))
        session.commit()

    yield

    with Session(engine) as session:
        session.exec(delete(Document))
        session.commit()

@pytest.fixture(autouse=True)
def isolated_document_store():
    """Provide a fresh document store for each test."""
    store: list[Document_Data] = []

    app.dependency_overrides[get_document_store] = lambda: store

    yield store

    app.dependency_overrides.clear()


@pytest.fixture
def client():
    """Start FastAPI with its lifespan for every test."""
    with TestClient(app) as test_client:
        yield test_client


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
