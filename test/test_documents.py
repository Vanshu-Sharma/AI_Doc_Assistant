import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, delete

from app.database import engine
from app.models import Document, DocumentStatus
from app.main import MAX_UPLOAD_BYTES, app


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


def test_upload_duplicate_file_returns_409(client):
    """Verify that uploading a file with the same name returns 409 Conflict."""
    file_content = b"Content of first upload."

    # 1. First upload succeeds
    response1 = client.post(
        "/documents",
        files={
            "file": (
                "report.txt",
                file_content,
                "text/plain",
            )
        },
    )
    assert response1.status_code == 201

    # 2. Second upload with same name is rejected
    response2 = client.post(
        "/documents",
        files={
            "file": (
                "report.txt",
                file_content,
                "text/plain",
            )
        },
    )
    assert response2.status_code == 409
    assert "already uploaded" in response2.json()["detail"].lower()

    # 3. Third upload is also rejected
    response3 = client.post(
        "/documents",
        files={
            "file": (
                "report.txt",
                b"Different content same name",
                "text/plain",
            )
        },
    )
    assert response3.status_code == 409
    assert "already uploaded" in response3.json()["detail"].lower()


def test_upload_duplicate_content_different_filename_returns_409(client):
    """Verify that uploading identical content under a different filename is rejected."""
    identical_content = b"Exact identical byte content for testing hash deduplication."

    # 1. First upload succeeds
    response1 = client.post(
        "/documents",
        files={
            "file": (
                "original.txt",
                identical_content,
                "text/plain",
            )
        },
    )
    assert response1.status_code == 201

    # 2. Upload same content with a different name
    response2 = client.post(
        "/documents",
        files={
            "file": (
                "duplicate_copy.txt",
                identical_content,
                "text/plain",
            )
        },
    )
    assert response2.status_code == 409
    assert "already uploaded" in response2.json()["detail"].lower()
    assert "original.txt" in response2.json()["detail"]


def test_upload_different_files_both_succeed(client):
    """Verify that uploading different files with different content both succeed."""
    res1 = client.post(
        "/documents",
        files={"file": ("doc_a.txt", b"Content A", "text/plain")},
    )
    assert res1.status_code == 201

    res2 = client.post(
        "/documents",
        files={"file": ("doc_b.txt", b"Content B", "text/plain")},
    )
    assert res2.status_code == 201


def test_reupload_after_failed_upload_succeeds(client):
    """Verify that if a document failed previously, re-uploading the same filename succeeds."""
    with Session(engine) as session:
        failed_doc = Document(
            id="failed-id-123",
            filename="failed_doc.txt",
            content_type="text/plain",
            size_bytes=100,
            status=DocumentStatus.FAILED,
            error_message="Previous parsing error",
        )
        session.add(failed_doc)
        session.commit()

    # Re-upload should replace the failed document and succeed
    res = client.post(
        "/documents",
        files={"file": ("failed_doc.txt", b"Recovered content", "text/plain")},
    )
    assert res.status_code == 201
    assert res.json()["filename"] == "failed_doc.txt"

