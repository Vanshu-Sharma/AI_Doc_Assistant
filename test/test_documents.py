
from fastapi.testclient import TestClient
from app.main import app, documents

client = TestClient(app)


def setup_function():
    documents.clear()


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_successful_upload():
    response = client.post(
        "/documents",
        files={"file": ("report.txt", b"Hello FastAPI", "text/plain")},
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "report.txt"
    assert response.json()["size_bytes"] == 13


def test_empty_file():
    response = client.post(
        "/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400


def test_wrong_file_type():
    response = client.post(
        "/documents",
        files={"file": ("script.exe", b"content", "application/octet-stream")},
    )

    assert response.status_code == 415


def test_oversized_file():
    response = client.post(
        "/documents",
        files={
            "file": (
                "large.txt",
                b"x" * (10 * 1024 * 1024 + 1),
                "text/plain",
            )
        },
    )

    assert response.status_code == 413


def test_list_documents():
    client.post(
        "/documents",
        files={"file": ("report.txt", b"Hello", "text/plain")},
    )

    response = client.get("/documents")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["filename"] == "report.txt"

