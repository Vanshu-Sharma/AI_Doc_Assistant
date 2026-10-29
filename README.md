# AI-Doc-Assistant

A beginner-friendly document management REST API built with **FastAPI**, **SQLModel**, **PostgreSQL**, and **Docker**. This project implements document uploads, metadata validation, persistent database storage, and containerized deployment with one command.

## Features

- **Health Check:** Verify that the API is running.
- **Upload Documents:** Upload PDF, TXT, and DOCX files.
- **File Validation:** Reject unsupported file types, empty files, and files larger than 10 MB.
- **Persistent Database Storage:** Store document metadata in a PostgreSQL database using SQLModel with automatic UUID generation.
- **List Documents:** Retrieve metadata for all uploaded documents directly from the database.
- **Dockerized Multi-Container Setup:** Run the FastAPI app and PostgreSQL database with a single `docker compose up` command.
- **Data Persistence:** Database data is backed by a Docker volume, persisting across restarts and container lifecycle (`docker compose down` and `docker compose up`).
- **Interactive Documentation:** Explore and test endpoints through Swagger UI.
- **Automated Testing:** Test API functionality using Pytest with SQLite support (no external Postgres instance required).

## Tech Stack

- Python 3.12+
- FastAPI
- SQLModel / SQLAlchemy
- PostgreSQL 16
- Psycopg / Psycopg2
- Docker & Docker Compose
- Uvicorn
- Pytest
- HTTPX
- uv — Python project and dependency manager

## Project Structure

```text
AI-Doc-Assistant/
├── app/
│   ├── __init__.py
│   ├── database.py
│   ├── main.py
│   └── models.py
├── test/
│   ├── conftest.py
│   └── test_documents.py
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

## Running with Docker (Recommended)

Run the entire application and database with one command:

### 1. Configure Environment Variables

Copy the example environment configuration:

```bash
cp .env.example .env
```

The default values in `.env` are configured to connect the app container to the PostgreSQL service inside Docker Compose.

### 2. Start Services

```bash
docker compose up --build
```

To run in the background (detached mode):

```bash
docker compose up -d --build
```

The services will be available at:
- **Base URL:** http://localhost:8000
- **Swagger UI (/docs):** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

### 3. Stop Services

```bash
docker compose down
```

The PostgreSQL data volume (`postgres_data`) persists after `docker compose down`. When starting up again with `docker compose up -d`, all previously uploaded documents remain intact.

---

## Running Locally (Without Docker)

### 1. Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

Verify installations:

```bash
python --version
uv --version
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure `.env`

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

If running without Docker, set `DATABASE_URL` to your local PostgreSQL instance (or omit it to default to a local SQLite database for local exploration):

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_doc_db
```

### 4. Start the application

```bash
uv run uvicorn app.main:app --reload
```

---

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

Checks whether the API is running.

```json
{
  "status": "ok"
}
```

Expected status code: `200 OK`

### 2. Upload a Document

**Endpoint:** `POST /documents`

Uploads a document and persists its metadata in the PostgreSQL `Document` table.

**Supported file formats:**
- PDF (`.pdf`)
- Text (`.txt`)
- Word document (`.docx`)

**File size limit:** 10 MB.

**Example success response:**

```json
{
  "id": "f12c3250-66b2-4786-b667-e1f5eab307af",
  "filename": "example.pdf",
  "content_type": "application/pdf",
  "size_bytes": 1024,
  "uploaded_at": "2026-10-29T10:00:00Z"
}
```

**Status codes:**

| Status code | Description |
|---|---|
| `201 Created` | Document uploaded and saved to database successfully |
| `400 Bad Request` | File is empty or filename/extension is missing |
| `413 Content Too Large` | File exceeds the 10 MB limit |
| `415 Unsupported Media Type` | Unsupported file extension |

### 3. List Uploaded Documents

**Endpoint:** `GET /documents`

Returns metadata for all documents stored in the database.

**Example response:**

```json
[
  {
    "id": "f12c3250-66b2-4786-b667-e1f5eab307af",
    "filename": "example.pdf",
    "content_type": "application/pdf",
    "size_bytes": 1024,
    "uploaded_at": "2026-10-29T10:00:00Z"
  }
]
```

Expected status code: `200 OK`

---

## Testing

Tests run using an isolated SQLite database so PostgreSQL does not need to be running:

```bash
uv run pytest
```

Or with verbose output:

```bash
uv run pytest -v
```

All tests pass independently of external database infrastructure.

---

## Database Model

The `Document` table contains:
- `id`: `String` (UUID primary key)
- `filename`: `String` (original filename)
- `content_type`: `String` (MIME type)
- `size_bytes`: `Integer` (file size in bytes)
- `uploaded_at`: `DateTime` (timestamp of upload in UTC)

## License

This project is intended for educational and learning purposes.
