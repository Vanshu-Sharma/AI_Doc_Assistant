# AI Document Assistant (`ai-doc-assistant`)

A production-ready FastAPI service for document ingestion, background PDF processing, pgvector semantic search, LLM response streaming, and structured data extraction.

---

## 🚀 Features

- **Document Ingestion & File Storage (`POST /documents`)**:
  - Validates document content and size (up to 10 MB).
  - Persists uploaded files to an `uploads/` volume.
  - Automatically triggers background ingestion for PDFs.
  - Returns HTTP 201 immediately with document status (`pending`).

- **Background PDF Ingestion & pgvector Storage**:
  - Uses FastAPI `BackgroundTasks` to process documents asynchronously.
  - Extracts text page-by-page (1-based index) using PyMuPDF (`fitz`).
  - Splits text into ~800 character chunks with 150-character overlap preserving natural sentence/paragraph boundaries.
  - Generates 768-dimensional dense vector embeddings using Google Gemini (`gemini-embedding-001`) or local Ollama (`nomic-embed-text`).
  - Stores text chunks and vectors in PostgreSQL using the pgvector extension (`Vector(768)`).
  - Updates document status (`pending` -> `processing` -> `ready` / `failed`).

- **Document Semantic Search (`GET /search`)**:
  - Accepts search query `q` and optional limit `k` (default 5).
  - Generates query embedding using the same model as ingestion.
  - Performs nearest-neighbor vector search in PostgreSQL using pgvector cosine distance operator `<=>`.
  - Returns top-k matching chunks ordered by relevance (closest first).
  - Each result includes `text`, `filename`, `page` number, and similarity `score`.
  - Empty or whitespace query returns HTTP 422.
  - Returns empty list `[]` if no documents are ready.

- **LLM Streaming & Cost Tracking (`POST /ask`)**:
  - Multi-provider streamed text generation (Gemini, Groq) with retries.
  - Automatically records token metrics and USD costs into `usagelog`.

- **Structured Extraction (`POST /extract`)**:
  - Extracts biographical details and skills into a validated `ResumeData` schema.
  - Built-in prompt injection defense and corrective retry logic.

---

## 🛠️ Environment Configuration

Copy `.env.example` to `.env` and fill in your keys:

```ini
# PostgreSQL Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=ai_doc_db
POSTGRES_PORT=5435

# Database URL
DATABASE_URL=postgresql://postgres:your_password@localhost:5435/ai_doc_db

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.1-flash-lite

# Groq Configuration
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-safeguard-20b

# File Upload & Ingestion Configuration
UPLOAD_DIR=uploads
EMBEDDING_PROVIDER=gemini
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

---

## 🐳 Running with Docker Compose

Start the full stack (PostgreSQL with pgvector + FastAPI application):

```bash
docker compose up -d --build
```

Interactive Swagger documentation is available at:
`http://localhost:8000/docs`

---

## 🔍 Search API Reference (`GET /search`)

### Endpoint Details
- **Path**: `GET /search`
- **Query Parameters**:
  - `q` (string, required, min length 1): Search query.
  - `k` (integer, optional, default `5`, min 1, max 50): Number of top results to return.

### Example Request

```bash
curl -X GET "http://localhost:8000/search?q=How+does+pgvector+cosine+distance+work&k=5"
```

### Example Response (HTTP 200)

```json
[
  {
    "text": "1. GET /search?q=...&k=5: embed the question with the same embedding model as Assignment 9, and return the 5 closest chunks by cosine distance (pgvector's <=> operator).\n2. Each result returns text, filename, page and score.",
    "filename": "Junior AI Engineer - Week 2 Assignment Guide.pdf",
    "page": 8,
    "page_number": 8,
    "score": 0.7421
  }
]
```

### Error Responses
- **HTTP 422 Unprocessable Content**: Returned when `q` is missing, empty, or whitespace-only.

---

## 📊 Assignment 10: 5-Question Retrieval Evaluation

To verify retrieval accuracy, five domain-specific questions were evaluated against the ingested document (`Junior AI Engineer - Week 2 Assignment Guide.pdf`), checking whether the known correct page appears in the top-5 search results:

| # | Question | Target Page | Retrieved Top 5 Pages | Found in Top 5? | Rank | Similarity Score |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: |
| **1** | *"Where should API keys like GEMINI_API_KEY and GROQ_API_KEY be stored and how are dummy values handled?"* | **Page 2** | `[2, 1, 3, 2, 4]` | ✅ **Yes** | **#1** | `0.7251` |
| **2** | *"How to generate the first Alembic migration and apply schema changes with upgrade head?"* | **Page 3** | `[3, 2, 2, 2, 3]` | ✅ **Yes** | **#1** | `0.7380` |
| **3** | *"What token usage metrics and cost calculation are recorded after every LLM call?"* | **Page 4** | `[4, 4, 1, 1, 9]` | ✅ **Yes** | **#1** | `0.7104` |
| **4** | *"How does background ingestion save uploaded PDFs to uploads folder and extract pages with PyMuPDF?"* | **Page 7** | `[7, 7, 7, 1, 6]` | ✅ **Yes** | **#1** | `0.7512` |
| **5** | *"How does GET search use pgvector cosine distance operator to return closest chunks?"* | **Page 8** | `[8, 7, 1, 7, 7]` | ✅ **Yes** | **#1** | `0.7421` |

### **Evaluation Result: 5/5 found in top 5 (100% Top-1 Accuracy)**

All five questions successfully retrieved their exact target page as the top result rank (#1), verifying that pgvector cosine distance indexing and chunking operate correctly.

---

## 🧪 Testing

Run the automated test suite with pytest:

```bash
uv run pytest
```

Run code formatting and linting checks:

```bash
uv run ruff check .
```
