# AI-Doc-Assistant

A production-ready document management and AI assistant REST API built with **FastAPI**, **SQLModel**, **PostgreSQL**, **Google Gemini**, **Groq**, and **Docker**.

This service implements document management, asynchronous streaming LLM chat endpoints, actual provider token tracking, financial-grade cost calculation using `Decimal`, and persistent database storage for document metadata and LLM usage logs.

---

## Features

- **Health Check:** Verify API health and operational status.
- **Upload & Validate Documents:** Accept PDF, TXT, and DOCX files up to 10 MB with validation.
- **Persistent Database Storage:** Store document metadata in PostgreSQL using SQLModel.
- **LLM Streaming Endpoint (`POST /ask`):** Stream responses from Google Gemini and Groq chunk-by-chunk using FastAPI `StreamingResponse`.
- **Actual Token Usage Tracking:** Capture exact input, output, and total token usage metadata directly from provider SDKs without estimation.
- **Financial Precision Cost Calculation:** Compute input, output, and combined costs using Python `Decimal` per million tokens.
- **Usage Logging:** Automatically persist provider, model, token usage, latency, and cost to PostgreSQL `UsageLog` table.
- **Resilience & Exponential Backoff:** Automatic retry logic for HTTP 429, 5xx, and connection/timeout errors (up to 3 retries, no retries after partial response output).
- **Alembic Database Migrations:** Version-controlled database schema migrations.
- **Dockerized Multi-Container Setup:** One-command startup with Docker Compose and persistent PostgreSQL volume.
- **Automated Testing & Linting:** 100% mocked automated tests with Pytest and SQLite support, formatted with Ruff.

---

## Tech Stack

- **Language:** Python 3.12+
- **Framework:** FastAPI
- **Package Manager:** `uv`
- **Database & ORM:** PostgreSQL 16, SQLModel, SQLAlchemy
- **Database Migrations:** Alembic
- **LLM Providers:** Google Gemini (`google-genai`), Groq (`groq`)
- **Containerization:** Docker & Docker Compose
- **Testing & Quality:** Pytest, HTTPX, Ruff

---

## Project Structure

```text
AI-Doc-Assistant/
├── alembic/
│   ├── versions/
│   │   ├── e45e75468606_baseline_existing_schema.py
│   │   ├── c840e63ab49e_add_description_column_to_document_table.py
│   │   └── 37325e88fe95_add_usage_logs_table.py
│   ├── env.py
│   └── script.py.mako
├── app/
│   ├── __init__.py
│   ├── database.py       # Engine, connection retry, and session management
│   ├── llm.py            # Provider integration, streaming, retries, token usage extraction
│   ├── main.py           # FastAPI routes and lifecycle
│   ├── models.py         # SQLModel database models (Document, UsageLog)
│   ├── pricing.py        # Model pricing registry and cost calculation
│   └── schema.py         # Pydantic validation schemas (AskRequest, UsageResponse)
├── test/
│   ├── conftest.py       # SQLite test database setup
│   ├── test_ask_endpoint.py # POST /ask API tests
│   ├── test_documents.py    # Document upload and retrieval tests
│   ├── test_llm.py          # LLM streaming and usage extraction tests
│   ├── test_llm_retry.py    # LLM retry and backoff tests
│   └── test_pricing.py      # Pricing and cost calculation tests
├── .env.example
├── alembic.ini
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Environment Variable Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Configure the following variables in `.env`:

```env
# PostgreSQL Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=ai_doc_db
POSTGRES_PORT=5432

# Database URL used by the application
# For Docker Compose:
DATABASE_URL=postgresql://postgres:your_secure_password@db:5432/ai_doc_db
# For local host development (mapped port 5435):
# DATABASE_URL=postgresql://postgres:your_secure_password@localhost:5435/ai_doc_db

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3-flash-preview

# Groq Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-safeguard-20b

# Optional: Model pricing overrides (USD per 1M tokens)
# GEMINI_INPUT_PRICE_PER_MILLION=0.50
# GEMINI_OUTPUT_PRICE_PER_MILLION=3.00
# GROQ_INPUT_PRICE_PER_MILLION=0.075
# GROQ_OUTPUT_PRICE_PER_MILLION=0.30
```

> **Security Note:** Never commit `.env` or real API keys to version control. The repository `.gitignore` ignores `.env`.

---

## Running with Docker (Recommended)

### 1. Start Services

To build and start both the FastAPI application and PostgreSQL database containers:

```bash
docker compose up -d --build
```

The database container exposes port `5435` on the host, and the application is exposed on port `8000`:
- **API Base URL:** http://localhost:8000
- **Interactive Swagger UI:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### 2. Stop Services

```bash
docker compose down
```

PostgreSQL data is stored in the Docker volume `postgres_data`, preserving uploaded documents and usage logs across restarts.

---

## Running Locally (Without Docker)

### 1. Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### 2. Install Dependencies

```bash
uv sync
```

### 3. Run Alembic Database Migrations

Ensure your PostgreSQL instance is running, then apply the migrations:

```bash
uv run alembic upgrade head
```

Verify migration status:

```bash
uv run alembic current
```

Expected output: `37325e88fe95 (head)`

### 4. Start the Application

```bash
uv run uvicorn app.main:app --reload --port 8000
```

---

## Alembic Migration Commands

Alembic manages database schema versions:

- **Check Current Migration:**
  ```bash
  uv run alembic current
  ```
- **Apply All Pending Migrations:**
  ```bash
  uv run alembic upgrade head
  ```
- **View Migration History:**
  ```bash
  uv run alembic history --verbose
  ```
- **Rollback One Revision:**
  ```bash
  uv run alembic downgrade -1
  ```

---

## Supported LLM Providers

The application integrates two providers via asynchronous streaming:

1. **Google Gemini:**
   - Client: `google.genai.Client`
   - Streaming method: `client.aio.models.generate_content_stream`
   - Token metadata: Extracted from `chunk.usage_metadata` (`prompt_token_count`, `candidates_token_count`, `total_token_count`).
   - Default model: `gemini-3-flash-preview` (also supports `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`).

2. **Groq:**
   - Client: `groq.AsyncGroq`
   - Streaming method: `client.chat.completions.create(..., stream=True, extra_body={"stream_options": {"include_usage": True}})`
   - Token metadata: Extracted from `chunk.usage` or `chunk.x_groq.usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`).
   - Default model: `openai/gpt-oss-safeguard-20b` (also supports `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`).

---

## API Endpoints

### 1. LLM Chat Streaming: `POST /ask`

Generates an LLM response streamed back to the client, tracks token usage, calculates cost, and persists the record into the `UsageLog` database table.

#### Request Body (`application/json`):

```json
{
  "question": "Explain Python generators in two sentences.",
  "provider": "gemini"
}
```

- `question` (string, required): Question to send to the LLM (1 to 10,000 characters; whitespace-only is rejected).
- `provider` (string, optional): Selected provider (`"gemini"` or `"groq"`). Defaults to `"gemini"`. Case-insensitive.

#### Streaming Response:

- Content-Type: `text/plain; charset=utf-8`
- Transfer-Encoding: `chunked`

#### Example using `curl`:

```bash
curl -N -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Python?", "provider": "gemini"}'
```

#### Example using Python (`httpx`):

```python
import httpx

with httpx.Client() as client:
    with client.stream(
        "POST",
        "http://localhost:8000/ask",
        json={"question": "What is an async generator?", "provider": "groq"},
    ) as response:
        for chunk in response.iter_text():
            print(chunk, end="", flush=True)
```

#### Status Codes:

| Status Code | Description |
|---|---|
| `200 OK` | Response stream successfully established |
| `422 Unprocessable Entity` | Empty/blank question, invalid question length, or unsupported provider |
| `429 Too Many Requests` | Provider rate limit exceeded after all retries |
| `502 Bad Gateway` | Provider server error (5xx) or network connection failure |
| `504 Gateway Timeout` | Provider request timed out (exceeded 30 seconds) |
| `500 Internal Server Error` | Missing provider configuration or server error |

---

### 2. Document Management Endpoints

- **Health Check:** `GET /health` — Returns `{"status": "ok"}`
- **Upload Document:** `POST /documents` — Accepts `multipart/form-data` with PDF, TXT, or DOCX files up to 10 MB.
- **List Documents:** `GET /documents` — Returns a list of all uploaded documents.

---

## Token Usage Tracking & Cost Management

### Actual Token Extraction

Token counts are captured directly from provider response metadata when streaming completes:
- **Input tokens:** `prompt_token_count` (Gemini) / `prompt_tokens` (Groq)
- **Output tokens:** `candidates_token_count` (Gemini) / `completion_tokens` (Groq)
- **Total tokens:** `total_token_count` (Gemini) / `total_tokens` (Groq)

Token counts are never estimated from text length when provider usage metadata is available.

### Cost Calculation Formula

Costs are calculated using Python `Decimal` for financial precision:

$$\text{Cost} = \frac{\text{input\_tokens} \times \text{input\_price\_per\_million}}{1,000,000} + \frac{\text{output\_tokens} \times \text{output\_price\_per\_million}}{1,000,000}$$

#### Default Registered Rates:

| Provider | Model | Input Price / 1M | Output Price / 1M |
|---|---|---|---|
| Gemini | `gemini-3-flash-preview` | \$0.50 | \$3.00 |
| Gemini | `gemini-2.5-flash` / `gemini-2.0-flash` | \$0.10 | \$0.40 |
| Gemini | `gemini-1.5-flash` | \$0.075 | \$0.30 |
| Gemini | `gemini-1.5-pro` | \$1.25 | \$5.00 |
| Groq | `openai/gpt-oss-safeguard-20b` | \$0.075 | \$0.30 |
| Groq | `llama-3.3-70b-versatile` | \$0.59 | \$0.79 |
| Groq | `llama-3.1-8b-instant` | \$0.05 | \$0.08 |
| Groq | `mixtral-8x7b-32768` | \$0.24 | \$0.24 |

Custom pricing can be provided dynamically through environment variables or by calling `register_model_pricing()`.

---

## Database Models

### `UsageLog` Table

| Column | Type | Description |
|---|---|---|
| `id` | `String` (UUID) | Primary key |
| `provider` | `String` | Provider name (`gemini` or `groq`) |
| `model` | `String` | Exact model name |
| `input_tokens` | `Integer` | Number of prompt tokens |
| `output_tokens` | `Integer` | Number of completion tokens |
| `total_tokens` | `Integer` | Total tokens consumed |
| `latency_seconds` | `Float` | End-to-end request duration |
| `cost_usd` | `Numeric(20, 12)` | Calculated request cost in USD |
| `created_at` | `DateTime` (UTC) | Creation timestamp |

### `Document` Table

| Column | Type | Description |
|---|---|---|
| `id` | `String` (UUID) | Primary key |
| `filename` | `String` | Uploaded filename |
| `content_type` | `String` | MIME type |
| `size_bytes` | `Integer` | File size in bytes |
| `uploaded_at` | `DateTime` (UTC) | Upload timestamp |
| `description` | `String` (Optional) | Document description |

---

## Testing & Linting

### Running Tests

Automated tests run against an isolated SQLite test database and use mocks to prevent real API costs:

```bash
uv run pytest -v
```

Tests cover:
- **LLM Streaming & Extraction (`test/test_llm.py`):** Gemini & Groq streaming, token usage extraction, `x_groq` fallback, missing credentials.
- **LLM Retry Logic (`test/test_llm_retry.py`):** Status code retries (429, 500, 503), timeout retry, non-retryable errors, max retry attempts, no retry after partial stream.
- **Pricing & Cost Calculation (`test/test_pricing.py`):** Input-only, output-only, combined cost, negative token validation, unknown model handling, env variable overrides.
- **API Endpoints (`test/test_ask_endpoint.py`):** Successful `POST /ask` for Gemini & Groq, chunked streaming, invalid provider, empty question, rate limit handling, service error handling, timeout handling, database persistence, and rollback handling.
- **Document Management (`test/test_documents.py`):** Upload validation, size limits, format checks, document listing.

### Running Linting

```bash
uv run ruff check .
```

---

## License

This project is intended for educational and assessment purposes.
