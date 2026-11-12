# AI-Doc-Assistant

A production-ready document management and AI assistant REST API built with **FastAPI**, **SQLModel**, **PostgreSQL**, **Google Gemini**, **Groq**, and **Docker**.

This service implements document management, asynchronous streaming LLM chat endpoints, actual provider token tracking, financial-grade cost calculation using `Decimal`, and persistent database storage for document metadata and LLM usage logs.

---

## Features

- **Health Check:** Verify API health and operational status.
- **Upload & Validate Documents:** Accept PDF, TXT, and DOCX files up to 10 MB with validation.
- **Persistent Database Storage:** Store document metadata in PostgreSQL using SQLModel.
- **LLM Streaming Endpoint (`POST /ask`):** Stream responses from Google Gemini and Groq chunk-by-chunk using FastAPI `StreamingResponse`.
- **Structured Output Resume Extraction (`POST /extract`):** Extract structured candidate details (`name`, `email`, `phone`, `skills`) from unstructured resume text using native provider structured output (`response_schema` / JSON mode).
- **Pydantic Validation & One-Time Self-Correction Retry:** Strong schema validation (`EmailStr`, list of strings, null for missing fields) with exactly one automatic corrective retry when validation fails.
- **Prompt Injection Defense:** Strict prompt security treating resume text as untrusted data to protect against prompt overrides.
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
│   ├── llm.py            # Provider integration, streaming, structured output, retries
│   ├── main.py           # FastAPI routes and lifecycle (POST /ask, POST /extract)
│   ├── models.py         # SQLModel database models (Document, UsageLog)
│   ├── pricing.py        # Model pricing registry and cost calculation
│   └── schema.py         # Pydantic validation schemas (AskRequest, ResumeData, ExtractRequest)
├── data/
│   └── resumes/          # Directory for 20 trainer-provided fictional resumes & expected.json
├── scripts/
│   └── eval_extract.py   # Dataset evaluation script measuring extraction accuracy
├── test/
│   ├── conftest.py          # SQLite test database setup
│   ├── test_ask_endpoint.py # POST /ask API tests
│   ├── test_documents.py    # Document upload and retrieval tests
│   ├── test_extract.py      # POST /extract, Pydantic validation, and retry tests
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

### 2. Resume Information Extraction: `POST /extract`

Extracts structured candidate information (`name`, `email`, `phone`, `skills`) from unstructured resume text using native provider structured output and enforces strict Pydantic validation with a single corrective retry.

#### Request Body (`application/json`):

```json
{
  "resume_text": "Rahul Sharma is a Python developer. Email: rahul@gmail.com. Skills: Python, SQL.",
  "provider": "gemini"
}
```

- `resume_text` (string, required): Raw text of the resume to extract data from.
- `provider` (string, optional): `"gemini"` or `"groq"`. Defaults to `"gemini"`.

#### Successful Response (`application/json`, HTTP 200):

```json
{
  "name": "Rahul Sharma",
  "email": "rahul@gmail.com",
  "phone": null,
  "skills": ["Python", "SQL"]
}
```

#### Explanation of Structured Output
Large Language Models generate probabilistic unstructured text by default. Structured output constrains the model's generation to strictly follow a defined JSON schema:
- **Google Gemini:** Leverages the `google-genai` SDK with `types.GenerateContentConfig(response_mime_type="application/json", response_schema=ResumeData)`. This directs the Gemini model engine to constrain generation directly according to the schema.
- **Groq:** Uses `response_format={"type": "json_object"}` to guarantee that the output is syntactically valid JSON.

#### Explanation of Pydantic Validation
While JSON mode guarantees valid JSON syntax, it does not guarantee semantic data correctness or strict field compliance. Pydantic validation parses the raw JSON into the strongly-typed `ResumeData` model:
- `name`: string or `null` if missing
- `email`: validated RFC-compliant email address via `EmailStr` or `null` if missing
- `phone`: string or `null` if missing
- `skills`: list of strings (`Field(default_factory=list)`)

If any field fails validation (for instance, an invalid email address format), Pydantic raises a `ValidationError` which triggers the self-correction mechanism.

#### One-Time Self-Correction Retry Mechanism
When LLM extraction output fails Pydantic validation or is malformed:
1. Attempt 1 response is parsed and validated against `ResumeData`.
2. If validation fails, the Pydantic error details are captured.
3. Exactly one retry attempt is executed, appending the captured validation error into a corrective prompt to instruct the model to correct its mistakes.
4. If Attempt 2 passes validation, the validated `ResumeData` is returned with HTTP 200.
5. If Attempt 2 also fails validation, the endpoint returns `HTTP 422 Unprocessable Entity` with a clear explanation of the validation failure.
6. A maximum of two LLM attempts is strictly enforced (no indefinite loops).
7. Network and provider errors (429, 5xx, timeouts) are never converted into validation retries; they are handled independently by provider resilience logic.

#### Prompt Ordering & Claude Prompt Caching
The extraction system prompt places the long, fixed system instructions, extraction rules, schema definition, and input/output example at the beginning of the prompt, while the variable, dynamic resume text is positioned at the very end. This structure enables APIs supporting prompt caching (such as Anthropic Claude or Google Gemini) to identify and cache the static prompt prefix across calls. This can significantly reduce repeated input-token processing costs and reduce generation latency for high-volume extraction workloads.

#### Running the Evaluation Script
To evaluate extraction accuracy against the 20 trainer-provided fictional resumes:

```bash
uv run python scripts/eval_extract.py --provider gemini --pause 1.5
```

The script will:
1. Load all 20 resumes from `data/resumes/` and ground truth from `data/resumes/expected.json`.
2. Extract fields for each resume with a rate-limiting pause between requests.
3. Compare actual values against expected ground truth across all 4 fields (`name`, `email`, `phone`, `skills`).
4. Print a summary in the required format:
   ```text
   Evaluation Results
   ------------------
   Total resumes: 20
   Valid responses: 19/20
   Fields correct: 74/80
   ```

*Note on Evaluation Score:* The actual evaluation score will be documented here once the trainer provides the official 20-resume dataset and `expected.json`. Synthetic dataset substitutions are not fabricated per assignment requirements.

---

### 3. Document Management Endpoints

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
- **Resume Extraction & Structured Output (`test/test_extract.py`):** `POST /extract` endpoint, missing fields returning `null`, skills as list, `EmailStr` format validation, malformed JSON handling, validation failure followed by successful retry, double failure returning HTTP 422, prompt injection resistance, provider error status mapping, and evaluation script normalization.
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
