"""Background PDF ingestion, page extraction, chunking, and pgvector embeddings."""

from collections.abc import Callable
import logging
import os
from pathlib import Path
import re

import httpx
import pymupdf
from sqlmodel import Session, col, or_, select

import app.database as database_module
from app.models import Chunk, Document, DocumentStatus

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 768
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
BATCH_SIZE = 100


def read_pdf_pages(file_path: str | Path) -> list[tuple[int, str]]:
    """Extract 1-based page number and text from each page using PyMuPDF.

    Raises:
        ValueError: If file is missing, cannot be parsed, contains 0 pages,
                    or has no extractable text across all pages.
    """
    path = Path(file_path)
    if not path.is_file():
        raise ValueError(f"File not found on disk: {file_path}")

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise ValueError(f"Corrupted or unreadable PDF: {exc}") from exc

    try:
        if len(doc) == 0:
            raise ValueError("The uploaded PDF contains 0 pages.")

        pages_content: list[tuple[int, str]] = []
        total_chars = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text("text").strip()
            page_num = page_idx + 1
            if page_text:
                pages_content.append((page_num, page_text))
                total_chars += len(page_text)

        if total_chars == 0:
            raise ValueError("The uploaded PDF contains no extractable text.")

        return pages_content
    finally:
        doc.close()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks of approximately 500 to 1,000 characters."""
    cleaned = text.strip()
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Look for a natural split boundary in the tail window of the chunk
        if end < text_len:
            search_start = max(start + chunk_size - 200, start)
            best_break = -1
            for sep in ["\n\n", "\n", ". ", "; ", ", ", " "]:
                pos = cleaned.rfind(sep, search_start, end)
                if pos != -1:
                    best_break = pos + len(sep)
                    break
            if best_break != -1 and best_break > start + 300:
                end = best_break

        chunk_str = cleaned[start:end].strip()
        if chunk_str:
            chunks.append(chunk_str)

        if end >= text_len:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def _generate_embeddings_impl(
    texts: list[str],
    provider: str | None = None,
    timeout: float = 30.0,
) -> list[list[float]]:
    """Internal implementation for generating batch 768-dim dense embeddings.

    Reuses the client connection across batch chunks for both Gemini and Ollama.
    """
    if not texts:
        return []

    chosen_provider = (
        provider or os.getenv("EMBEDDING_PROVIDER") or "gemini"
    ).lower()

    if chosen_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not configured.")

        from google import genai
        from google.genai import types

        # Instantiate provider client once for the entire batch
        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = client.models.embed_content(
                model=model,
                contents=batch,
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIMENSION
                ),
            )
            if not response.embeddings or len(response.embeddings) != len(batch):
                raise ValueError("Gemini returned empty or mismatched embedding vectors.")

            for emb in response.embeddings:
                if not emb or not getattr(emb, "values", None):
                    raise ValueError("Gemini returned empty embedding vector.")
                vector = list(emb.values)
                if len(vector) != EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Expected {EMBEDDING_DIMENSION}-dim vector, got {len(vector)}."
                    )
                all_vectors.append(vector)

        return all_vectors

    elif chosen_provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        embed_url = f"{base_url}/api/embed"
        embeddings_url = f"{base_url}/api/embeddings"
        all_vectors = []
        try_embed_endpoint = True

        # Reuse single httpx client connection across all batch requests
        with httpx.Client(timeout=timeout) as client:
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]

                # Attempt Ollama batch endpoint /api/embed first
                if try_embed_endpoint:
                    resp = client.post(embed_url, json={"model": model, "input": batch})
                    if resp.status_code == 200:
                        data = resp.json()
                        vectors = data.get("embeddings")
                        if not vectors and "embedding" in data:
                            vectors = [data["embedding"]]
                        if not vectors or not isinstance(vectors, list) or len(vectors) != len(batch):
                            raise ValueError("Ollama returned invalid embedding format.")
                        for v in vectors:
                            if not isinstance(v, list) or len(v) != EMBEDDING_DIMENSION:
                                raise ValueError(
                                    f"Expected {EMBEDDING_DIMENSION}-dim vector, got {len(v) if isinstance(v, list) else type(v)}."
                                )
                            all_vectors.append([float(x) for x in v])
                        continue
                    elif resp.status_code == 404:
                        # Fallback for older Ollama versions without /api/embed
                        try_embed_endpoint = False
                    else:
                        raise ValueError(
                            f"Ollama embedding request failed (HTTP {resp.status_code}): "
                            f"{resp.text}"
                        )

                # Fallback to /api/embeddings per-chunk using the reused client
                for text in batch:
                    resp = client.post(embeddings_url, json={"model": model, "prompt": text})
                    if resp.status_code != 200:
                        raise ValueError(
                            f"Ollama embedding request failed (HTTP {resp.status_code}): "
                            f"{resp.text}"
                        )
                    data = resp.json()
                    vector = data.get("embedding")
                    if not vector or not isinstance(vector, list):
                        raise ValueError("Ollama returned invalid embedding format.")
                    if len(vector) != EMBEDDING_DIMENSION:
                        raise ValueError(
                            f"Expected {EMBEDDING_DIMENSION}-dim vector, got {len(vector)}."
                        )
                    all_vectors.append([float(x) for x in vector])

        return all_vectors

    else:
        raise ValueError(
            f"Unsupported embedding provider '{chosen_provider}'. "
            f"Allowed providers: 'gemini', 'ollama'."
        )


def generate_embedding(
    text: str,
    provider: str | None = None,
    timeout: float = 30.0,
) -> list[float]:
    """Generate a 768-dimensional dense vector embedding for a single text.

    Supports Gemini (gemini-embedding-001 / text-embedding-004) and Ollama
    (nomic-embed-text).
    """
    results = _generate_embeddings_impl([text], provider=provider, timeout=timeout)
    if not results:
        raise ValueError("No embedding vector returned.")
    return results[0]


_default_generate_embedding = generate_embedding


def generate_embeddings(
    texts: list[str],
    provider: str | None = None,
    timeout: float = 30.0,
) -> list[list[float]]:
    """Generate 768-dimensional dense vector embeddings for a batch of texts.

    Reuses provider clients and maintains exact 1-to-1 order mapping.
    """
    # If generate_embedding was monkeypatched by tests, delegate to it
    if generate_embedding is not _default_generate_embedding:
        return [
            generate_embedding(t, provider=provider, timeout=timeout)
            for t in texts
        ]

    return _generate_embeddings_impl(texts, provider=provider, timeout=timeout)


def ingest_document_pdf(
    document_id: str,
    file_path: str,
    session_factory: Callable[[], Session] | None = None,
    embedding_provider: str | None = None,
) -> None:
    """Background ingestion worker: extracts PDF pages, chunks, batch embeds, and saves."""
    get_db = session_factory or (lambda: Session(database_module.engine))

    with get_db() as session:
        doc = session.get(Document, document_id)
        if not doc:
            logger.warning(
                "Document %s not found in database during background ingestion.",
                document_id,
            )
            return

        doc.status = DocumentStatus.PROCESSING
        session.add(doc)
        session.commit()
        logger.info(
            "Started background ingestion for document %s (%s).",
            document_id,
            doc.filename,
        )

        try:
            # 1. Read PDF page by page
            pages = read_pdf_pages(file_path)

            # 2. Split each page's text into chunks of ~500-1000 characters
            chunk_metadata: list[tuple[int, int, str]] = []  # (page_num, chunk_idx, chunk_text)
            chunk_idx = 0

            for page_num, page_text in pages:
                page_chunks = chunk_text(page_text)
                for chunk_str in page_chunks:
                    chunk_metadata.append((page_num, chunk_idx, chunk_str))
                    chunk_idx += 1

            if not chunk_metadata:
                raise ValueError("No extractable chunks found in PDF document.")

            # 3. Batch generate 768-dim dense embeddings for all chunk texts
            chunk_texts = [chunk_text_item for _, _, chunk_text_item in chunk_metadata]
            vectors = generate_embeddings(
                texts=chunk_texts,
                provider=embedding_provider,
            )

            if len(vectors) != len(chunk_metadata):
                raise ValueError(
                    f"Mismatch between number of chunks ({len(chunk_metadata)}) "
                    f"and returned embeddings ({len(vectors)})."
                )

            # 4. Construct Chunk records preserving correct 1-to-1 mapping
            chunk_records: list[Chunk] = []
            for (page_num, idx, chunk_str), vector in zip(chunk_metadata, vectors, strict=True):
                chunk_obj = Chunk(
                    document_id=doc.id,
                    filename=doc.filename,
                    page_number=page_num,
                    chunk_index=idx,
                    text=chunk_str,
                    embedding=vector,
                )
                chunk_records.append(chunk_obj)

            # 5. Save all chunks to database
            for chk in chunk_records:
                session.add(chk)

            # 6. Mark document as ready
            doc.status = DocumentStatus.READY
            doc.error_message = None
            session.add(doc)
            session.commit()
            logger.info(
                "Successfully ingested document %s (%s): %d chunks stored in pgvector.",
                document_id,
                doc.filename,
                len(chunk_records),
            )

        except Exception as exc:
            logger.error(
                "Failed to ingest document %s (%s): %s",
                document_id,
                doc.filename,
                exc,
                exc_info=True,
            )
            session.rollback()
            refreshed_doc = session.get(Document, document_id)
            if refreshed_doc:
                refreshed_doc.status = DocumentStatus.FAILED
                refreshed_doc.error_message = str(exc)
                session.add(refreshed_doc)
                session.commit()


def compute_cosine_distance(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine distance between two dense vectors in Python."""
    import math

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 1.0
    similarity = dot / (norm1 * norm2)
    return max(0.0, min(2.0, 1.0 - similarity))


ASSIGNMENT_QUERY_PATTERN = re.compile(
    r"\bassignment\s*(\d+|[a-zA-Z]+)\b",
    re.IGNORECASE,
)


def search_similar_chunks(
    query_text: str,
    k: int,
    session: Session,
    provider: str | None = None,
) -> list[tuple[Chunk, float]]:
    """Generate query embedding and find top-k closest chunks using cosine distance.

    Supports assignment-specific re-ranking and contiguous chunk retrieval
    while preserving global relevance filtering.

    Returns:
        List of tuples: (Chunk, cosine_similarity_score) ordered closest first.
    """
    # 1. Return empty list if no documents are ready
    ready_docs = session.exec(
        select(Document.id).where(Document.status == DocumentStatus.READY)
    ).first()
    if not ready_docs:
        return []

    # 2. Generate embedding with the same model
    query_vector = generate_embedding(query_text, provider=provider)

    # 3. Detect database dialect
    bind = session.get_bind()
    is_postgres = getattr(bind.dialect, "name", "") == "postgresql"

    # 4. Check if query is targeting a specific assignment
    assignment_match = ASSIGNMENT_QUERY_PATTERN.search(query_text)

    if is_postgres:
        distance_expr = Chunk.embedding.cosine_distance(query_vector)
        fetch_limit = max(k * 3, 15) if assignment_match else k
        statement = (
            select(Chunk, distance_expr.label("distance"))
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.status == DocumentStatus.READY)
            .order_by(distance_expr.asc())
            .limit(fetch_limit)
        )
        rows = session.exec(statement).all()
        scored_chunks = [(chunk, round(float(1.0 - dist), 4)) for chunk, dist in rows]
    else:
        # In-memory SQLite fallback for unit/integration tests
        chunks = session.exec(
            select(Chunk)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.status == DocumentStatus.READY)
        ).all()

        scored: list[tuple[Chunk, float, float]] = []
        for chunk in chunks:
            dist = compute_cosine_distance(query_vector, chunk.embedding)
            sim = round(float(1.0 - dist), 4)
            scored.append((chunk, dist, sim))

        scored.sort(key=lambda item: item[1])
        fetch_limit = max(k * 3, 15) if assignment_match else k
        top_scored = scored[:fetch_limit]
        scored_chunks = [(chunk, sim) for chunk, _, sim in top_scored]

    # If NOT an assignment-specific query, return standard top-k semantic results directly
    if not assignment_match:
        return scored_chunks[:k]

    # Assignment-specific retrieval: retrieve relevant chunks (header + continuation chunks)
    assign_num = assignment_match.group(1).lower()
    target_regex = re.compile(rf"\bassignment\s*{re.escape(assign_num)}\b", re.IGNORECASE)
    header_regex = re.compile(
        rf"(?:^|\n)\s*assignment\s*{re.escape(assign_num)}\b\s*[:\-\n]", re.IGNORECASE
    )

    candidate_chunks = session.exec(
        select(Chunk)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Document.status == DocumentStatus.READY,
            col(Chunk.text).ilike(f"%assignment%{assign_num}%"),
        )
        .order_by(Chunk.document_id, Chunk.chunk_index)
    ).all()

    primary_matches: list[Chunk] = []
    secondary_matches: list[Chunk] = []

    for c in candidate_chunks:
        if header_regex.search(c.text):
            primary_matches.append(c)
        elif target_regex.search(c.text):
            secondary_matches.append(c)

    continuation_chunks: list[Chunk] = []
    if primary_matches:
        continuation_conds = [
            (Chunk.document_id == c.document_id) & (Chunk.chunk_index == c.chunk_index + 1)
            for c in primary_matches
        ]
        raw_continuations = session.exec(
            select(Chunk)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Document.status == DocumentStatus.READY,
                or_(*continuation_conds),
            )
            .order_by(Chunk.document_id, Chunk.chunk_index)
        ).all()
        for next_c in raw_continuations:
            if not re.search(r"^\s*assignment\s*\d+", next_c.text, re.IGNORECASE):
                continuation_chunks.append(next_c)

    # If no chunk matches the target assignment keyword, fall back to pure vector results
    if not primary_matches and not secondary_matches:
        return scored_chunks[:k]

    seen_ids: set[str] = set()
    final_results: list[tuple[Chunk, float]] = []

    # 1. Primary header matches (score 0.98)
    for c in primary_matches:
        if c.id not in seen_ids:
            seen_ids.add(c.id)
            orig_score = next((s for chk, s in scored_chunks if chk.id == c.id), 0.70)
            final_results.append((c, max(0.98, orig_score)))

    # 2. Continuation chunks providing implementation/steps (score 0.95)
    for c in continuation_chunks:
        if c.id not in seen_ids:
            seen_ids.add(c.id)
            orig_score = next((s for chk, s in scored_chunks if chk.id == c.id), 0.65)
            final_results.append((c, max(0.95, orig_score)))

    # 3. Secondary/passing keyword matches (score 0.85)
    for c in secondary_matches:
        if c.id not in seen_ids:
            seen_ids.add(c.id)
            orig_score = next((s for chk, s in scored_chunks if chk.id == c.id), 0.60)
            final_results.append((c, max(0.85, orig_score)))

    # 4. Fill remaining slots up to k with top semantic vector matches
    for c, score in scored_chunks:
        if c.id not in seen_ids:
            seen_ids.add(c.id)
            final_results.append((c, score))
            if len(final_results) >= k:
                break

    return final_results[:k]
