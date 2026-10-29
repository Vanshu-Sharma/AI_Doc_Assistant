import logging
import os
import socket
import time
from urllib.parse import urlparse

from dotenv import load_dotenv
from sqlmodel import Session, SQLModel, create_engine

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


def get_resolved_database_url() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:///./doc_assistant.db")

    # SQLAlchemy 2.0 requires postgresql:// instead of postgres://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # If the URL specifies host 'db' (Docker service name),
    # but 'db' cannot be resolved (i.e. running outside Docker on the host),
    # fallback to localhost so running locally works seamlessly.
    try:
        parsed = urlparse(url)
        if parsed.hostname == "db":
            try:
                socket.gethostbyname("db")
            except socket.gaierror:
                netloc = parsed.netloc.replace("@db", "@localhost")
                url = parsed._replace(netloc=netloc).geturl()
    except Exception:
        pass

    return url


DATABASE_URL = get_resolved_database_url()

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)


def init_db(max_retries: int = 5, retry_delay: float = 2.0) -> None:
    """Initialize database tables with retry logic for container startup."""
    for attempt in range(1, max_retries + 1):
        try:
            SQLModel.metadata.create_all(engine)
            logger.info("Database tables verified/created successfully.")
            return
        except Exception as exc:
            if attempt == max_retries:
                logger.error(
                    "Could not connect to database after %d attempts: %s",
                    max_retries,
                    exc,
                )
                raise
            logger.warning(
                "Database connection attempt %d failed: %s. Retrying in %.1fs...",
                attempt,
                exc,
                retry_delay,
            )
            time.sleep(retry_delay)


def get_session():
    """FastAPI dependency for database sessions."""
    with Session(engine) as session:
        yield session
