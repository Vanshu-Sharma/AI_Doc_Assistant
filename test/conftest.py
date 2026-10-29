import os
import pytest

# Ensure tests use an isolated SQLite database file so PostgreSQL is not required
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

from app.database import engine, init_db
from sqlmodel import SQLModel


@pytest.fixture(autouse=True, scope="session")
def setup_test_db():
    init_db()
    yield
    # Cleanup test database file after test session
    if os.path.exists("./test.db"):
        try:
            os.remove("./test.db")
        except OSError:
            pass
