"""add document status and chunk table with pgvector

Revision ID: a9b8c7d6e5f4
Revises: 37325e88fe95
Create Date: 2026-11-16 01:45:00.000000

"""

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
import sqlmodel

# revision identifiers, used by Alembic.
revision: str = "a9b8c7d6e5f4"
down_revision: str | Sequence[str] | None = "37325e88fe95"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: enable pgvector, add document status, and create chunk table."""
    # 1. Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Add status, error_message, file_path to document table
    op.add_column(
        "document",
        sa.Column(
            "status",
            sqlmodel.sql.sqltypes.AutoString(),
            server_default="pending",
            nullable=False,
        ),
    )
    op.add_column(
        "document",
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.add_column(
        "document",
        sa.Column(
            "file_path",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
        ),
    )
    op.create_index(
        op.f("ix_document_status"), "document", ["status"], unique=False
    )

    # 3. Create chunk table
    op.create_table(
        "chunk",
        sa.Column("id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("document_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("filename", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column(
            "created_at",
            sqlmodel.sql.sqltypes.UTCDateTime(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["document.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chunk_id"), "chunk", ["id"], unique=False)
    op.create_index(
        op.f("ix_chunk_document_id"), "chunk", ["document_id"], unique=False
    )
    op.create_index(
        op.f("ix_chunk_filename"), "chunk", ["filename"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_chunk_filename"), table_name="chunk")
    op.drop_index(op.f("ix_chunk_document_id"), table_name="chunk")
    op.drop_index(op.f("ix_chunk_id"), table_name="chunk")
    op.drop_table("chunk")

    op.drop_index(op.f("ix_document_status"), table_name="document")
    op.drop_column("document", "file_path")
    op.drop_column("document", "error_message")
    op.drop_column("document", "status")

    op.execute("DROP EXTENSION IF EXISTS vector")
