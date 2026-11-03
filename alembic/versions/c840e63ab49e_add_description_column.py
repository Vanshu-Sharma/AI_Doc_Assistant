
"""Add description column to document table."""

from alembic import op
import sqlalchemy as sa

# Revision identifiers
revision = "c840e63ab49e"
down_revision = "e45e75468606"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add description column."""
    op.add_column(
        "document",
        sa.Column("description", sa.String(), nullable=True),
    )


def downgrade() -> None:
    """Remove description column if it exists."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    columns = [
        column["name"]
        for column in inspector.get_columns("document")
    ]

    if "description" in columns:
        op.drop_column("document", "description")
