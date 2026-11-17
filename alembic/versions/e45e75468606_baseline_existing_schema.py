
"""Baseline existing database schema."""

from typing import Optional, Sequence, Union

revision: str = "e45e75468606"
down_revision: Union[str, None] = None
branch_labels: Optional[Sequence[str]] = None
depends_on: Optional[Sequence[str]] = None


def upgrade() -> None:
    """Baseline: existing tables are already present."""
    pass


def downgrade() -> None:
    """No operation for the baseline migration."""
    pass
