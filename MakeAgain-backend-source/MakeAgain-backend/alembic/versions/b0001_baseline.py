"""B07 baseline: bring an empty or legacy SQLite database to the current schema.

The upgrade deliberately reuses the existing, already-battle-tested idempotent chain
instead of rewriting it as Alembic DDL:

    Base.metadata.create_all(engine)   # create new tables
    _migrate_sqlite()                  # add columns to existing tables
    migrations_b07.migrate(...)        # business backfill (companions, scope, diary,
                                       # membership) + schema_migrations ledger
    seed_welcome_letter(...)           # first-run seed

Existing rows and original text/IDs are preserved by that chain; see
``memory/migrations_b07.py`` for its contract.

Downgrade is intentionally a no-op: SQLite downgrades here would be destructive
(dropping tables/columns and the business backfill), so they are never run
automatically.  Use a database backup/restore instead.
"""
from __future__ import annotations

from alembic import op  # noqa: F401  -- standard Alembic import, kept for convention

revision = "b0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    from memory import b07_models  # noqa: F401  -- register B07 tables
    from memory import db as db_module
    from memory import models  # noqa: F401  -- register ORM tables
    from memory.migrations_b07 import migrate
    from memory.seed import seed_welcome_letter

    db_module.Base.metadata.create_all(db_module.engine)
    db_module._migrate_sqlite()
    # Business backfill stays as custom Python inside the revision (not rewritten).
    migrate(db_module.engine, db_module._RawSessionLocal)
    # Same first-run seed the legacy init_db() performs.
    try:
        with db_module._RawSessionLocal() as seed_db:
            seed_welcome_letter(seed_db)
            seed_db.commit()
    except Exception:
        # Match legacy init_db(): a seed failure must not fail initialization.
        pass


def downgrade() -> None:
    # Destructive SQLite downgrade is deliberately not automated.
    return None
