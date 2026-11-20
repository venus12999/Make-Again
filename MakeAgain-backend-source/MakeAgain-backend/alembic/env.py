"""Alembic environment for Make Again backend.

Design notes (PR-A):
- The migration engine is *the* engine from ``memory.db`` (same URL / PRAGMAs / pool),
  whose URL is read from ``config.get_settings().database_url`` (DATABASE_URL env var).
- The baseline revision deliberately delegates to the existing idempotent chain
  (create_all + _migrate_sqlite + migrations_b07.migrate + welcome-letter seed) instead
  of re-implementing the schema in DDL, so legacy first-start/upgrade behaviour is kept.
- The connection used by Alembic itself is put in AUTOCOMMIT isolation.  The baseline
  revision opens its own engine connections (and ORM sessions) to run the legacy chain;
  with AUTOCOMMIT there is no stale read snapshot on Alembic's connection and therefore
  no SQLite "database is locked" / BUSY_SNAPSHOT conflict when Alembic writes
  ``alembic_version`` at the end of the run.
"""
from __future__ import annotations

from logging.config import fileConfig

from alembic import context

from config import get_settings
from memory import db as db_module
from memory import models, b07_models  # noqa: F401  -- populate Base.metadata

config = context.config

if config.config_file_name:
    try:
        fileConfig(config.config_file_name)
    except Exception:
        # Logging config is cosmetic; never let it block a migration.
        pass

# Always use the application's own DATABASE_URL, not the ini placeholder.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = db_module.Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout (no DB connection). Kept for CLI completeness."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = db_module.engine
    with connectable.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=True,
            # The legacy chain writes through its own connections; do not let Alembic
            # hold an open DDL transaction around it.
            transactional_ddl=False,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
