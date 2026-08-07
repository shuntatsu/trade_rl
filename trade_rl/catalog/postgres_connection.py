"""Optional PostgreSQL connection construction shared by catalog adapters."""

from __future__ import annotations

from typing import Any


def _import_psycopg() -> Any | None:
    try:
        import psycopg
    except ImportError:
        return None
    return psycopg


def default_connection_factory(database_url: str) -> Any:
    """Open one PostgreSQL connection or report the missing optional dependency."""

    psycopg = _import_psycopg()
    if psycopg is None:
        raise RuntimeError(
            "PostgreSQL catalog support requires the 'postgres' optional dependency"
        )
    return psycopg.connect(database_url)


__all__ = ["default_connection_factory"]
