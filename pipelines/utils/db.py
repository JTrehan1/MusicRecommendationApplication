"""
utils/db.py — Postgres connection pool (works for both Supabase dev and local prod).
Uses psycopg2 with a simple connection-pool wrapper.
"""

import os
import logging
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values, RealDictCursor
from dotenv import load_dotenv

load_dotenv("../credentials/.env")  # Load env vars from .env.local (if it exists)
log = logging.getLogger(__name__)


def _build_dsn() -> str:
    """Build a Postgres DSN from environment variables."""
    return (
        f"host={os.environ['SUPABASE_HOST']} "
        f"port={os.environ.get('SUPABASE_PORT', 6543)} "
        f"dbname={os.environ.get('SUPABASE_DB', 'postgres')} "
        f"user={os.environ['SUPABASE_USER']} "
        f"password={os.environ['SUPABASE_PASSWORD']} "
        f"sslmode=require"  
    )


# Module-level pool; initialised once on first import.
_pool: pool.ThreadedConnectionPool | None = None


def get_pool(minconn: int = 1, maxconn: int = 10) -> pool.ThreadedConnectionPool:
    """
    Get the global Postgres connection pool, creating it if necessary.
    
    :param minconn: Minimum number of connections in the pool.
    :param maxconn: Maximum number of connections in the pool.
    """
    
    global _pool
    if _pool is None:
        _pool = pool.ThreadedConnectionPool(minconn, maxconn, dsn=_build_dsn())
        log.info("Postgres connection pool initialised (max=%d)", maxconn)
    return _pool


@contextmanager
def get_conn():
    """Yield a connection from the pool; auto-commit/rollback on exit."""
    p = get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


@contextmanager
def get_cursor(dict_cursor: bool = False):
    """
    Yield a cursor (optionally RealDict) within a managed connection.
    
    :param dict_cursor: If True, use RealDictCursor to return rows as dicts.
    """
    factory = RealDictCursor if dict_cursor else None
    with get_conn() as conn:
        with conn.cursor(cursor_factory=factory) as cur:
            yield cur


def bulk_upsert(table: str, rows: list[dict], conflict_col: str, update_cols: list[str]) -> int:
    """
    Perform a bulk upsert (insert or update on conflict) into the specified table.
    
    :param table: Name of the table to upsert into.
    :param rows: List of dictionaries representing rows to upsert.
    :param conflict_col: Column(s) to check for conflicts.
    :param update_cols: Columns to update on conflict.
    :return: Number of rows affected.
    """
    if not rows:
        return 0

    cols = list(rows[0].keys())
    values = [[r[c] for c in cols] for r in rows]

    # If no update cols, use DO NOTHING instead of DO UPDATE SET
    if update_cols:
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        conflict_clause = f"ON CONFLICT ({conflict_col}) DO UPDATE SET {set_clause}"
    else:
        conflict_clause = f"ON CONFLICT ({conflict_col}) DO NOTHING"

    sql = f"""
        INSERT INTO {table} ({', '.join(cols)})
        VALUES %s
        {conflict_clause}
    """
    with get_cursor() as cur:
        execute_values(cur, sql, values, page_size=500)
        return cur.rowcount