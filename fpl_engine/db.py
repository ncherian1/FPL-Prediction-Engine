from __future__ import annotations

import json
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import psycopg
from psycopg.types.json import Jsonb

from .config import ROOT, get_settings

SQL_DIR = ROOT / "sql"


def connect() -> psycopg.Connection:
    # autocommit: snapshot/run bookkeeping must persist even if a data write fails.
    # Data writes use explicit `with conn.transaction():` blocks.
    # prepare_threshold=None: Supabase's transaction pooler (port 6543) cannot use
    # server-side prepared statements; harmless on direct / session connections.
    return psycopg.connect(get_settings().database_url, autocommit=True, prepare_threshold=None)


def migrate(conn: psycopg.Connection) -> list[str]:
    conn.execute(
        "CREATE SCHEMA IF NOT EXISTS ops; CREATE TABLE IF NOT EXISTS ops.schema_migrations "
        "(version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    applied = {r[0] for r in conn.execute("SELECT version FROM ops.schema_migrations")}
    ran = []
    for path in sorted(Path(SQL_DIR).glob("*.sql")):
        if path.stem in applied:
            continue
        with conn.transaction():
            conn.execute(path.read_text())
            conn.execute("INSERT INTO ops.schema_migrations (version) VALUES (%s)", (path.stem,))
        ran.append(path.stem)
    return ran


def _adapt(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return Jsonb(v)
    return v


def upsert(
    conn: psycopg.Connection,
    table: str,
    rows: Sequence[dict[str, Any]],
    conflict: Sequence[str],
    update: Iterable[str] | None = None,
    touch_updated_at: bool = False,
) -> int:
    """INSERT ... ON CONFLICT DO UPDATE for a batch of homogeneous dict rows.

    update=None updates every non-conflict column; update=[] means DO NOTHING.
    """
    if not rows:
        return 0
    cols = list(rows[0].keys())
    upd = [c for c in cols if c not in conflict] if update is None else list(update)
    col_sql = ", ".join(cols)
    ph = ", ".join(["%s"] * len(cols))
    if upd:
        sets = [f"{c} = EXCLUDED.{c}" for c in upd]
        if touch_updated_at:
            sets.append("updated_at = now()")
        action = "DO UPDATE SET " + ", ".join(sets)
    else:
        action = "DO NOTHING"
    sql = f"INSERT INTO {table} ({col_sql}) VALUES ({ph}) ON CONFLICT ({', '.join(conflict)}) {action}"
    with conn.cursor() as cur:
        cur.executemany(sql, [tuple(_adapt(r[c]) for c in cols) for r in rows])
    return len(rows)


def insert(conn: psycopg.Connection, table: str, rows: Sequence[dict[str, Any]]) -> int:
    if not rows:
        return 0
    cols = list(rows[0].keys())
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})"
    with conn.cursor() as cur:
        cur.executemany(sql, [tuple(_adapt(r[c]) for c in cols) for r in rows])
    return len(rows)


class RunTracker:
    def __init__(self) -> None:
        self.rows = 0
        self.details: dict[str, Any] = {}

    def add(self, key: str, n: int) -> None:
        self.rows += n
        self.details[key] = self.details.get(key, 0) + n


@contextmanager
def track_run(conn: psycopg.Connection, job: str) -> Iterator[RunTracker]:
    """Record a job in ops.ingestion_runs (last-successful-update tracking)."""
    run_id = conn.execute(
        "INSERT INTO ops.ingestion_runs (job) VALUES (%s) RETURNING id", (job,)
    ).fetchone()[0]
    tracker = RunTracker()
    try:
        yield tracker
    except BaseException as exc:
        conn.execute(
            "UPDATE ops.ingestion_runs SET finished_at = now(), status = 'failed', rows_written = %s, "
            "details = %s, error_message = %s WHERE id = %s",
            (tracker.rows, Jsonb(tracker.details), f"{exc!r}\n{traceback.format_exc()[-4000:]}", run_id),
        )
        raise
    conn.execute(
        "UPDATE ops.ingestion_runs SET finished_at = now(), status = 'success', rows_written = %s, "
        "details = %s WHERE id = %s",
        (tracker.rows, Jsonb(tracker.details), run_id),
    )


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
