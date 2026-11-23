"""Data freshness and missing-field reports (milestone definition of done)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg

from .config import ROOT
from .transform import MATCH_STAT_FIELDS

REPORT_DIR = ROOT / "data" / "reports"

CONSOLIDATED_SQL = """
WITH cur AS (SELECT season FROM core.seasons WHERE is_current),
pms AS (
  SELECT s.player_id,
         count(*)                                   AS fixtures_recorded,
         sum(s.minutes)                             AS minutes,
         sum(s.fpl_points)                          AS points,
         count(s.expected_goals)                    AS rows_with_xg,
         count(s.defensive_contribution)            AS rows_with_defcon,
         max(s.observed_at)                         AS match_stats_observed_at
  FROM core.player_match_stats s JOIN cur USING (season)
  GROUP BY s.player_id
),
px AS (
  SELECT DISTINCT ON (player_id) player_id, price, ownership, observed_at
  FROM core.player_price_history ORDER BY player_id, observed_at DESC
)
SELECT p.player_id, ps.fpl_element_id, p.web_name, p.first_name, p.second_name, ps.position,
       t.short_name AS team, p.date_of_birth, p.nationality,
       px.price, px.ownership, px.observed_at AS price_observed_at,
       a.injury_status, a.chance_of_playing_next_round, a.news, a.observed_at AS availability_observed_at,
       pms.fixtures_recorded, pms.minutes, pms.points, pms.rows_with_xg, pms.rows_with_defcon,
       pms.match_stats_observed_at,
       (SELECT count(*) FROM core.player_match_stats h WHERE h.player_id = p.player_id
          AND h.season <> (SELECT season FROM cur)) AS historical_fixture_rows,
       (SELECT array_agg(DISTINCT source_name) FROM core.player_source_ids s
          WHERE s.canonical_player_id = p.player_id) AS linked_sources
FROM core.player_seasons ps
JOIN cur USING (season)
JOIN core.players p USING (player_id)
LEFT JOIN core.teams t ON t.team_id = ps.team_id
LEFT JOIN px ON px.player_id = p.player_id
LEFT JOIN core.v_latest_availability a ON a.player_id = p.player_id AND a.source = 'fpl_api'
LEFT JOIN pms ON pms.player_id = p.player_id
ORDER BY t.short_name, ps.position, p.web_name
"""


def freshness(conn: psycopg.Connection) -> pd.DataFrame:
    rows = conn.execute("""
        SELECT job,
               max(finished_at) FILTER (WHERE status = 'success') AS last_success,
               max(finished_at) FILTER (WHERE status = 'failed')  AS last_failure,
               (array_agg(rows_written ORDER BY started_at DESC) FILTER (WHERE status='success'))[1] AS last_rows,
               (array_agg(error_message ORDER BY started_at DESC) FILTER (WHERE status='failed'))[1] AS last_error
        FROM ops.ingestion_runs GROUP BY job ORDER BY job""").fetchall()
    df = pd.DataFrame(rows, columns=["job", "last_success", "last_failure", "last_rows", "last_error"])
    now = datetime.now(timezone.utc)
    df["age_hours"] = df["last_success"].map(lambda t: round((now - t).total_seconds() / 3600, 1) if t else None)
    df["last_error"] = df["last_error"].map(lambda e: (e or "").split("\n")[0][:120] or None)
    return df


def field_coverage(conn: psycopg.Connection) -> pd.DataFrame:
    """Share of player_match_stats rows with a non-NULL value, per season and column."""
    cols = list(MATCH_STAT_FIELDS) + ["price"]
    exprs = ", ".join(f"round(100.0 * count({c}) / count(*), 1) AS {c}" for c in cols)
    rows = conn.execute(f"SELECT season, count(*) AS n_rows, {exprs} FROM core.player_match_stats "
                        f"GROUP BY season ORDER BY season").fetchall()
    return pd.DataFrame(rows, columns=["season", "n_rows", *cols])


def consolidated_players(conn: psycopg.Connection) -> pd.DataFrame:
    cur = conn.execute(CONSOLIDATED_SQL)
    return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def provider_coverage(conn: psycopg.Connection, source: str) -> pd.DataFrame:
    rows = conn.execute("""SELECT entity, season, count(*), max(observed_at)
                           FROM raw.provider_records WHERE source_name = %s
                           GROUP BY entity, season ORDER BY season, entity""", (source,)).fetchall()
    return pd.DataFrame(rows, columns=["entity", "season", "records", "last_observed"])


def write_all(conn: psycopg.Connection) -> dict[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, df in {
        "freshness": freshness(conn),
        "field_coverage": field_coverage(conn),
        "players_consolidated": consolidated_players(conn),
    }.items():
        path = REPORT_DIR / f"{name}_{stamp}.csv"
        df.to_csv(path, index=False)
        out[name] = path
    return out
