"""Football-Data.co.uk — historical Premier League match results, team stats and odds.

Team names are stored as published by the provider (e.g. "Man United"); mapping to
canonical team IDs happens in the reconciliation layer.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg

from .. import db
from ..http import Fetcher
from ..transform import is_missing, season_label, to_float, to_int

log = logging.getLogger(__name__)
SOURCE = "football_data"
BASE_URL = "https://www.football-data.co.uk/mmz4281"

TYPED = {
    "fthg": ("FTHG", to_int), "ftag": ("FTAG", to_int), "ftr": ("FTR", str),
    "hthg": ("HTHG", to_int), "htag": ("HTAG", to_int),
    "home_shots": ("HS", to_int), "away_shots": ("AS", to_int),
    "home_shots_on_target": ("HST", to_int), "away_shots_on_target": ("AST", to_int),
    "home_fouls": ("HF", to_int), "away_fouls": ("AF", to_int),
    "home_corners": ("HC", to_int), "away_corners": ("AC", to_int),
    "home_yellow": ("HY", to_int), "away_yellow": ("AY", to_int),
    "home_red": ("HR", to_int), "away_red": ("AR", to_int),
    "home_xg": ("HxG", to_float), "away_xg": ("AxG", to_float),
}


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def parse_date(s: str):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def parse_rows(text: str, season: str, division: str, observed: datetime, snapshot_id: int) -> list[dict]:
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, on_bad_lines="skip")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    rows = []
    for rec in df.to_dict("records"):
        d = parse_date(rec.get("Date", ""))
        if d is None or is_missing(rec.get("HomeTeam")):
            continue
        row = {"season": season, "division": division, "match_date": d,
               "home_team": rec["HomeTeam"].strip(), "away_team": rec["AwayTeam"].strip()}
        for col, (src, conv) in TYPED.items():
            v = rec.get(src)
            row[col] = None if is_missing(v) else conv(v)
        row["data"] = {k: v for k, v in rec.items() if k and not is_missing(v)}
        row["observed_at"] = observed
        row["snapshot_id"] = snapshot_id
        rows.append(row)
    return rows


def import_season(conn: psycopg.Connection, start_year: int, division: str = "E0",
                  is_current: bool = False) -> int:
    season = season_label(start_year)
    with db.track_run(conn, f"football_data:{season}") as run, \
            Fetcher(conn, SOURCE, BASE_URL, min_interval=1.0) as f:
        ttl = None if is_current else timedelta(days=30)
        resp = f.get(f"{season_code(start_year)}/{division}.csv", cache_ttl=ttl)
        rows = parse_rows(resp.text(), season, division, datetime.now(timezone.utc), resp.snapshot_id)
        with conn.transaction():
            run.add("matches", db.upsert(conn, "raw.football_data_matches", rows,
                                         ["season", "division", "match_date", "home_team", "away_team"]))
        return run.rows
