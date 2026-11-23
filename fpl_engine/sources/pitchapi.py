"""PitchAPI (https://pitchapi.dev) — supplementary advanced stats: shots/xG, player match
stats, lineups, team stats, PPDA / field tilt. Free, key required (PITCHAPI_API_KEY).
History back to 2021.

Field coverage is not yet validated, so responses are landed in raw.provider_records
as-is. Run `coverage pitchapi` after a sync to see which fields are populated.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterator

import psycopg

from .. import db
from ..config import get_settings
from ..http import FetchError, Fetcher

log = logging.getLogger(__name__)
SOURCE = "pitchapi"
BASE_URL = "https://api.pitchapi.dev/v1"
# Per finished match: entity name -> path suffix
MATCH_ENDPOINTS = {
    "match_players": "players",
    "match_shots": "shots",
    "match_lineups": "lineups",
    "match_stats": "stats",
    "match_advanced": "advanced",
    "match_advanced_players": "advanced/players",
}
FINISHED = {"finished", "played", "ft", "full_time", "complete", "completed"}


def fetcher(conn: psycopg.Connection) -> Fetcher:
    key = get_settings().pitchapi_key
    if not key:
        raise RuntimeError("PITCHAPI_API_KEY is not set (free key: https://pitchapi.dev)")
    return Fetcher(conn, SOURCE, BASE_URL, headers={"X-API-KEY": key}, min_interval=0.25)


def to_provider_season(season: str) -> str:
    """'2026-27' -> '2026/2027'"""
    y = int(season[:4])
    return f"{y}/{y + 1}"


def _data(resp_json: Any) -> Any:
    if isinstance(resp_json, dict) and "error" in resp_json:
        raise FetchError(f"PitchAPI error: {resp_json['error']}")
    return resp_json.get("data", resp_json) if isinstance(resp_json, dict) else resp_json


def _items(data: Any, key: str) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in (key, "items", "results"):
            if isinstance(data.get(k), list):
                return data[k]
    return []


def find_premier_league(f: Fetcher) -> dict:
    leagues = _items(_data(f.get("/leagues").json()), "leagues")
    for lg in leagues:
        if lg.get("name") == "Premier League" and lg.get("country_code") in ("ENG", "GB", "GB-ENG", None):
            return lg
    raise RuntimeError(f"Premier League not found among {len(leagues)} PitchAPI leagues")


def iter_matches(f: Fetcher, league_id: str, season: str) -> Iterator[tuple[dict, int]]:
    params: dict[str, Any] = {"season": to_provider_season(season), "status": "all"}
    while True:
        resp = f.get(f"/leagues/{league_id}/matches", params=params)
        data = _data(resp.json())
        for m in _items(data, "matches"):
            yield m, resp.snapshot_id
        cursor = data.get("next_cursor") if isinstance(data, dict) else None
        if not cursor:
            break
        params = {**params, "cursor": cursor}


def is_finished(m: dict) -> bool:
    return str(m.get("status", "")).lower() in FINISHED


def sync_season(conn: psycopg.Connection, season: str, refresh: bool = False,
                limit: int | None = None) -> int:
    with db.track_run(conn, f"pitchapi:{season}") as run, fetcher(conn) as f:
        league = find_premier_league(f)
        observed = datetime.now(timezone.utc)
        matches = list(iter_matches(f, league["id"], season))
        rows = [{"source_name": SOURCE, "entity": "match", "entity_id": m["id"], "season": season,
                 "parent_id": league["id"], "payload": m, "observed_at": observed, "snapshot_id": sid}
                for m, sid in matches]
        with conn.transaction():
            run.add("match", db.upsert(conn, "raw.provider_records", rows, ["source_name", "entity", "entity_id"]))

        have = {(r[0], r[1]) for r in conn.execute(
            "SELECT entity, entity_id FROM raw.provider_records WHERE source_name = %s AND season = %s",
            (SOURCE, season)).fetchall()}
        todo = [m for m, _ in matches if is_finished(m)]
        if limit:
            todo = todo[:limit]
        log.info("pitchapi %s: %d matches, %d finished to detail", season, len(matches), len(todo))
        for i, m in enumerate(todo, 1):
            for entity, suffix in MATCH_ENDPOINTS.items():
                if not refresh and (entity, m["id"]) in have:
                    continue
                try:
                    resp = f.get(f"/matches/{m['id']}/{suffix}")
                except FetchError as exc:
                    if exc.status == 404:  # e.g. ANALYTICS_UNAVAILABLE: match exists, not rated
                        run.add(f"{entity}_unavailable", 1)
                        continue
                    raise
                with conn.transaction():
                    run.add(entity, db.upsert(conn, "raw.provider_records", [{
                        "source_name": SOURCE, "entity": entity, "entity_id": m["id"], "season": season,
                        "parent_id": league["id"], "payload": _data(resp.json()),
                        "observed_at": datetime.now(timezone.utc), "snapshot_id": resp.snapshot_id,
                    }], ["source_name", "entity", "entity_id"]))
            if i % 25 == 0:
                log.info("pitchapi %s: %d/%d matches detailed", season, i, len(todo))
        return run.rows
