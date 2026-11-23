"""StrikerAPI (https://strikerapi.com) — supplementary historical shot-level xG, match
events and odds. Free tier: 500 credits / month (key: STRIKERAPI_API_KEY).

Every call costs credits, so jobs here are opt-in and stop before the monthly allowance
drops below `reserve`. Responses land in raw.provider_records as-is. Seasons are given by
start year (2026 = 2026-27). License: storing and modelling is allowed; redistributing the
raw data is not.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import psycopg

from .. import db
from ..config import get_settings
from ..http import Fetcher, Response
from ..transform import season_label

log = logging.getLogger(__name__)
SOURCE = "strikerapi"
BASE_URL = "https://mcp.strikerapi.com/v1"
COST = {"list_teams": 1, "list_games": 2, "get_shot_map": 10, "get_events": 10, "get_lineups": 5}


class CreditBudgetExceeded(RuntimeError):
    pass


class StrikerClient:
    def __init__(self, conn: psycopg.Connection, reserve: int = 50):
        key = get_settings().strikerapi_key
        if not key:
            raise RuntimeError("STRIKERAPI_API_KEY is not set (free key: https://strikerapi.com)")
        self.f = Fetcher(conn, SOURCE, BASE_URL, headers={"x-api-key": key}, min_interval=1.0)
        self.reserve = reserve
        self.remaining: int | None = None
        self.used = 0

    def call(self, tool: str, **params: Any) -> Response:
        cost = COST.get(tool, 10)
        if self.remaining is not None and self.remaining - cost < self.reserve:
            raise CreditBudgetExceeded(f"{tool} would leave {self.remaining - cost} credits (< reserve {self.reserve})")
        resp = self.f.post(f"/{tool}", params)
        rem = resp.headers.get("x-credits-remaining")
        if rem is not None and rem.lstrip("-").isdigit():
            self.remaining = int(rem)
        used = resp.headers.get("x-credits-used")
        self.used += int(used) if used and used.isdigit() else cost
        return resp

    def close(self) -> None:
        self.f.close()


def _store(conn, entity: str, entity_id: str, season: str | None, resp: Response) -> int:
    with conn.transaction():
        return db.upsert(conn, "raw.provider_records", [{
            "source_name": SOURCE, "entity": entity, "entity_id": entity_id, "season": season,
            "parent_id": None, "payload": resp.json(), "observed_at": datetime.now(timezone.utc),
            "snapshot_id": resp.snapshot_id,
        }], ["source_name", "entity", "entity_id"])


def probe(conn: psycopg.Connection, start_year: int) -> dict:
    """Cheap coverage check (~3 credits): teams + fixtures/results for one season."""
    season = season_label(start_year)
    client = StrikerClient(conn)
    try:
        with db.track_run(conn, f"strikerapi_probe:{season}") as run:
            run.add("teams", _store(conn, "teams", f"EPL:{start_year}", season,
                                    client.call("list_teams", season=start_year)))
            run.add("games", _store(conn, "games", f"EPL:{start_year}", season,
                                    client.call("list_games", season=start_year)))
            run.details.update(credits_used=client.used, credits_remaining=client.remaining)
            return run.details
    finally:
        client.close()


def sync_shot_maps(conn: psycopg.Connection, start_year: int, teams: list[str] | None = None,
                   reserve: int = 50) -> dict:
    """Shot map (every located shot with xG) per club for a season — 10 credits per club.
    With no teams given, the season's clubs are looked up first (1 credit)."""
    season = season_label(start_year)
    client = StrikerClient(conn, reserve=reserve)
    try:
        with db.track_run(conn, f"strikerapi_shots:{season}") as run:
            if not teams:
                resp = client.call("list_teams", season=start_year)
                _store(conn, "teams", f"EPL:{start_year}", season, resp)
                teams = [t["abbrev"] for t in resp.json()["data"]["teams"]]
            for team in teams:
                key = f"EPL:{start_year}:{team}"
                if conn.execute("SELECT 1 FROM raw.provider_records WHERE source_name=%s AND entity='shot_map' "
                                "AND entity_id=%s", (SOURCE, key)).fetchone():
                    continue
                try:
                    resp = client.call("get_shot_map", team=team, season=start_year, include_shots=True)
                except CreditBudgetExceeded as exc:
                    log.warning("stopping: %s", exc)
                    run.details["stopped"] = str(exc)
                    break
                run.add("shot_map", _store(conn, "shot_map", key, season, resp))
            run.details.update(credits_used=client.used, credits_remaining=client.remaining)
            return run.details
    finally:
        client.close()
