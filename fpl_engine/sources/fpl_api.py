"""Official Fantasy Premier League API — authoritative source for FPL-specific data.

Endpoints: /bootstrap-static/, /fixtures/, /element-summary/{id}/, /event/{gw}/live/
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Any

import psycopg

from .. import db
from ..http import Fetcher
from ..transform import (
    FPL_STATUS,
    availability_hash,
    match_stats,
    price,
    season_label,
    to_bool,
    to_float,
    to_int,
    to_ts,
)

log = logging.getLogger(__name__)
SOURCE = "fpl_api"
BASE_URL = "https://fantasy.premierleague.com/api"


def fetcher(conn: psycopg.Connection) -> Fetcher:
    return Fetcher(conn, SOURCE, BASE_URL, min_interval=0.5)


# ---------------------------------------------------------------------------
# lookups
# ---------------------------------------------------------------------------
def team_map(conn: psycopg.Connection, season: str) -> dict[int, int]:
    return dict(conn.execute(
        "SELECT fpl_team_id, team_id FROM core.team_seasons WHERE season = %s", (season,)
    ).fetchall())


def element_map(conn: psycopg.Connection, season: str) -> dict[int, int]:
    return dict(conn.execute(
        "SELECT fpl_element_id, player_id FROM core.player_seasons WHERE season = %s", (season,)
    ).fetchall())


def fixture_map(conn: psycopg.Connection, season: str) -> dict[int, tuple]:
    rows = conn.execute(
        "SELECT fpl_fixture_id, fixture_id, home_team_id, away_team_id, gameweek, kickoff_time "
        "FROM core.fixtures WHERE season = %s", (season,)
    ).fetchall()
    return {r[0]: r[1:] for r in rows}


def current_season(conn: psycopg.Connection) -> str | None:
    row = conn.execute("SELECT season FROM core.seasons WHERE is_current").fetchone()
    return row[0] if row else None


def season_from_bootstrap(data: dict) -> str:
    first_deadline = to_ts(data["events"][0]["deadline_time"])
    return season_label(first_deadline.year)


_RETURN_RE = re.compile(r"Expected back (\d{1,2}) (\w{3})", re.I)


def parse_expected_return(news: str | None, observed: datetime) -> date | None:
    """FPL news like 'Hamstring injury - Expected back 30 Nov' -> date (next occurrence)."""
    if not news:
        return None
    m = _RETURN_RE.search(news)
    if not m:
        return None
    try:
        d = datetime.strptime(f"{m.group(1)} {m.group(2)[:3].title()} {observed.year}", "%d %b %Y").date()
    except ValueError:
        return None
    if d < observed.date() - (date(2000, 3, 1) - date(2000, 1, 1)):  # >~2 months in the past => next year
        d = d.replace(year=d.year + 1)
    return d


# ---------------------------------------------------------------------------
# transforms (pure)
# ---------------------------------------------------------------------------
def bootstrap_rows(data: dict, observed: datetime, snapshot_id: int) -> dict[str, list[dict]]:
    season = season_from_bootstrap(data)
    positions = {et["id"]: et["singular_name_short"] for et in data["element_types"]}
    team_code = {t["id"]: t["code"] for t in data["teams"]}

    teams = [{"team_id": t["code"], "team_name": t["name"], "short_name": t["short_name"]} for t in data["teams"]]
    team_seasons = [{
        "season": season, "fpl_team_id": t["id"], "team_id": t["code"],
        "strength": t.get("strength"),
        "strength_overall_home": t.get("strength_overall_home"),
        "strength_overall_away": t.get("strength_overall_away"),
        "strength_attack_home": t.get("strength_attack_home"),
        "strength_attack_away": t.get("strength_attack_away"),
        "strength_defence_home": t.get("strength_defence_home"),
        "strength_defence_away": t.get("strength_defence_away"),
        "source": SOURCE,
    } for t in data["teams"]]
    gameweeks = [{
        "season": season, "gameweek": e["id"], "name": e.get("name"),
        "deadline_time": to_ts(e.get("deadline_time")), "finished": e.get("finished"),
        "data_checked": e.get("data_checked"), "is_current": e.get("is_current"),
        "is_next": e.get("is_next"), "average_score": e.get("average_entry_score"),
        "highest_score": e.get("highest_score"), "source": SOURCE,
    } for e in data["events"]]

    players, player_seasons, source_ids, prices, availability = [], [], [], [], []
    for el in data["elements"]:
        pid = el["code"]
        team_id = team_code.get(el["team"])
        pos = positions.get(el["element_type"])
        players.append({
            "player_id": pid, "first_name": el.get("first_name"), "second_name": el.get("second_name"),
            "web_name": el.get("web_name"), "position": pos, "team_id": team_id,
            "date_of_birth": el.get("birth_date"), "current_price": price(el.get("now_cost")),
            "ownership": to_float(el.get("selected_by_percent")),
            "opta_code": el.get("opta_code"),
        })
        player_seasons.append({
            "season": season, "fpl_element_id": el["id"], "player_id": pid, "team_id": team_id,
            "position": pos,
            "start_price": price((el.get("now_cost") or 0) - (el.get("cost_change_start") or 0))
            if el.get("now_cost") is not None else None,
            "source": SOURCE,
        })
        source_ids.append({"canonical_player_id": pid, "source_name": "fpl_element",
                           "source_player_id": str(el["id"]), "season": season})
        if el.get("opta_code"):
            source_ids.append({"canonical_player_id": pid, "source_name": "opta",
                               "source_player_id": str(el["opta_code"]), "season": None})
        prices.append({
            "player_id": pid, "season": season, "price": price(el.get("now_cost")),
            "ownership": to_float(el.get("selected_by_percent")),
            "transfers_in_event": to_int(el.get("transfers_in_event")),
            "transfers_out_event": to_int(el.get("transfers_out_event")),
            "transfers_in": to_int(el.get("transfers_in")),
            "transfers_out": to_int(el.get("transfers_out")),
            "cost_change_event": to_int(el.get("cost_change_event")),
            "observed_at": observed, "source": SOURCE, "snapshot_id": snapshot_id,
        })
        status = el.get("status")
        chance_this = to_int(el.get("chance_of_playing_this_round"))
        chance_next = to_int(el.get("chance_of_playing_next_round"))
        news = el.get("news") or None
        news_added = to_ts(el.get("news_added"))
        availability.append({
            "player_id": pid, "source": SOURCE, "status_code": status,
            "injury_status": FPL_STATUS.get(status, status),
            "chance_of_playing_this_round": chance_this,
            "chance_of_playing_next_round": chance_next,
            # NULL chance = FPL made no statement; do not read it as 100%.
            "availability_probability": None if chance_next is None else chance_next / 100,
            "injury_description": news if status in {"d", "i"} else None,
            "expected_return_date": parse_expected_return(news, observed),
            "suspension_status": "suspended" if status == "s" else None,
            "news": news, "news_added": news_added,
            "confidence": 1.0,  # official source
            "content_hash": availability_hash(status, chance_this, chance_next, news, news_added),
            "observed_at": observed, "snapshot_id": snapshot_id,
        })
    return {
        "season": season, "teams": teams, "team_seasons": team_seasons, "gameweeks": gameweeks,
        "players": players, "player_seasons": player_seasons, "source_ids": source_ids,
        "prices": prices, "availability": availability,
    }


def fixture_rows(fixtures: list[dict], season: str, teams: dict[int, int], source: str = SOURCE) -> list[dict]:
    rows = []
    for f in fixtures:
        rows.append({
            "season": season, "fpl_fixture_id": to_int(f["id"]), "fpl_code": to_int(f.get("code")),
            "gameweek": to_int(f.get("event")),
            "home_team_id": teams[to_int(f["team_h"])], "away_team_id": teams[to_int(f["team_a"])],
            "kickoff_time": to_ts(f.get("kickoff_time")),
            "started": to_bool(f.get("started")), "finished": to_bool(f.get("finished")),
            "finished_provisional": to_bool(f.get("finished_provisional")),
            "postponed": to_int(f.get("event")) is None,
            "home_score": to_int(f.get("team_h_score")), "away_score": to_int(f.get("team_a_score")),
            "official_home_fdr": to_int(f.get("team_h_difficulty")),
            "official_away_fdr": to_int(f.get("team_a_difficulty")),
            "source": source,
        })
    return rows


def history_rows(history: list[dict], player_id: int, season: str, fixtures: dict[int, tuple],
                 observed: datetime, source: str = SOURCE) -> list[dict]:
    rows = []
    for h in history:
        fx = fixtures.get(to_int(h["fixture"]))
        if fx is None:
            log.warning("unknown fixture %s for player %s (%s)", h["fixture"], player_id, season)
            continue
        fixture_id, home_id, away_id, gw, kickoff = fx
        was_home = to_bool(h.get("was_home"))
        rows.append({
            "player_id": player_id, "fixture_id": fixture_id, "season": season,
            "gameweek": to_int(h.get("round")) or gw,
            "team_id": home_id if was_home else away_id,
            "opponent_team_id": away_id if was_home else home_id,
            "was_home": was_home, "kickoff_time": to_ts(h.get("kickoff_time")) or kickoff,
            **match_stats(h),
            "shots": None,
            "source": source, "observed_at": observed,
        })
    return rows


# ---------------------------------------------------------------------------
# sync jobs
# ---------------------------------------------------------------------------
def upsert_season(conn: psycopg.Connection, season: str, is_current: bool = False) -> None:
    if is_current:
        conn.execute("UPDATE core.seasons SET is_current = false WHERE season <> %s", (season,))
    conn.execute(
        "INSERT INTO core.seasons (season, start_year, is_current) VALUES (%s, %s, %s) "
        "ON CONFLICT (season) DO UPDATE SET is_current = core.seasons.is_current OR EXCLUDED.is_current",
        (season, int(season[:4]), is_current),
    )


def sync_bootstrap(conn: psycopg.Connection) -> str:
    with db.track_run(conn, "fpl_bootstrap") as run, fetcher(conn) as f:
        resp = f.get("/bootstrap-static/")
        observed = datetime.now(timezone.utc)
        r = bootstrap_rows(resp.json(), observed, resp.snapshot_id)
        season = r["season"]
        latest_hash = dict(conn.execute(
            "SELECT player_id, content_hash FROM core.v_latest_availability WHERE source = %s", (SOURCE,)
        ).fetchall())
        changed_availability = [a for a in r["availability"] if latest_hash.get(a["player_id"]) != a["content_hash"]]
        with conn.transaction():
            upsert_season(conn, season, is_current=True)
            run.add("teams", db.upsert(conn, "core.teams", r["teams"], ["team_id"], touch_updated_at=True))
            run.add("team_seasons", db.upsert(conn, "core.team_seasons", r["team_seasons"],
                                              ["season", "fpl_team_id"], touch_updated_at=True))
            run.add("gameweeks", db.upsert(conn, "core.gameweeks", r["gameweeks"],
                                           ["season", "gameweek"], touch_updated_at=True))
            run.add("players", db.upsert(conn, "core.players", r["players"], ["player_id"], touch_updated_at=True))
            run.add("player_seasons", db.upsert(conn, "core.player_seasons", r["player_seasons"],
                                                ["season", "fpl_element_id"], touch_updated_at=True))
            db.upsert(conn, "core.player_source_ids", r["source_ids"],
                      ["source_name", "source_player_id", "(COALESCE(season, ''))"], update=[])
            run.add("price_history", db.insert(conn, "core.player_price_history", r["prices"]))
            run.add("availability_changes", db.insert(conn, "core.player_availability", changed_availability))
        log.info("bootstrap %s: %s", season, run.details)
        return season


def sync_fixtures(conn: psycopg.Connection, season: str | None = None) -> int:
    season = season or current_season(conn)
    with db.track_run(conn, "fpl_fixtures") as run, fetcher(conn) as f:
        resp = f.get("/fixtures/")
        rows = fixture_rows(resp.json(), season, team_map(conn, season))
        with conn.transaction():
            run.add("fixtures", db.upsert(conn, "core.fixtures", rows, ["season", "fpl_fixture_id"],
                                          touch_updated_at=True))
        return run.rows


def sync_player_histories(conn: psycopg.Connection, season: str | None = None,
                          element_ids: list[int] | None = None) -> int:
    """Per-fixture stats for the current season + past-season totals, one request per player."""
    season = season or current_season(conn)
    elements = element_map(conn, season)
    fixtures = fixture_map(conn, season)
    ids = element_ids or sorted(elements)
    with db.track_run(conn, "fpl_player_histories") as run, fetcher(conn) as f:
        for i, eid in enumerate(ids, 1):
            pid = elements[eid]
            resp = f.get(f"/element-summary/{eid}/")
            data = resp.json()
            observed = datetime.now(timezone.utc)
            rows = history_rows(data.get("history", []), pid, season, fixtures, observed)
            past = [{
                "player_id": pid, "season": p["season_name"].replace("/", "-"),
                "start_price": price(p.get("start_cost")), "end_price": price(p.get("end_cost")),
                "total_points": to_int(p.get("total_points")), "minutes": to_int(p.get("minutes")),
                "data": p, "observed_at": observed,
            } for p in data.get("history_past", [])]
            with conn.transaction():
                run.add("player_match_stats", db.upsert(conn, "core.player_match_stats", rows,
                                                        ["player_id", "fixture_id"]))
                run.add("player_season_history", db.upsert(conn, "core.player_season_history", past,
                                                           ["player_id", "season"]))
            if i % 100 == 0:
                log.info("player histories: %d/%d", i, len(ids))
        return run.rows


def sync_live(conn: psycopg.Connection, gameweek: int | None = None) -> int:
    season = current_season(conn)
    if gameweek is None:
        row = conn.execute("SELECT gameweek FROM core.gameweeks WHERE season = %s AND is_current",
                           (season,)).fetchone()
        if not row:
            raise RuntimeError("No current gameweek; run sync-bootstrap first")
        gameweek = row[0]
    with db.track_run(conn, "fpl_live") as run, fetcher(conn) as f:
        resp = f.get(f"/event/{gameweek}/live/")
        observed = datetime.now(timezone.utc)
        rows = [{
            "season": season, "gameweek": gameweek, "fpl_element_id": e["id"],
            "stats": e["stats"], "explain": e.get("explain"),
            "observed_at": observed, "snapshot_id": resp.snapshot_id,
        } for e in resp.json()["elements"]]
        with conn.transaction():
            run.add("live", db.upsert(conn, "raw.fpl_live_gameweek", rows,
                                      ["season", "gameweek", "fpl_element_id"]))
        return run.rows
