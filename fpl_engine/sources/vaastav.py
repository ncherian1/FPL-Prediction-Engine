"""Vaastav Fantasy-Premier-League GitHub archive — historical FPL data, 2016-17 onwards.

Used for historical backfill only; the current season comes from the official FPL API.
Column sets differ by season (e.g. xG from 2022-23, key passes/big chances only 2016-17..2018-19);
anything a season lacks is stored as NULL.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg

from .. import db
from ..http import FetchError, Fetcher
from ..transform import is_missing, price, season_label, to_bool, to_int
from .fpl_api import fixture_map, fixture_rows, history_rows, upsert_season

log = logging.getLogger(__name__)
SOURCE = "vaastav"
BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
FIRST_SEASON = 2016
POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD", 5: "AM"}
# Historical files never change once a season is over.
CACHE_TTL = timedelta(days=30)


def _csv(f: Fetcher, path: str, required: bool = True) -> list[dict] | None:
    try:
        resp = f.get(f"{path}", cache_ttl=CACHE_TTL)
    except FetchError as exc:
        if exc.status == 404 and not required:
            return None
        raise
    df = pd.read_csv(io.StringIO(resp.text()), dtype=str, keep_default_na=False)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df.to_dict("records")


def derive_fixtures(gw_rows: list[dict]) -> list[dict]:
    """Rebuild a fixtures list from per-player rows (seasons without fixtures.csv).

    Home team = opponent of any away-side row; away team = opponent of any home-side row."""
    fx: dict[int, dict] = {}
    for r in gw_rows:
        fid = to_int(r.get("fixture"))
        if fid is None:
            continue
        f = fx.setdefault(fid, {"id": fid, "event": r.get("round") or r.get("GW"),
                                "kickoff_time": r.get("kickoff_time"), "finished": True,
                                "team_h_score": r.get("team_h_score"), "team_a_score": r.get("team_a_score")})
        if to_bool(r.get("was_home")):
            f.setdefault("team_a", r.get("opponent_team"))
        else:
            f.setdefault("team_h", r.get("opponent_team"))
    complete = [f for f in fx.values() if "team_h" in f and "team_a" in f]
    if len(complete) != len(fx):
        log.warning("%d fixtures could not be fully reconstructed", len(fx) - len(complete))
    return complete


def import_season(conn: psycopg.Connection, season: str) -> int:
    with db.track_run(conn, f"vaastav_import:{season}") as run, \
            Fetcher(conn, SOURCE, BASE_URL, min_interval=0.2, timeout=120) as f:
        observed = datetime.now(timezone.utc)
        players_raw = _csv(f, f"{season}/players_raw.csv")
        teams_csv = _csv(f, f"{season}/teams.csv", required=False)
        fixtures_csv = _csv(f, f"{season}/fixtures.csv", required=False)
        gw_rows = _csv(f, f"{season}/gws/merged_gw.csv")
        master = _csv(f, "master_team_list.csv", required=False) or []

        # --- teams: season fpl id -> canonical code ------------------------------------
        team_code: dict[int, int] = {}
        for p in players_raw:
            if not is_missing(p.get("team")) and not is_missing(p.get("team_code")):
                team_code[to_int(p["team"])] = to_int(p["team_code"])
        names = {to_int(t["team"]): t["team_name"] for t in master if t.get("season") == season}
        strengths = {to_int(t["id"]): t for t in (teams_csv or [])}
        teams, team_seasons = [], []
        for fpl_id, code in sorted(team_code.items()):
            t = strengths.get(fpl_id, {})
            teams.append({"team_id": code, "team_name": t.get("name") or names.get(fpl_id) or f"team_{code}",
                          "short_name": t.get("short_name") or None})
            team_seasons.append({
                "season": season, "fpl_team_id": fpl_id, "team_id": code,
                **{k: to_int(t.get(k)) for k in (
                    "strength", "strength_overall_home", "strength_overall_away",
                    "strength_attack_home", "strength_attack_away",
                    "strength_defence_home", "strength_defence_away")},
                "source": SOURCE,
            })

        # --- players --------------------------------------------------------------------
        players, player_seasons, source_ids = [], [], []
        element_to_player: dict[int, int] = {}
        for p in players_raw:
            eid, code = to_int(p.get("id")), to_int(p.get("code"))
            if eid is None or code is None:
                continue
            element_to_player[eid] = code
            pos = POSITIONS.get(to_int(p.get("element_type")))
            players.append({"player_id": code, "first_name": p.get("first_name") or None,
                            "second_name": p.get("second_name") or None, "web_name": p.get("web_name") or None,
                            "position": pos, "team_id": team_code.get(to_int(p.get("team")))})
            now_cost, change = to_int(p.get("now_cost")), to_int(p.get("cost_change_start"))
            player_seasons.append({
                "season": season, "fpl_element_id": eid, "player_id": code,
                "team_id": team_code.get(to_int(p.get("team"))), "position": pos,
                "start_price": price(now_cost - change) if now_cost is not None and change is not None else None,
                "source": SOURCE,
            })
            source_ids.append({"canonical_player_id": code, "source_name": "fpl_element",
                               "source_player_id": str(eid), "season": season})

        # --- fixtures -------------------------------------------------------------------
        raw_fixtures = fixtures_csv if fixtures_csv else derive_fixtures(gw_rows)
        fixtures = fixture_rows(raw_fixtures, season, team_code, source=SOURCE)

        # --- gameweeks (deadlines not in the archive) ------------------------------------
        gws = sorted({to_int(r.get("GW") or r.get("round")) for r in gw_rows} - {None})
        gameweeks = [{"season": season, "gameweek": g, "finished": True, "source": SOURCE} for g in gws]

        with conn.transaction():
            upsert_season(conn, season)
            db.upsert(conn, "core.teams", teams, ["team_id"], update=[])
            run.add("team_seasons", db.upsert(conn, "core.team_seasons", team_seasons, ["season", "fpl_team_id"]))
            db.upsert(conn, "core.players", players, ["player_id"], update=[])
            run.add("player_seasons", db.upsert(conn, "core.player_seasons", player_seasons,
                                                ["season", "fpl_element_id"]))
            db.upsert(conn, "core.player_source_ids", source_ids,
                      ["source_name", "source_player_id", "(COALESCE(season, ''))"], update=[])
            run.add("gameweeks", db.upsert(conn, "core.gameweeks", gameweeks, ["season", "gameweek"], update=[]))
            run.add("fixtures", db.upsert(conn, "core.fixtures", fixtures, ["season", "fpl_fixture_id"]))

            fmap = fixture_map(conn, season)
            stats, unmatched = [], 0
            for r in gw_rows:
                pid = element_to_player.get(to_int(r.get("element")))
                if pid is None:
                    unmatched += 1
                    continue
                stats.extend(history_rows([r], pid, season, fmap, observed, source=SOURCE))
            run.add("player_match_stats", db.upsert(conn, "core.player_match_stats", stats,
                                                    ["player_id", "fixture_id"]))
            run.details["unmatched_rows"] = unmatched
        log.info("vaastav %s: %s", season, run.details)
        return run.rows


def available_seasons(current_start_year: int) -> list[str]:
    return [season_label(y) for y in range(FIRST_SEASON, current_start_year)]

