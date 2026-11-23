"""Live predictions for an upcoming gameweek: build pre-deadline rows, apply the availability and penalty-taker
overlays, simulate, and store an append-only prediction run."""
from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import psycopg

from . import db
from .features import load_base
from .features.build import add_deadlines
from .models import attacking as at
from .models import team_strength as ts
from .models.dataset import build_model_frame
from .models.minutes import apply_availability
from .models.simulate import ComponentModel
from .scoring import load_rules
from .sources import fpl_api

log = logging.getLogger(__name__)
MODEL_VERSION = "component-v1"
PEN_XG = 0.79


def target_gameweek(conn: psycopg.Connection) -> tuple[str, int]:
    season = fpl_api.current_season(conn)
    row = conn.execute("SELECT gameweek FROM core.gameweeks WHERE season = %s AND is_next", (season,)).fetchone()
    if row is None:
        row = conn.execute("SELECT min(gameweek) FROM core.gameweeks WHERE season = %s AND NOT finished",
                           (season,)).fetchone()
    return season, int(row[0])


def upcoming_rows(conn: psycopg.Connection, season: str, gameweek: int) -> pd.DataFrame:
    """One row per current player per fixture of his team in `gameweek` (stats unknown = NaN)."""
    cur = conn.execute("""
        SELECT ps.player_id, f.fixture_id, f.season, f.gameweek, f.kickoff_time, ps.team_id,
               CASE WHEN f.home_team_id = ps.team_id THEN f.away_team_id ELSE f.home_team_id END AS opponent_team_id,
               f.home_team_id = ps.team_id AS was_home, ps.position, p.current_price::float AS price,
               gw.deadline_time
        FROM core.player_seasons ps
        JOIN core.players p USING (player_id)
        JOIN core.fixtures f ON f.season = ps.season AND f.gameweek = %s
                            AND ps.team_id IN (f.home_team_id, f.away_team_id)
        LEFT JOIN core.gameweeks gw ON gw.season = f.season AND gw.gameweek = f.gameweek
        WHERE ps.season = %s AND ps.position IN ('GKP','DEF','MID','FWD')""", (gameweek, season))
    up = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    up["kickoff_time"] = pd.to_datetime(up.kickoff_time, utc=True)
    up["deadline_time"] = pd.to_datetime(up.deadline_time, utc=True)
    return up


def availability(conn: psycopg.Connection) -> pd.DataFrame:
    cur = conn.execute("""SELECT player_id, status_code, chance_of_playing_next_round, news, observed_at
                          FROM core.v_latest_availability WHERE source = 'fpl_api'""")
    return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description]).set_index("player_id")


def penalty_adjustment(conn: psycopg.Connection, frame: pd.DataFrame, rows: pd.DataFrame,
                       penalties_order: dict[int, int], calib: dict[str, float]) -> pd.Series:
    """Extra goals per 90 for the penalty overlay (decision after notebook 06).

    The xG share already contains penalties a player has *recently taken*. The overlay moves that penalty xG to
    whoever FPL lists as first-choice taker now: + the team's expected penalty xG for the designated taker, − the
    penalty xG a player has been taking if he is no longer first choice. Calibrated like goals."""
    cur = conn.execute("""
        SELECT i.canonical_player_id AS player_id, (m.payload->>'date')::date AS date,
               (s->>'expected_goals')::float AS xg
        FROM raw.provider_records r
        JOIN raw.provider_records m ON m.source_name = 'pitchapi' AND m.entity = 'match' AND m.entity_id = r.entity_id
        CROSS JOIN jsonb_array_elements(r.payload->'periods') p
        CROSS JOIN jsonb_array_elements(p->'shots') s
        JOIN core.player_source_ids i ON i.source_name = 'pitchapi' AND i.source_player_id = s->'player'->>'id'
        WHERE r.source_name = 'pitchapi' AND r.entity = 'match_shots' AND s->>'situation' = 'Penalty'""")
    pens = pd.DataFrame(cur.fetchall(), columns=["player_id", "date", "xg"])
    since = (pd.Timestamp.now(tz="UTC") - timedelta(days=365)).date()
    recent = pens[pens.date >= since]
    played = frame[(frame.kickoff_time >= pd.Timestamp(since, tz="UTC")) & (frame.y_minutes > 0)]
    team_matches = played.groupby("team_id").fixture_id.nunique().sum() or 1
    team_pen_xg_per_match = recent.xg.sum() / team_matches                 # per team per match
    mins90 = played.groupby("player_id").y_minutes.sum() / 90
    own_pen_xg90 = (recent.groupby("player_id").xg.sum() / mins90).fillna(0)
    designated = rows.player_id.map(lambda p: penalties_order.get(int(p)) == 1)
    adj = (np.where(designated, team_pen_xg_per_match, 0.0)
           - rows.player_id.map(own_pen_xg90).fillna(0).to_numpy())
    return pd.Series(adj * rows.position.map(calib).fillna(1.0).to_numpy(), index=rows.index)


def predict_gameweek(conn: psycopg.Connection, gameweek: int | None = None, n_sims: int = 10_000,
                     write: bool = True, details: bool = False):
    """Predict a gameweek. details=True also returns the fitted model and inputs (for notebooks)."""
    season, gw = (fpl_api.current_season(conn), gameweek) if gameweek else target_gameweek(conn)
    base = load_base(conn)
    up = upcoming_rows(conn, season, gw)
    up = add_deadlines(pd.concat([up, base.iloc[0:0]], ignore_index=True))
    base_all = pd.concat([base, up], ignore_index=True)
    matches = ts.load_matches(conn)
    frame = build_model_frame(base_all, matches)
    frame["y_saves"] = pd.to_numeric(frame.y_saves, errors="coerce")
    test = frame[(frame.season == season) & (frame.gameweek == gw) & frame.fixture_id.isin(up.fixture_id)
                 & frame.y_minutes.isna()]
    train = frame[frame.y_minutes.notna()]
    rules = load_rules(conn, season)
    model = ComponentModel(rules, n_sims=n_sims).fit(train)

    # overlays
    av = availability(conn)
    mp = model.minutes.predict(test)
    chance = test.player_id.map(av.chance_of_playing_next_round).astype(float)
    status = test.player_id.map(av.status_code).fillna("a")
    mp = apply_availability(mp, chance, status)
    with fpl_api.fetcher(conn) as f:
        boot = f.get("/bootstrap-static/").json()
    order = {e["code"]: e.get("penalties_order") for e in boot["elements"]}
    pen = penalty_adjustment(conn, train, test, order, model.attacking.calib["goals"])
    pred = model.predict(test, minutes_probs=mp, extra_goal_rate=pen)
    gw_xpts = pred.groupby(test.player_id).xpts.transform("sum")
    out = pd.concat([test[["player_id", "fixture_id", "season", "gameweek", "position", "team_id", "was_home"]],
                     pred.assign(gw_xpts=gw_xpts, penalty_adjustment=pen,
                                 status=status, chance=chance)], axis=1)
    if write:
        _write(conn, out, season, gw, train, n_sims)
    if details:
        return out, {"model": model, "test": test, "minutes_probs": mp, "penalty": pen, "train": train}
    return out


def _write(conn, out, season, gw, train, n_sims):
    with conn.transaction():
        run_id = conn.execute(
            """INSERT INTO predictions.model_runs (model_name, model_version, target_season, target_gameweek,
                   trained_through, n_train_rows, config) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING run_id""",
            ("component_sim", MODEL_VERSION, season, gw, train.kickoff_time.max(), len(train),
             db.Jsonb({"n_sims": n_sims, "rules_season": season}))).fetchone()[0]
        parts = [c for c in out if c.startswith("xp_")]
        rows = [{
            "run_id": run_id, "player_id": int(r.player_id), "fixture_id": int(r.fixture_id), "season": r.season,
            "gameweek": int(r.gameweek), "xpts": float(r.xpts), "xmins": float(r.xmins), "p_plays": float(r.p_plays),
            "p_60": float(r.p_60), "gw_xpts": float(r.gw_xpts), "p_return": float(r.p_return),
            "p_haul": float(r.p_haul), "p10": float(r.p10), "p50": float(r.p50), "p90": float(r.p90),
            "breakdown": {c[3:]: round(float(getattr(r, c)), 4) for c in parts},
            "availability": {"status": r.status, "chance_next": None if pd.isna(r.chance) else float(r.chance)},
            "penalty_adjustment": float(r.penalty_adjustment)} for r in out.itertuples()]
        db.insert(conn, "predictions.player_fixture_predictions", rows)
    log.info("prediction run %s: %d player-fixtures for %s GW%s", run_id, len(rows), season, gw)
    return run_id
