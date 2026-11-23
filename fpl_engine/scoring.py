"""FPL scoring engine.

Rules are data (core.scoring_rules), never hard-coded in models. `score()` turns per-fixture
stats into a per-component points breakdown under any season's rules — which also lets us
re-score history under *current* rules ("rule-normalised points").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import psycopg

POSITIONS = ("GKP", "DEF", "MID", "FWD")

# Every scoring event, in the order components are reported.
# threshold semantics: minutes_* = minute cut-off; saves / goals_conceded = "per N";
# defensive_contribution = count needed to score.
EVENTS = (
    "appearance_1_59", "appearance_60", "goal", "assist", "clean_sheet", "goals_conceded",
    "saves", "penalty_saved", "penalty_missed", "yellow_card", "red_card", "own_goal",
    "defensive_contribution", "bonus",
)


@dataclass
class Rules:
    season: str
    points: dict[str, dict[str, float]] = field(default_factory=dict)      # event -> position -> points
    threshold: dict[str, dict[str, float]] = field(default_factory=dict)   # event -> position -> threshold

    def p(self, event: str, pos: str) -> float:
        return self.points.get(event, {}).get(pos, 0.0)

    def t(self, event: str, pos: str) -> float | None:
        return self.threshold.get(event, {}).get(pos)

    source: str = "derived"

    def to_rows(self) -> list[dict]:
        return [{"season": self.season, "event_type": e, "player_position": pos,
                 "points_value": self.p(e, pos), "threshold": self.t(e, pos), "source": self.source}
                for e in EVENTS for pos in POSITIONS]


def _uniform(v: float) -> dict[str, float]:
    return {p: v for p in POSITIONS}


def legacy_rules(season: str, *, gk_goal: float = 6, defensive_contribution: bool = False) -> Rules:
    """Rules used 2016-17 .. 2024-25 (no defensive contributions)."""
    r = Rules(season)
    r.points = {
        "appearance_1_59": _uniform(1), "appearance_60": _uniform(2),
        "goal": {"GKP": gk_goal, "DEF": 6, "MID": 5, "FWD": 4},
        "assist": _uniform(3),
        "clean_sheet": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
        "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},
        "saves": {"GKP": 1, "DEF": 0, "MID": 0, "FWD": 0},
        "penalty_saved": _uniform(5), "penalty_missed": _uniform(-2),
        "yellow_card": _uniform(-1), "red_card": _uniform(-3), "own_goal": _uniform(-2),
        "bonus": _uniform(1),
        "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2} if defensive_contribution else _uniform(0),
    }
    r.threshold = {
        "appearance_60": _uniform(60),
        "goals_conceded": _uniform(2),
        "saves": _uniform(3),
        "defensive_contribution": {"GKP": None, "DEF": 10, "MID": 12, "FWD": 12},
    }
    return r


def rules_from_game_config(season: str, scoring: dict) -> Rules:
    """Build rules from bootstrap-static game_config.scoring (current season)."""
    def per_pos(v):
        return {p: float(v.get(p, 0)) for p in POSITIONS} if isinstance(v, dict) else _uniform(float(v))
    r = legacy_rules(season)
    r.source = "game_config"
    r.points.update({
        "appearance_1_59": per_pos(scoring["short_play"]), "appearance_60": per_pos(scoring["long_play"]),
        "goal": per_pos(scoring["goals_scored"]), "assist": per_pos(scoring["assists"]),
        "clean_sheet": per_pos(scoring["clean_sheets"]), "goals_conceded": per_pos(scoring["goals_conceded"]),
        "saves": {**_uniform(0), "GKP": float(scoring["saves"])},
        "penalty_saved": per_pos(scoring["penalties_saved"]), "penalty_missed": per_pos(scoring["penalties_missed"]),
        "yellow_card": per_pos(scoring["yellow_cards"]), "red_card": per_pos(scoring["red_cards"]),
        "own_goal": per_pos(scoring["own_goals"]), "bonus": per_pos(scoring["bonus"]),
        "defensive_contribution": per_pos(scoring.get("defensive_contribution", 0)),
    })
    return r


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def save_rules(conn: psycopg.Connection, rules: Rules) -> None:
    from . import db
    with conn.transaction():
        db.upsert(conn, "core.scoring_rules", rules.to_rows(), ["season", "event_type", "player_position"])


# Historical rules, validated: they reproduce official points for 100% of appearances
# 2016-17 .. 2025-26 (notebooks/01_scoring_engine.ipynb). GK goals were 6 points then.
def historical_rules(season: str) -> Rules:
    return legacy_rules(season, gk_goal=6, defensive_contribution=int(season[:4]) >= 2025)


def seed_rules(conn: psycopg.Connection, current_scoring: dict | None = None) -> list[str]:
    """Store rules for every season in core.seasons; the current season comes from game_config."""
    seasons = conn.execute("SELECT season, is_current FROM core.seasons ORDER BY season").fetchall()
    for season, is_current in seasons:
        rules = (rules_from_game_config(season, current_scoring) if is_current and current_scoring
                 else historical_rules(season))
        save_rules(conn, rules)
    return [s for s, _ in seasons]


def load_rules(conn: psycopg.Connection, season: str) -> Rules:
    rows = conn.execute("SELECT event_type, player_position, points_value, threshold FROM core.scoring_rules "
                        "WHERE season = %s", (season,)).fetchall()
    if not rows:
        raise LookupError(f"no scoring rules stored for {season}")
    r = Rules(season)
    for event, pos, pts, thr in rows:
        r.points.setdefault(event, {})[pos] = float(pts)
        if thr is not None:
            r.threshold.setdefault(event, {})[pos] = float(thr)
    return r


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
STAT_COLUMNS = ("minutes", "goals", "assists", "clean_sheet", "goals_conceded", "saves", "penalties_saved",
                "penalties_missed", "yellow_cards", "red_cards", "own_goals", "defensive_contribution", "bonus")


def score(stats: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """Points per component for each row of `stats` (needs `position` + STAT_COLUMNS).

    Returns one column per event plus `total`. A component whose input stat is missing
    (NaN) is NaN, and so is `total` — unknown is never silently treated as zero.
    """
    s = stats.copy()
    for c in STAT_COLUMNS:
        s[c] = pd.to_numeric(s[c], errors="coerce").astype("float64")
    pos = s["position"].astype(object)

    def pts(event):
        return pos.map(lambda p: rules.p(event, p)).astype("float64")

    def thr(event):
        return pos.map(lambda p: rules.t(event, p)).astype("float64")

    played = s.minutes > 0
    long_play = s.minutes >= thr("appearance_60")
    out = pd.DataFrame(index=s.index)
    out["appearance"] = np.where(long_play, pts("appearance_60"), np.where(played, pts("appearance_1_59"), 0.0))
    out.loc[s.minutes.isna(), "appearance"] = np.nan
    out["goals"] = s.goals * pts("goal")
    out["assists"] = s.assists * pts("assist")
    out["clean_sheet"] = s.clean_sheet * pts("clean_sheet")
    out["goals_conceded"] = np.floor(s.goals_conceded / thr("goals_conceded")) * pts("goals_conceded")
    out["saves"] = np.floor(s.saves / thr("saves")) * pts("saves")
    out["penalties"] = s.penalties_saved * pts("penalty_saved") + s.penalties_missed * pts("penalty_missed")
    out["cards"] = s.yellow_cards * pts("yellow_card") + s.red_cards * pts("red_card")
    out["own_goals"] = s.own_goals * pts("own_goal")
    dc_pts, dc_thr = pts("defensive_contribution"), thr("defensive_contribution")
    out["defensive_contribution"] = np.where(dc_pts == 0, 0.0,
                                             np.where(s.defensive_contribution >= dc_thr, dc_pts, 0.0))
    out.loc[(dc_pts != 0) & s.defensive_contribution.isna(), "defensive_contribution"] = np.nan
    out["bonus"] = s.bonus * pts("bonus")
    # A component worth 0 points for this position is 0, even if its stat is missing.
    for col, event in (("goals_conceded", "goals_conceded"), ("saves", "saves"), ("clean_sheet", "clean_sheet")):
        out.loc[pts(event) == 0, col] = 0.0
    out["total"] = out.sum(axis=1, skipna=False)
    return out
