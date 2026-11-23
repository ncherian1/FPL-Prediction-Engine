"""One place that assembles the modelling frame used by the component models (notebooks 06–08)."""
from __future__ import annotations

import pandas as pd
import psycopg

from ..features import build_features, load_base
from . import attacking as at
from . import team_strength as ts


def build_model_frame(base: pd.DataFrame, matches: pd.DataFrame, ctx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Player-fixture features + pre-deadline team context + the per-match actuals some models train on."""
    ctx = at.team_context(matches) if ctx is None else ctx
    b = at.add_team_columns(base, ctx)
    f = build_features(b)
    extra = b[["player_id", "fixture_id", "team_lam_for", "team_lam_against", "team_p_cs", "team_xg_on", "xg", "xa",
               "yellow_cards", "red_cards", "own_goals", "penalties_missed", "penalties_saved", "goals_conceded",
               "bps", "clean_sheet"]].rename(columns={
        "team_xg_on": "team_xg_on_actual", "xg": "y_xg_raw", "xa": "y_xa_raw", "yellow_cards": "y_yellow",
        "red_cards": "y_red", "own_goals": "y_own_goals", "penalties_missed": "y_pen_missed",
        "penalties_saved": "y_pen_saved", "goals_conceded": "y_goals_conceded", "bps": "y_bps",
        "clean_sheet": "y_cs"})
    return f.merge(extra, on=["player_id", "fixture_id"], how="left")


def load_model_frame(conn: psycopg.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = load_base(conn)
    matches = ts.load_matches(conn)
    return build_model_frame(base, matches), matches
