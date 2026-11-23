import math

import pandas as pd

from fpl_engine.scoring import historical_rules, legacy_rules, rules_from_game_config, score

BASE = dict(minutes=90, goals=0, assists=0, clean_sheet=0, goals_conceded=0, saves=0, penalties_saved=0,
            penalties_missed=0, yellow_cards=0, red_cards=0, own_goals=0, defensive_contribution=0, bonus=0)


def row(position, **kw):
    return {**BASE, "position": position, **kw}


def total(rows, rules):
    return score(pd.DataFrame(rows), rules)["total"].tolist()


def test_appearance_points():
    r = legacy_rules("2024-25")
    assert total([row("MID", minutes=0), row("MID", minutes=59), row("MID", minutes=60)], r) == [0, 1, 2]


def test_position_specific_goals_and_clean_sheets():
    r = legacy_rules("2024-25")
    got = total([row(p, goals=1, clean_sheet=1) for p in ("GKP", "DEF", "MID", "FWD")], r)
    assert got == [2 + 6 + 4, 2 + 6 + 4, 2 + 5 + 1, 2 + 4 + 0]


def test_per_n_rules():
    r = legacy_rules("2024-25")
    # 5 saves -> 1 point; 3 conceded -> -1 for GK/DEF, 0 for MID
    assert total([row("GKP", saves=5, goals_conceded=3), row("MID", goals_conceded=3)], r) == [2 + 1 - 1, 2]


def test_defensive_contribution_threshold_by_position():
    r = historical_rules("2025-26")
    got = total([row("DEF", defensive_contribution=9), row("DEF", defensive_contribution=10),
                 row("MID", defensive_contribution=11), row("MID", defensive_contribution=12)], r)
    assert got == [2, 4, 2, 4]
    # before 2025-26 the same count is worth nothing
    assert total([row("DEF", defensive_contribution=15)], historical_rules("2024-25")) == [2]


def test_unknown_is_not_zero():
    r = historical_rules("2025-26")
    out = score(pd.DataFrame([row("DEF", defensive_contribution=None)]), r)
    assert math.isnan(out.loc[0, "defensive_contribution"]) and math.isnan(out.loc[0, "total"])
    # ...but a stat that can't score for this position doesn't make the total unknown
    assert score(pd.DataFrame([row("FWD", saves=None)]), r).loc[0, "total"] == 2


def test_rules_from_game_config():
    cfg = {"long_play": 2, "short_play": 1, "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
           "assists": 3, "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
           "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0}, "saves": 1, "penalties_saved": 5,
           "penalties_missed": -2, "yellow_cards": -1, "red_cards": -3, "own_goals": -2, "bonus": 1,
           "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2}}
    r = rules_from_game_config("2026-27", cfg)
    assert total([row("GKP", goals=1)], r) == [12]
    assert r.source == "game_config"
