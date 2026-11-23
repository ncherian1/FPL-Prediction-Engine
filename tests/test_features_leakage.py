"""Leakage guard: features for gameweek g must not change when anything from gameweek g onwards changes."""
import numpy as np
import pandas as pd
import pytest

from fpl_engine.features.build import FeatureConfig, add_deadlines, build_features

STATS = ["minutes", "starts", "goals", "assists", "clean_sheet", "goals_conceded", "saves", "penalties_saved",
         "penalties_missed", "yellow_cards", "red_cards", "own_goals", "bonus", "bps", "xg", "xa", "xgi", "xgc",
         "dc", "fpl_points", "points", "price", "selected", "transfers_balance"]


def synthetic(seed=0, players=6, gws=12, season="2022-23"):
    rng = np.random.default_rng(seed)
    rows, fid = [], 0
    for gw in range(1, gws + 1):
        n_fx = 2 if gw == 7 else 1                      # gameweek 7 is a double gameweek
        for k in range(n_fx):
            fid += 1
            ko = pd.Timestamp("2022-08-06", tz="UTC") + pd.Timedelta(days=7 * gw + 3 * k)
            for p in range(players):
                rows.append({"player_id": p, "fixture_id": fid, "season": season, "gameweek": gw,
                             "kickoff_time": ko, "deadline_time": pd.NaT, "team_id": 1, "opponent_team_id": 2,
                             "was_home": bool(k % 2), "position": ["DEF", "MID", "FWD"][p % 3],
                             **{s: float(rng.integers(0, 4)) for s in STATS},
                             "minutes": float(rng.choice([0, 30, 90]))})
    return add_deadlines(pd.DataFrame(rows))


@pytest.mark.parametrize("target_gw", [2, 5, 7, 8, 12])
def test_features_ignore_current_and_future(target_gw):
    df = synthetic()
    cfg = FeatureConfig(prior_season="2022-23")
    # priors come from the whole warm-up season by design; hold them fixed so only rolling features are tested
    base = build_features(df, cfg)

    tampered = df.copy()
    future = tampered.gameweek >= target_gw
    rng = np.random.default_rng(99)
    for s in STATS:
        if s != "minutes":
            tampered.loc[future, s] = rng.integers(0, 50, future.sum()).astype(float)
    tampered.loc[future, "minutes"] = rng.choice([0.0, 90.0], future.sum())
    # keep the warm-up-season priors identical (they're computed once, before training seasons)
    from fpl_engine.features import build as b
    priors = b.position_priors(df, cfg)
    orig = b.position_priors
    b.position_priors = lambda *_: priors
    try:
        after = build_features(tampered, cfg)
    finally:
        b.position_priors = orig

    feature_cols = [c for c in base.columns if not c.startswith("y_") and c not in ("was_home", "price")]
    a = base[base.gameweek == target_gw].sort_values(["player_id", "fixture_id"])[feature_cols].reset_index(drop=True)
    z = after[after.gameweek == target_gw].sort_values(["player_id", "fixture_id"])[feature_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, z)


def test_double_gameweek_second_match_cannot_see_first():
    f = build_features(synthetic(), FeatureConfig(prior_season="2022-23"))
    dgw = f[f.gameweek == 7].sort_values(["player_id", "kickoff_time"])
    first, second = dgw.groupby("player_id").nth(0), dgw.groupby("player_id").nth(1)
    cols = ["mins_l5", "xgi90_l5", "points90_l5"]
    np.testing.assert_allclose(first[cols].to_numpy(), second[cols].to_numpy())


def test_double_gameweek_counts_as_two_matches_not_one_inflated_gameweek():
    """Features are per match: a DGW haul enters history as two separate matches, never as one doubled total."""
    df = synthetic()
    df["points"] = 2.0
    df["minutes"] = 90.0
    df.loc[df.gameweek == 7, "points"] = 10.0              # big double gameweek: 2 x 10
    f = build_features(df, FeatureConfig(prior_season="2022-23", shrink_minutes={"default": 1e-9}))
    after = f[f.gameweek == 8].iloc[0]
    # last 5 matches before GW8 = GW5, GW6, GW7a, GW7b + GW4 -> (2+2+2+10+10)/5 per match = 5.2 per 90
    assert abs(after["points90_l5"] - 5.2) < 1e-6
    assert after["mins_l5"] == 90.0                         # minutes per match, not 180 for the DGW


def test_absence_cap_and_flag():
    df = synthetic()
    df.loc[df.gameweek >= 6, "kickoff_time"] += pd.Timedelta(days=120)   # 4-month gap before GW6
    df = add_deadlines(df.drop(columns=["deadline", "gw_key"]))
    f = build_features(df, FeatureConfig(prior_season="2022-23"))
    gw6 = f[f.gameweek == 6]
    assert (gw6.days_since_last == 60).all() and (gw6.long_absence == 1).all()
    assert (f[f.gameweek == 5].long_absence == 0).all()
