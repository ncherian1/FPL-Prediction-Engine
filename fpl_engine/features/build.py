"""Leak-free player-fixture feature table.

One row per player per fixture. Every feature for a fixture in gameweek g is computed ONLY from that player's
fixtures in gameweeks before g (i.e. known at g's deadline). Rolling state is taken at the end of the previous
gameweek, so in a double gameweek the second match can't see the first.

Targets (what actually happened) sit alongside as `y_*` columns and are never used as inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import psycopg

from ..scoring import load_rules, score

TARGET_RULES_SEASON = "2026-27"          # targets are re-scored under these rules (decision after notebook 01)
POSITIONS = ("GKP", "DEF", "MID", "FWD")
DC_THRESHOLD = {"GKP": np.inf, "DEF": 10, "MID": 12, "FWD": 12}


@dataclass
class FeatureConfig:
    windows: tuple[int, ...] = (3, 5, 10)        # last-N squad fixtures (0-minute fixtures included)
    rate_windows: tuple[int, ...] = (5, 10)      # windows for per-90 rates
    ewm_halflives: tuple[float, ...] = (5.0, 10.0, 20.0)   # exponentially weighted per-90 rates, in fixtures
    # shrinkage toward the position average, in minutes of "benefit of the doubt" (notebook 02 decision):
    # 540 for attacking/scoring stats, 90 for defensive counts (many actions per match -> reliable quickly)
    shrink_minutes: dict = field(default_factory=lambda: {"default": 540.0, "dc": 90.0})
    absence_cap_days: float = 60.0               # days_since_last is capped here; longer gaps set a flag
    prior_season: str = "2021-22"                # position priors come from this warm-up season only
    rate_stats: tuple[str, ...] = ("xg", "xa", "xgi", "dc", "bonus", "bps", "points")
    # extra per-match stats whose windowed SUMS are exposed (only if the column exists), e.g. for shares
    sum_stats: tuple[str, ...] = ("goals", "assists", "team_xg_on")
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
BASE_SQL = """
SELECT v.player_id, v.fixture_id, v.season, f.gameweek, f.kickoff_time, v.team_id, v.opponent_team_id,
       v.was_home, v.position, v.minutes, v.starts, v.goals, v.assists, v.clean_sheet, v.goals_conceded,
       v.saves, v.penalties_saved, v.penalties_missed, v.yellow_cards, v.red_cards, v.own_goals, v.bonus, v.bps,
       v.expected_goals AS xg, v.expected_assists AS xa, v.expected_goal_involvements AS xgi,
       v.expected_goals_conceded AS xgc, v.defensive_contribution_adjusted::float AS dc,
       v.fpl_points, v.price::float AS price, v.selected, v.transfers_balance,
       gw.deadline_time
FROM core.v_player_match_stats v
JOIN core.fixtures f USING (fixture_id)
LEFT JOIN core.gameweeks gw ON gw.season = f.season AND gw.gameweek = f.gameweek
WHERE v.season >= %s AND f.gameweek IS NOT NULL AND v.position IN ('GKP','DEF','MID','FWD')
"""


def load_base(conn: psycopg.Connection, since: str = "2021-22") -> pd.DataFrame:
    cur = conn.execute(BASE_SQL, (since,))
    df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    num = ["minutes", "starts", "goals", "assists", "clean_sheet", "goals_conceded", "saves", "penalties_saved",
           "penalties_missed", "yellow_cards", "red_cards", "own_goals", "bonus", "bps", "xg", "xa", "xgi", "xgc",
           "dc", "fpl_points", "price", "selected", "transfers_balance"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").astype("float64")
    df["kickoff_time"] = pd.to_datetime(df.kickoff_time, utc=True)
    df["deadline_time"] = pd.to_datetime(df.deadline_time, utc=True)
    df = add_deadlines(df)
    df["points"] = rescore(conn, df)
    return df


def add_deadlines(df: pd.DataFrame) -> pd.DataFrame:
    """Historical deadlines aren't stored: approximate as first kickoff of the gameweek minus 90 minutes."""
    first_ko = df.groupby(["season", "gameweek"]).kickoff_time.transform("min")
    df["deadline"] = df.deadline_time.fillna(first_ko - pd.Timedelta(minutes=90))
    df["gw_key"] = df.season.str[:4].astype(int) * 100 + df.gameweek.astype(int)   # sortable across seasons
    return df


def rescore(conn: psycopg.Connection, df: pd.DataFrame) -> pd.Series:
    """Points under TARGET_RULES_SEASON rules using definition-adjusted defensive counts."""
    s = df.rename(columns={"dc": "defensive_contribution"})
    return score(s, load_rules(conn, TARGET_RULES_SEASON))["total"]


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def position_priors(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Per-90 means by position, per stat, from the EARLIEST season that has the stat (xG only exists from
    2022-23). Using one early season keeps later training seasons free of leakage."""
    played = df[df.minutes > 0]
    out: dict[str, dict[str, float]] = {}
    for stat in cfg.rate_stats:
        cov = played.groupby("season")[stat].apply(lambda x: x.notna().mean())
        seasons = [se for se in sorted(cov.index) if se >= cfg.prior_season and cov[se] > 0.9]
        if not seasons:
            continue
        w = played[played.season == seasons[0]].dropna(subset=[stat])
        for pos, g in w.groupby("position"):
            out.setdefault(pos, {})[stat] = g[stat].sum() / (g.minutes.sum() / 90)
            out[pos][f"{stat}__season"] = seasons[0]
    return pd.DataFrame(out).T


def build_features(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    cfg = cfg or FeatureConfig()
    extra_sums = tuple(c for c in cfg.sum_stats if c in df.columns and c not in cfg.rate_stats)
    all_stats = tuple(cfg.rate_stats) + extra_sums
    df = df.sort_values(["player_id", "kickoff_time", "fixture_id"]).reset_index(drop=True)
    priors = position_priors(df, cfg)

    d = df.copy()
    d["played"] = (d.minutes > 0).astype(float)
    d["start60"] = (d.minutes >= 60).astype(float)
    d["mins90"] = d.minutes / 90
    for s in all_stats:
        d[f"_{s}"] = d[s].where(d.minutes > 0, 0.0)          # stat only counts when he played

    g = d.groupby("player_id", sort=False)
    roll = pd.DataFrame(index=d.index)
    for n in cfg.windows:
        r = g[["minutes", "played", "start60"]].rolling(n, min_periods=1)
        mean = r.mean().reset_index(level=0, drop=True)
        roll[f"mins_l{n}"] = mean.minutes
        roll[f"apps_l{n}"] = mean.played
        roll[f"starts_l{n}"] = mean.start60
        roll[f"n_l{n}"] = g.minutes.rolling(n, min_periods=1).count().reset_index(level=0, drop=True)
    roll["mins_l1"] = d.minutes
    for n in cfg.rate_windows:
        cols = ["mins90"] + [f"_{s}" for s in all_stats]
        sums = g[cols].rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)
        for s in all_stats:
            roll[f"{s}_sum_l{n}"] = sums[f"_{s}"]
        roll[f"mins90_sum_l{n}"] = sums.mins90
    for hl in cfg.ewm_halflives:
        cols = ["mins90"] + [f"_{s}" for s in all_stats]
        e = g[cols].transform(lambda x: x.ewm(halflife=hl, ignore_na=True).mean())
        for s in all_stats:
            roll[f"{s}_ewm{int(hl)}_num"] = e[f"_{s}"]
        roll[f"mins90_ewm{int(hl)}"] = e.mins90
    # per-match points (baselines): bench games count as 0, so rotation risk is priced in
    for n in cfg.windows:
        roll[f"ptsmatch_l{n}"] = g.points.rolling(n, min_periods=1).mean().reset_index(level=0, drop=True)
    sg = d.groupby(["player_id", "season"], sort=False).points
    roll["ptsmatch_season"] = sg.cumsum() / (sg.cumcount() + 1)   # at a season's first GW this carries last season's
    # official-style FPL form: mean official points per APPEARANCE over the last 30 days
    played_pts = d.fpl_points.where(d.minutes > 0)
    form = (pd.DataFrame({"player_id": d.player_id, "kickoff_time": d.kickoff_time, "v": played_pts})
            .set_index("kickoff_time").groupby("player_id", sort=False).v.rolling("30D", min_periods=1).mean())
    roll["fpl_form30"] = form.to_numpy()                          # same (player, kickoff) order as d
    roll["last_kickoff"] = d.kickoff_time
    roll["n_fixtures_seen"] = g.cumcount() + 1

    # state at the END of each (player, gameweek) block, then shifted to the NEXT block (=> pre-deadline)
    roll["player_id"], roll["gw_key"] = d.player_id, d.gw_key
    block_end = roll.groupby(["player_id", "gw_key"], sort=False).tail(1)
    prev = block_end.groupby("player_id", sort=False).shift(1)
    prev["player_id"], prev["gw_key"] = block_end.player_id, block_end.gw_key
    feats = d[["player_id", "gw_key"]].merge(prev, on=["player_id", "gw_key"], how="left")
    feats.index = d.index

    # per-90 rates with shrinkage toward the position prior
    def k_for(stat: str) -> float:
        sm = cfg.shrink_minutes if isinstance(cfg.shrink_minutes, dict) else {"default": cfg.shrink_minutes}
        return sm.get(stat, sm.get("default", 540.0)) / 90

    out: dict[str, pd.Series] = {}          # collect columns, build the frame once (avoids fragmentation)
    for c in [c for c in feats if c.startswith(("mins_l", "apps_l", "starts_l", "n_l", "ptsmatch_", "fpl_form"))]:
        out[c] = feats[c]
    for s in cfg.rate_stats:
        prior = d.position.map(priors[s]).astype(float) if s in priors else 0.0
        k = k_for(s)
        for n in cfg.rate_windows:
            m90 = feats[f"mins90_sum_l{n}"]
            out[f"{s}90_l{n}"] = (feats[f"{s}_sum_l{n}"] + k * prior) / (m90 + k)
            out[f"{s}90_l{n}_raw"] = feats[f"{s}_sum_l{n}"] / m90.where(m90 > 0)
        for hl in cfg.ewm_halflives:
            # EWM means -> approximate sums via the effective number of fixtures, then shrink like the fixed windows
            n_eff = 1 / (1 - 0.5 ** (1 / hl))
            num, den = feats[f"{s}_ewm{int(hl)}_num"] * n_eff, feats[f"mins90_ewm{int(hl)}"] * n_eff
            out[f"{s}90_ewm{int(hl)}"] = (num + k * prior) / (den + k)
    # windowed sums (per match, pre-deadline) for share-type features built elsewhere
    for n in cfg.rate_windows:
        out[f"mins90_sum_l{n}"] = feats[f"mins90_sum_l{n}"]
        for s in ("xg", "xa") + extra_sums:
            if f"{s}_sum_l{n}" in feats:
                out[f"{s}_sum_l{n}"] = feats[f"{s}_sum_l{n}"]
    for hl in cfg.ewm_halflives:
        n_eff = 1 / (1 - 0.5 ** (1 / hl))
        out[f"mins90_sum_ewm{int(hl)}"] = feats[f"mins90_ewm{int(hl)}"] * n_eff
        for s in ("xg", "xa") + extra_sums:
            if f"{s}_ewm{int(hl)}_num" in feats:
                out[f"{s}_sum_ewm{int(hl)}"] = feats[f"{s}_ewm{int(hl)}_num"] * n_eff
    gap = (d.kickoff_time - feats.last_kickoff).dt.total_seconds() / 86400
    out["days_since_last"] = gap.clip(upper=cfg.absence_cap_days)
    out["long_absence"] = (gap > cfg.absence_cap_days).astype(float)     # long injury, new signing, summer break
    out["fixtures_seen"] = feats.n_fixtures_seen.fillna(0)

    # context known before the deadline
    out["was_home"] = d.was_home.astype(float)
    out["price"] = d.price
    for p in POSITIONS:
        out[f"pos_{p}"] = (d.position == p).astype(float)
    out["dgw"] = d.groupby(["player_id", "gw_key"]).fixture_id.transform("count").gt(1).astype(float)
    # squad fixtures at his CURRENT club before this gameweek (new signings start at 0); counted per gameweek
    # block so a double gameweek's second match doesn't count the first
    blk = d.groupby(["player_id", "team_id", "gw_key"], sort=False).size().rename("n").reset_index()
    blk = blk.sort_values(["player_id", "team_id", "gw_key"])
    blk["prior"] = blk.groupby(["player_id", "team_id"]).n.cumsum() - blk.n
    out["fixtures_at_club"] = d[["player_id", "team_id", "gw_key"]].merge(
        blk, on=["player_id", "team_id", "gw_key"], how="left").prior.to_numpy()
    out["new_at_club"] = pd.Series(out["fixtures_at_club"] < 5, index=d.index).astype(float)
    out["gw_of_season"] = d.gameweek.astype(float)

    ids = d[["player_id", "fixture_id", "season", "gameweek", "gw_key", "deadline", "kickoff_time", "team_id",
             "opponent_team_id", "position"]]
    y = d[["minutes", "points", "fpl_points", "goals", "assists", "clean_sheet", "bonus", "dc", "saves"]].add_prefix("y_")
    return pd.concat([ids, pd.DataFrame(out, index=d.index), y], axis=1)


def rotation_relevant(f: pd.DataFrame) -> pd.Series:
    """Headline evaluation population (decision after notebook 00): played in 3+ of last 5 squad fixtures."""
    return f.apps_l5.fillna(0) * f.n_l5.fillna(0) >= 3
