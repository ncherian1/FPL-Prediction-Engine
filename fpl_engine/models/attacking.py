"""Player attacking model: expected goals and assists per player-fixture.

    λ_goals   = goal share   × team expected goals (notebook 05) × minutes/90 × calibration
    λ_assists = assist share × team expected goals              × minutes/90 × calibration

share = player's xG (or xA) ÷ his team's xG while he was on the pitch, over recent matches, shrunk toward his
position's average share. The team expected goals already include opponent strength and home advantage, so the
player model doesn't need its own home/away split (notebook 05 §7).
Goals and assists in a match are modelled as Poisson counts: P(at least one) = 1 − e^(−λ).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

from . import team_strength as ts


# ---------------------------------------------------------------------------
# team context joined onto player rows
# ---------------------------------------------------------------------------
def team_context(matches: pd.DataFrame, cfg: ts.TeamStrengthConfig | None = None,
                 from_season: str = "2022-23") -> pd.DataFrame:
    """Per (fixture, team): actual team xG (history) and pre-deadline predictions (targets)."""
    preds = ts.rolling_predictions(matches, cfg, from_season=from_season)
    view = ts.team_view(matches, preds)
    actual = pd.concat([
        pd.DataFrame({"fixture_id": matches.fixture_id, "team_id": matches.home_team_id, "team_xg": matches.h_xg_pa}),
        pd.DataFrame({"fixture_id": matches.fixture_id, "team_id": matches.away_team_id, "team_xg": matches.a_xg_pa})])
    return actual.merge(view.drop(columns=["opponent_team_id", "was_home"]), on=["fixture_id", "team_id"], how="left")


def add_team_columns(base: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """team_xg_on = team's actual xG scaled by the share of the match the player was on the pitch."""
    b = base.drop(columns=[c for c in ("team_xg", "team_lam_for", "team_lam_against", "team_p_cs", "team_xg_on")
                           if c in base]).merge(ctx, on=["fixture_id", "team_id"], how="left")
    b["team_xg_on"] = (b.team_xg * b.minutes.clip(upper=90) / 90).where(b.minutes > 0, 0.0)
    return b


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
@dataclass
class AttackingConfig:
    window: str = "ewm5"           # which pre-deadline sums: "l5", "l10", "ewm5", "ewm10", "ewm20" (notebook 06)
    k_share: float = 10.0          # shrinkage toward the position share, in team-xG units (~6 matches; notebook 06)
    finishing: bool = False        # multiply by a (heavily shrunk) personal goals/xG ratio?
    k_finishing: float = 10.0      # shrinkage for finishing, in xG units
    calib_days: float = 365.0      # calibrate on the most recent N days of training only: FPL's xG/goals relationship
                                   # shifted (2022-23 goals/xG = 1.42, ~1.0 since), so old seasons would mis-scale


class AttackingModel:
    name = "attacking_share"

    def __init__(self, cfg: AttackingConfig | None = None):
        self.cfg = cfg or AttackingConfig()

    def _sums(self, d: pd.DataFrame, stat: str) -> pd.Series:
        return d[f"{stat}_sum_{self.cfg.window}"].fillna(0.0)

    def shares(self, d: pd.DataFrame) -> pd.DataFrame:
        k = self.cfg.k_share
        team_on = self._sums(d, "team_xg_on")
        out = pd.DataFrame(index=d.index)
        for stat in ("xg", "xa"):
            prior = d.position.map(self.prior_share[stat]).astype(float)
            out[f"share_{stat}"] = (self._sums(d, stat) + k * prior) / (team_on + k)
        if self.cfg.finishing:
            g, x = self._sums(d, "goals"), self._sums(d, "xg")
            kf = self.cfg.k_finishing
            out["finishing"] = (g + kf) / (x + kf)
        return out

    def fit(self, train: pd.DataFrame) -> None:
        t = train[(train.y_minutes > 0) & train.team_lam_for.notna()]
        played = t  # priors from actual per-match totals in the training window
        self.prior_share = {
            stat: played.groupby("position").apply(
                lambda g: g[f"y_{stat}_raw"].sum() / g.team_xg_on_actual.sum(), include_groups=False)
            for stat in ("xg", "xa")}
        if "kickoff_time" in t and self.cfg.calib_days:
            t = t[t.kickoff_time >= t.kickoff_time.max() - pd.Timedelta(days=self.cfg.calib_days)]
        sh = self.shares(t)
        mins = t.y_minutes.clip(upper=90) / 90
        raw_g = sh.share_xg * t.team_lam_for * mins * (sh.finishing if self.cfg.finishing else 1.0)
        raw_a = sh.share_xa * t.team_lam_for * mins
        # calibration: actual goals / assists per unit of raw expectation, by position
        self.calib = {
            "goals": (t.y_goals.groupby(t.position).sum() / raw_g.groupby(t.position).sum()).to_dict(),
            "assists": (t.y_assists.groupby(t.position).sum() / raw_a.groupby(t.position).sum()).to_dict()}

    def predict(self, test: pd.DataFrame, minutes: pd.Series | None = None) -> pd.DataFrame:
        """λ for goals and assists given minutes played (actual minutes by default, for component evaluation)."""
        mins = (test.y_minutes if minutes is None else minutes).clip(upper=90) / 90
        sh = self.shares(test)
        fin = sh.finishing if self.cfg.finishing else 1.0
        lam_g = sh.share_xg * test.team_lam_for * mins * fin * test.position.map(self.calib["goals"]).astype(float)
        lam_a = sh.share_xa * test.team_lam_for * mins * test.position.map(self.calib["assists"]).astype(float)
        return pd.DataFrame({"lam_goals": lam_g, "lam_assists": lam_a,
                             "p_goal": 1 - np.exp(-lam_g), "p_assist": 1 - np.exp(-lam_a)}, index=test.index)


class RatePerNinety:
    """Baselines: a per-90 rate × minutes/90, with no team or opponent adjustment."""

    def __init__(self, goal_col: str, assist_col: str, name: str):
        self.goal_col, self.assist_col, self.name = goal_col, assist_col, name

    def fit(self, train):
        t = train[train.y_minutes > 0]
        m = t.y_minutes.clip(upper=90) / 90
        self.cg = t.y_goals.sum() / (t[self.goal_col].fillna(0) * m).sum()
        self.ca = t.y_assists.sum() / (t[self.assist_col].fillna(0) * m).sum()

    def predict(self, test, minutes=None):
        mins = (test.y_minutes if minutes is None else minutes).clip(upper=90) / 90
        lg, la = test[self.goal_col].fillna(0) * mins * self.cg, test[self.assist_col].fillna(0) * mins * self.ca
        return pd.DataFrame({"lam_goals": lg, "lam_assists": la, "p_goal": 1 - np.exp(-lg),
                             "p_assist": 1 - np.exp(-la)}, index=test.index)


def evaluate(test: pd.DataFrame, p: pd.DataFrame) -> dict:
    """Poisson log-likelihood (higher = better) and Brier for 'at least one' (lower = better)."""
    g, a = test.y_goals.astype(int), test.y_assists.astype(int)
    lg, la = p.lam_goals.clip(lower=1e-6), p.lam_assists.clip(lower=1e-6)
    return {"goals log-lik": float(poisson.logpmf(g, lg).mean()),
            "goal Brier": float(((p.p_goal - (g > 0)) ** 2).mean()),
            "assists log-lik": float(poisson.logpmf(a, la).mean()),
            "assist Brier": float(((p.p_assist - (a > 0)) ** 2).mean()),
            "rows": len(test)}
