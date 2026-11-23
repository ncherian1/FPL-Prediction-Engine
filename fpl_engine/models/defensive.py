"""Defensive contributions, goalkeeper saves, cards and bonus.

Defensive contributions & saves: model the expected COUNT for the match, then use a negative-binomial distribution
(a Poisson that allows some matches to be much busier than others) to get the probability of crossing a threshold
(DC: 10 for DEF, 12 for MID/FWD) or the expected save points (1 per 3 saves).

    μ = player's recent rate per 90 × minutes/90 × exp(β_home·home + β_pressure·log(expected goals against / avg))

Cards: a personal per-90 rate, shrunk toward the position average.
Bonus: learned from the match's own events (goals, assists, clean sheet, defensive actions, saves...), so the
simulation in notebook 08 can give bonus that's consistent with each simulated match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import nbinom, poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor

DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}
AVG_LAM = 1.4        # typical expected goals per team-match; centres the pressure term


def _mins(d, minutes):
    return (d.y_minutes if minutes is None else minutes).clip(upper=90) / 90


def nb_size(y: pd.Series, mu: pd.Series) -> float:
    """Method-of-moments negative-binomial size r (Var = μ + μ²/r). Large r ≈ Poisson."""
    excess = ((y - mu) ** 2 - mu).sum()
    return float((mu ** 2).sum() / excess) if excess > 0 else 1e6


def p_at_least(k, mu, r):
    """P(count ≥ k) under NB(mean μ, size r); k may vary per row."""
    mu = np.clip(np.asarray(mu, float), 1e-9, None)
    r = np.asarray(r, float)
    return nbinom.sf(np.asarray(k) - 1, r, r / (r + mu))


class CountModel:
    """Shared machinery for defensive contributions and saves."""

    def __init__(self, stat: str, rate_col: str, positions: tuple[str, ...], dispersion: bool = True,
                 context: bool = True, calib_days: float = 365.0):
        self.stat, self.rate_col, self.positions = stat, rate_col, positions
        self.dispersion, self.context, self.calib_days = dispersion, context, calib_days

    def _x(self, d):
        return pd.DataFrame({"home": d.was_home.astype(float),
                             "pressure": np.log(d.team_lam_against.fillna(AVG_LAM) / AVG_LAM)}, index=d.index)

    def base_mu(self, d, minutes=None):
        return d[self.rate_col].fillna(0).clip(lower=0) * _mins(d, minutes)

    def fit(self, train):
        t = train[(train.y_minutes > 0) & train.position.isin(self.positions)].dropna(subset=[f"y_{self.stat}"])
        if self.calib_days:
            t = t[t.kickoff_time >= t.kickoff_time.max() - pd.Timedelta(days=self.calib_days)]
        e = self.base_mu(t).clip(lower=1e-3)
        y = t[f"y_{self.stat}"]
        if self.context:
            # Poisson GLM on y/e with weights e  ==  log-linear model with offset log(e)
            self.glm = PoissonRegressor(alpha=1e-4, max_iter=500).fit(self._x(t), y / e, sample_weight=e)
        mu = self.predict_mu(t)
        self.size = {pos: (nb_size(y[t.position == pos], mu[t.position == pos]) if self.dispersion else 1e6)
                     for pos in self.positions}
        return self

    def predict_mu(self, d, minutes=None):
        e = self.base_mu(d, minutes)
        if not self.context:
            return e
        return e * self.glm.predict(self._x(d))


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class DefensiveContributionModel(CountModel):
    """Count model + Platt recalibration: logit(p_final) = a + b·logit(p_raw), fitted on the latest 365 days of
    training. Two parameters on ~20k appearances, shared across positions, so there's little room to overfit."""
    name = "dc_nb"

    def __init__(self, recalibrate: bool = True, **kw):
        super().__init__("dc", "dc90_ewm5", ("DEF", "MID", "FWD"), **kw)
        self.recalibrate = recalibrate
        self.platt = None

    def _raw(self, d, minutes=None):
        mu = self.predict_mu(d, minutes)
        r = d.position.map(self.size).astype(float).fillna(1e6)
        k = d.position.map(DC_THRESHOLD).astype(float).fillna(np.inf)
        return mu, pd.Series(p_at_least(k, mu, r), index=d.index)

    def fit(self, train):
        super().fit(train)
        if self.recalibrate:
            t = train[(train.y_minutes > 0) & train.position.isin(self.positions)].dropna(subset=["y_dc"])
            t = t[t.kickoff_time >= t.kickoff_time.max() - pd.Timedelta(days=self.calib_days)]
            _, p = self._raw(t)
            y = (t.y_dc >= t.position.map(DC_THRESHOLD)).astype(int)
            # features are pre-deadline rates, so these are genuine out-of-sample predictions for each row
            self.platt = LogisticRegression(C=1e6).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, d, minutes=None):
        mu, p = self._raw(d, minutes)
        if self.platt is not None:
            p = pd.Series(self.platt.predict_proba(_logit(p).reshape(-1, 1))[:, 1], index=d.index)
        p = p.where(d.position.isin(self.positions), 0.0)
        return pd.DataFrame({"mu_dc": mu, "p_dc": p}, index=d.index)


class HitRateBaseline:
    """Baseline: share of his last 10 appearances (adjusted counts) that reached the threshold, shrunk."""
    name = "dc_hitrate"

    def fit(self, train):
        t = train[(train.y_minutes >= 60) & train.position.isin(DC_THRESHOLD)]
        self.prior = (t.y_dc >= t.position.map(DC_THRESHOLD)).groupby(t.position).mean().to_dict()
        return self

    def predict(self, d, minutes=None):
        # approximate the player's hit probability from his rate via a Poisson around his per-90 rate
        lam = d.dc90_l10.fillna(0) * _mins(d, minutes)
        k = d.position.map(DC_THRESHOLD).astype(float).fillna(np.inf)
        return pd.DataFrame({"mu_dc": lam, "p_dc": poisson.sf(k - 1, lam)}, index=d.index)


class SavesModel(CountModel):
    name = "saves_nb"

    def __init__(self, **kw):
        super().__init__("saves", "saves90_ewm5", ("GKP",), **kw)

    def predict(self, d, minutes=None):
        mu = self.predict_mu(d, minutes).where(d.position == "GKP", 0.0)
        r = self.size.get("GKP", 1e6)
        # expected save points = Σ_k P(saves ≥ 3k), k = 1.. (1 point per 3 saves)
        p = r / (r + mu.clip(lower=1e-9))
        exp_pts = sum(nbinom.sf(3 * k - 1, r, p) for k in range(1, 8))
        return pd.DataFrame({"mu_saves": mu, "x_save_pts": np.where(d.position == "GKP", exp_pts, 0.0)}, index=d.index)


class CardsModel:
    """P(yellow card) from a shrunk personal per-90 rate × minutes, calibrated on the latest year."""
    name = "cards"

    def fit(self, train):
        t = train[train.y_minutes > 0]
        t = t[t.kickoff_time >= t.kickoff_time.max() - pd.Timedelta(days=365)]
        lam = t.yellow_cards90_ewm5.fillna(0) * _mins(t, None)
        self.calib = float(t.y_yellow.sum() / lam.sum())
        self.red_rate = float(t.y_red.sum() / (t.y_minutes.clip(upper=90).sum() / 90))
        return self

    def predict(self, d, minutes=None):
        lam = d.yellow_cards90_ewm5.fillna(0) * _mins(d, minutes) * self.calib
        return pd.DataFrame({"p_yellow": 1 - np.exp(-lam), "p_red": 1 - np.exp(-self.red_rate * _mins(d, minutes))},
                            index=d.index)


class BonusModel:
    """E[bonus | what happened in the match] — trained on realised events, used with simulated events."""
    name = "bonus_lgbm"
    EVENT_FEATURES = ["y_minutes", "y_goals", "y_assists", "y_cs", "y_saves", "y_dc", "y_goals_conceded", "y_yellow",
                      "y_own_goals", "y_pen_missed", "y_pen_saved", "team_goals", "opp_goals"]
    PRIOR_FEATURES = ["bps90_ewm5", "bonus90_ewm5", "pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"]

    def __init__(self, **params):
        self.params = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
                       "objective": "poisson", "verbose": -1, **params}

    @property
    def features(self):
        return self.EVENT_FEATURES + self.PRIOR_FEATURES

    def fit(self, train):
        t = train[train.y_minutes > 0].dropna(subset=["y_bonus"])
        self.reg = LGBMRegressor(**self.params).fit(t[self.features].astype(float), t.y_bonus)
        return self

    def predict(self, d):
        return pd.Series(np.clip(self.reg.predict(d[self.features].astype(float)), 0, 3), index=d.index)


def add_match_scores(f: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Team and opponent goals in the match (realised), needed by the bonus model."""
    m = matches[["fixture_id", "home_team_id", "home_score", "away_score"]]
    g = f[["fixture_id", "team_id"]].merge(m, on="fixture_id", how="left")
    home = (g.team_id == g.home_team_id).to_numpy()
    f = f.copy()
    f["team_goals"] = np.where(home, g.home_score, g.away_score)
    f["opp_goals"] = np.where(home, g.away_score, g.home_score)
    return f
