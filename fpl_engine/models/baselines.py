"""Simple baselines every real model must beat (spec §10). All predict expected points per fixture."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class Zero:
    name = "zero"

    def fit(self, train): pass

    def predict(self, test):
        return pd.Series(0.0, index=test.index)


class LastNAverage:
    """Average points per squad fixture over the last N (bench games count as 0)."""

    def __init__(self, n: int = 5):
        self.n, self.name = n, f"last{n}_avg"

    def fit(self, train): pass

    def predict(self, test):
        return test[f"ptsmatch_l{self.n}"].fillna(0.0)


class SeasonAverage:
    """Average points per squad fixture this season (last season's average at a season's first gameweek)."""
    name = "season_avg"

    def fit(self, train): pass

    def predict(self, test):
        return test.ptsmatch_season.fillna(0.0)


class FplForm:
    """Official-style FPL form: average points per appearance over the last 30 days (0 if no appearance)."""
    name = "fpl_form"

    def fit(self, train): pass

    def predict(self, test):
        return test.fpl_form30.fillna(0.0)


class SimpleLinear:
    """A deliberately small, explainable expected-points baseline: ridge regression per position on a handful of
    playing-time and underlying-rate features."""
    name = "simple_linear"
    FEATURES = ["mins_l1", "mins_l3", "starts_l5", "xgi90_ewm5", "dc90_ewm5", "bps90_ewm5", "was_home", "price",
                "long_absence"]

    def __init__(self):
        self.models: dict[str, Ridge] = {}

    def _x(self, d):
        return d[self.FEATURES].astype(float).fillna(0.0)

    def fit(self, train):
        t = train.dropna(subset=["y_points"])
        for pos, g in t.groupby("position"):
            self.models[pos] = Ridge(alpha=1.0).fit(self._x(g), g.y_points)

    def predict(self, test):
        out = pd.Series(0.0, index=test.index)
        for pos, g in test.groupby("position"):
            if pos in self.models:
                out.loc[g.index] = np.clip(self.models[pos].predict(self._x(g)), 0, None)
        return out


def all_baselines():
    return [Zero(), LastNAverage(5), SeasonAverage(), FplForm(), SimpleLinear()]
