"""Direct benchmark: one LightGBM regression from all pre-deadline features straight to points.

No components, no simulation, just "learn points from everything". If the component model can't beat this,
the extra structure isn't paying its way."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

EXCLUDE_PREFIXES = ("y_",)
EXCLUDE = {"player_id", "fixture_id", "season", "gameweek", "gw_key", "deadline", "kickoff_time", "team_id",
           "opponent_team_id", "position", "rotation_relevant", "team_xg_on_actual", "team_xg"}


def feature_columns(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c not in EXCLUDE and not c.startswith(EXCLUDE_PREFIXES)
            and pd.api.types.is_numeric_dtype(f[c])]


class DirectLGBM:
    name = "direct_lgbm"

    def __init__(self, sample_weight=None, **params):
        self.sample_weight = sample_weight
        self.params = {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 200,
                       "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "verbose": -1, **params}

    def fit(self, train: pd.DataFrame) -> "DirectLGBM":
        t = train[train.season >= "2022-23"].dropna(subset=["y_points"])
        self.features = feature_columns(t)
        w = self.sample_weight(t) if self.sample_weight else None
        self.reg = LGBMRegressor(**self.params).fit(t[self.features].astype(float), t.y_points, sample_weight=w)
        return self

    def predict(self, test: pd.DataFrame) -> pd.Series:
        return pd.Series(np.clip(self.reg.predict(test[self.features].astype(float)), -1, None), index=test.index)
