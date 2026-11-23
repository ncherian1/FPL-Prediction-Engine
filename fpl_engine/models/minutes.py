"""Minutes model: P(0), P(1-59), P(60+) per player-fixture, and expected minutes.

Appearance points (1 or 2) and clean-sheet eligibility (60+) depend exactly on this split.
Live predictions are additionally capped by FPL availability (`apply_availability`); there is no availability
history to train on, so the model itself learns from playing-time patterns only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

CLASSES = ("p0", "p_1_59", "p_60")
FEATURES = [
    "mins_l1", "mins_l3", "mins_l5", "mins_l10", "apps_l3", "apps_l5", "apps_l10",
    "starts_l3", "starts_l5", "starts_l10", "n_l10", "days_since_last", "long_absence", "fixtures_seen",
    "fixtures_at_club", "new_at_club", "price", "was_home", "dgw", "gw_of_season",
    "pos_GKP", "pos_DEF", "pos_MID", "pos_FWD",
]


def minutes_class(minutes: pd.Series) -> pd.Series:
    return pd.cut(minutes, [-1, 0, 59, 1000], labels=[0, 1, 2]).astype(int)


class MinutesModel:
    name = "minutes_lgbm"

    def __init__(self, features: list[str] | None = None, sample_weight=None, **params):
        self.features = features or FEATURES
        self.sample_weight = sample_weight          # callable(train) -> weights (recency, notebook 09)
        self.params = {"n_estimators": 400, "learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 200,
                       "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1, **params}
        self.clf: LGBMClassifier | None = None
        self.class_minutes: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> None:
        t = train.dropna(subset=["y_minutes"])
        y = minutes_class(t.y_minutes)
        w = self.sample_weight(t) if self.sample_weight else None
        self.clf = LGBMClassifier(objective="multiclass", **self.params).fit(t[self.features], y, sample_weight=w)
        # average minutes within each class, by position (for expected minutes)
        self.class_minutes = t.assign(c=y).groupby(["position", "c"]).y_minutes.mean().unstack()

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        p = pd.DataFrame(self.clf.predict_proba(test[self.features]), columns=CLASSES, index=test.index)
        cm = self.class_minutes.reindex(test.position)
        p["xmins"] = p.p_1_59 * cm[1].to_numpy() + p.p_60 * cm[2].to_numpy()
        p["p_plays"] = 1 - p.p0
        return p


class LookupMinutes:
    """Baseline: empirical class frequencies for players with the same recent starts / appearances (last 5)."""
    name = "minutes_lookup"

    def fit(self, train: pd.DataFrame) -> None:
        t = train.dropna(subset=["y_minutes"]).copy()
        t["key"] = self._key(t)
        t["c"] = minutes_class(t.y_minutes)
        self.table = pd.crosstab(t.key, t.c, normalize="index").reindex(columns=[0, 1, 2], fill_value=0)
        self.prior = t.c.value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)
        self.class_minutes = t.groupby(["position", "c"]).y_minutes.mean().unstack()

    @staticmethod
    def _key(d):
        return (d.starts_l5.fillna(0) * 5).round().astype(int).astype(str) + "_" + \
               (d.apps_l5.fillna(0) * 5).round().astype(int).astype(str)

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        probs = self.table.reindex(self._key(test)).fillna(self.prior).to_numpy()
        p = pd.DataFrame(probs, columns=CLASSES, index=test.index)
        cm = self.class_minutes.reindex(test.position)
        p["xmins"] = p.p_1_59 * cm[1].to_numpy() + p.p_60 * cm[2].to_numpy()
        p["p_plays"] = 1 - p.p0
        return p


def apply_availability(p: pd.DataFrame, chance_next: pd.Series, status: pd.Series) -> pd.DataFrame:
    """Live overlay (plan step 4): FPL's chance_of_playing caps P(plays); injured/suspended/unavailable -> 0.
    Probability mass removed from playing goes to P(0); the 1-59 / 60+ split is kept proportional."""
    cap = (chance_next / 100).where(chance_next.notna(), 1.0)
    cap = cap.where(~status.isin(["i", "s", "u", "n"]), 0.0)
    scale = np.minimum(1.0, cap / p.p_plays.clip(lower=1e-9))
    out = p.copy()
    out["p_1_59"] *= scale
    out["p_60"] *= scale
    out["p0"] = 1 - out.p_1_59 - out.p_60
    out["p_plays"] = 1 - out.p0
    out["xmins"] = p.xmins * scale
    return out
