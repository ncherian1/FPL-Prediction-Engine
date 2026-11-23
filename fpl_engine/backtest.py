"""Rolling-origin backtest harness (spec §10).

For each target gameweek g in the evaluation seasons:
  1. train on every player-fixture row from gameweeks before g (what was known at g's deadline),
     retraining every `retrain_every` gameweeks;
  2. predict expected points for every player-fixture row in g;
  3. sum per player per gameweek (double gameweeks = sum of both matches) and compare with the actual total.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .features.build import rotation_relevant

log = logging.getLogger(__name__)

RETURN_PTS, HAUL_PTS = 6, 10          # decision after notebook 00


class Model(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> None: ...

    def predict(self, test: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Expected points per fixture row (Series), or a DataFrame with `xpts` and optional
        `p_return` / `p_haul` probability columns."""
        ...


@dataclass
class BacktestConfig:
    eval_seasons: tuple[str, ...] = ("2024-25", "2025-26", "2026-27")
    first_train_season: str = "2022-23"       # 2021-22 is warm-up history only
    retrain_every: int = 5


def run(features: pd.DataFrame, models: list[Model], cfg: BacktestConfig | None = None) -> pd.DataFrame:
    """Per-fixture predictions for every model; long format."""
    cfg = cfg or BacktestConfig()
    f = features[features.season >= cfg.first_train_season].copy()
    f["rotation_relevant"] = rotation_relevant(f)
    targets = sorted(f.loc[f.season.isin(cfg.eval_seasons), "gw_key"].unique())
    out = []
    for m in models:
        fitted_at = None
        for i, g in enumerate(targets):
            if fitted_at is None or i % cfg.retrain_every == 0:
                train = f[f.gw_key < g]
                m.fit(train)
                fitted_at = g
            test = f[f.gw_key == g]
            p = m.predict(test)
            p = p.to_frame("xpts") if isinstance(p, pd.Series) else p
            out.append(pd.concat([test[["player_id", "fixture_id", "season", "gameweek", "gw_key", "position", "price",
                                        "rotation_relevant", "y_points", "y_minutes"]].reset_index(drop=True),
                                  p.reset_index(drop=True)], axis=1).assign(model=m.name, trained_through=fitted_at))
        log.info("backtest %s done", m.name)
    return pd.concat(out, ignore_index=True)


def to_gameweeks(preds: pd.DataFrame) -> pd.DataFrame:
    """Sum fixtures into player-gameweeks (double gameweek = both matches)."""
    agg = {"xpts": "sum", "y_points": "sum", "y_minutes": "sum", "rotation_relevant": "max", "position": "first",
           "price": "first"}
    for c in ("p_return", "p_haul"):
        if c in preds:
            agg[c] = "max"   # placeholder for single-fixture GWs; DGW probabilities come from the simulator later
    return preds.groupby(["model", "season", "gameweek", "gw_key", "player_id"], as_index=False).agg(agg)


def metrics(gw: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Headline metrics per model on three populations."""
    rows = []
    pops = {"rotation-relevant": gw.rotation_relevant, "played": gw.y_minutes > 0, "all": pd.Series(True, gw.index)}
    for (model, pop), d in ((k, gw[mask & (gw.model == k[0])]) for k in
                            [(m, p) for m in gw.model.unique() for p in pops] for mask in [pops[k[1]]]):
        err = d.xpts - d.y_points
        # ranking quality: within each gameweek, how well do predictions order players?
        rank = d.groupby("gw_key").apply(
            lambda x: x.xpts.corr(x.y_points, method="spearman") if x.xpts.nunique() > 1 else np.nan,
            include_groups=False)
        row = {"model": model, "population": pop, "rows": len(d), "MAE": err.abs().mean(),
               "RMSE": np.sqrt((err ** 2).mean()), "bias": err.mean(), "rank corr / GW": rank.mean()}
        if pop == "rotation-relevant":
            # practical: average actual points of the model's top-N picks each gameweek
            top = d.sort_values("xpts", ascending=False).groupby("gw_key").head(top_n)
            best = d.sort_values("y_points", ascending=False).groupby("gw_key").head(top_n)
            row[f"top{top_n} picks: actual pts"] = top.y_points.mean()
            row[f"top{top_n} hindsight max"] = best.y_points.mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index(["population", "model"]).sort_index()


def by_group(gw: pd.DataFrame, col: str, population: str = "rotation-relevant") -> pd.DataFrame:
    d = gw[gw.rotation_relevant] if population == "rotation-relevant" else gw
    return d.assign(ae=(d.xpts - d.y_points).abs()).pivot_table(index=col, columns="model", values="ae", aggfunc="mean")
