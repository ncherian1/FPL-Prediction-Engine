"""The full component model: every component combined by Monte Carlo simulation.

For each player-fixture we simulate N matches:
  minutes class (0 / 1-59 / 60+)          <- minutes model
  team goals for / against                  <- team-strength model (Poisson)
  goals while he's on the pitch             <- Binomial(team goals, minutes/90)
  his goals / assists                       <- each on-pitch team goal is his with prob = his share (attacking model)
  clean sheet / goals conceded              <- on-pitch goals against; CS needs 60+ minutes
  defensive contribution, saves, cards      <- their component models, at the simulated minutes
  bonus                                     <- E[bonus | this draw's events] (bonus model), drawn as Binomial(3, e/3)
and score each draw with the target season's rules. Draws for a player's fixtures in the same gameweek are summed
(double gameweeks), so P(return), P(haul) and percentiles are gameweek-level.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import nbinom

from ..scoring import Rules
from .attacking import AttackingModel
from .defensive import CardsModel, DefensiveContributionModel, SavesModel
from .minutes import MinutesModel

RETURN_PTS, HAUL_PTS = 6, 10


class SimBonusModel:
    """E[bonus | minutes class, goals, assists, clean sheet, DC hit, save points, position, BPS habits].

    A compact version of the notebook 07 bonus model whose inputs are exactly what the simulation produces, so it
    can be pre-computed on a small grid of event combinations per player."""
    FEATURES = ["cls", "goals", "assists", "cs", "dc_hit", "save_pts", "pos_GKP", "pos_DEF", "pos_MID", "pos_FWD",
                "bps90_ewm5", "bonus90_ewm5"]

    def fit(self, train: pd.DataFrame) -> "SimBonusModel":
        t = train[train.y_minutes > 0].dropna(subset=["y_bonus"]).copy()
        thr = t.position.map({"DEF": 10, "MID": 12, "FWD": 12}).fillna(np.inf)
        t["cls"] = np.where(t.y_minutes >= 60, 2, 1)
        t["goals"], t["assists"] = t.y_goals.clip(upper=3), t.y_assists.clip(upper=2)
        t["cs"] = t.y_cs.fillna(0).clip(upper=1)
        t["dc_hit"] = (t.y_dc >= thr).astype(float)
        t["save_pts"] = np.floor(t.y_saves.fillna(0).astype(float) / 3).clip(upper=3)
        self.reg = LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                                 objective="poisson", verbose=-1).fit(t[self.FEATURES].astype(float), t.y_bonus)
        return self

    # event grid: (cls, goals, assists, cs, dc_hit, save_pts)
    GRID = np.array(list(itertools.product([1, 2], range(4), range(3), [0, 1], [0, 1], range(4))))

    def grid_expectations(self, d: pd.DataFrame) -> np.ndarray:
        """Expected bonus for every grid combination, per row: shape (rows, len(GRID))."""
        n, g = len(d), len(self.GRID)
        X = pd.DataFrame(np.tile(self.GRID, (n, 1)),
                         columns=["cls", "goals", "assists", "cs", "dc_hit", "save_pts"])
        for c in ["pos_GKP", "pos_DEF", "pos_MID", "pos_FWD", "bps90_ewm5", "bonus90_ewm5"]:
            X[c] = np.repeat(d[c].fillna(0).to_numpy(float), g)
        return np.clip(self.reg.predict(X[self.FEATURES]), 0, 3).reshape(n, g)

    @staticmethod
    def grid_index(cls, goals, assists, cs, dc_hit, save_pts):
        """Map simulated events (arrays) to GRID row numbers (product order: cls, goals, assists, cs, dc, saves)."""
        return ((((((cls - 1) * 4 + np.minimum(goals, 3)) * 3 + np.minimum(assists, 2)) * 2 + cs) * 2 + dc_hit) * 4
                + np.minimum(save_pts, 3))


class ComponentModel:
    name = "component_sim"

    def __init__(self, rules: Rules, n_sims: int = 10_000, seed: int = 0, minutes_model=None):
        self.rules, self.n_sims, self.seed = rules, n_sims, seed
        self.minutes = minutes_model or MinutesModel()

    def fit(self, train: pd.DataFrame) -> "ComponentModel":
        tr = train[train.season >= "2022-23"]
        self.minutes.fit(tr)
        self.attacking = AttackingModel(); self.attacking.fit(tr)
        self.dc = DefensiveContributionModel().fit(tr)
        self.saves = SavesModel().fit(tr)
        self.cards = CardsModel().fit(tr)
        self.bonus = SimBonusModel().fit(tr)
        return self

    # -- per-row ingredients (no randomness) ------------------------------------------------------------------
    def ingredients(self, d: pd.DataFrame, minutes_probs: pd.DataFrame | None = None,
                    extra_goal_rate: pd.Series | None = None) -> pd.DataFrame:
        mp = minutes_probs if minutes_probs is not None else self.minutes.predict(d)
        cm = self.minutes.class_minutes.reindex(d.position)
        m1, m2 = cm[1].to_numpy(), cm[2].to_numpy()
        lam_team = d.team_lam_for.fillna(1.4).to_numpy()
        att90 = self.attacking.predict(d, minutes=pd.Series(90.0, index=d.index))
        q_g = (att90.lam_goals.to_numpy() / lam_team)
        if extra_goal_rate is not None:                       # e.g. penalty-taker adjustment, goals per 90
            q_g = q_g + extra_goal_rate.reindex(d.index).fillna(0).to_numpy() / lam_team
        q_a = att90.lam_assists.to_numpy() / lam_team
        ing = pd.DataFrame({
            "p0": mp.p0.to_numpy(), "p1": mp.p_1_59.to_numpy(), "p2": mp.p_60.to_numpy(), "m1": m1, "m2": m2,
            "lam_for": lam_team, "lam_against": d.team_lam_against.fillna(1.4).to_numpy(),
            "q_g": np.clip(q_g, 0, 0.9), "q_a": np.clip(q_a, 0, 0.9)}, index=d.index)
        for cls, mins in ((1, m1), (2, m2)):
            mm = pd.Series(mins, index=d.index)
            ing[f"p_dc{cls}"] = self.dc.predict(d, minutes=mm).p_dc.to_numpy()
            ing[f"mu_sv{cls}"] = self.saves.predict(d, minutes=mm).mu_saves.to_numpy()
            ing[f"p_y{cls}"] = self.cards.predict(d, minutes=mm).p_yellow.to_numpy()
        ing["sv_size"] = self.saves.size.get("GKP", 1e6)
        return ing

    # -- simulation ---------------------------------------------------------------------------------------------
    def simulate_points(self, d: pd.DataFrame, ing: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        """Points draws, shape (rows, n_sims), under self.rules."""
        n, N, R = len(d), self.n_sims, self.rules
        pos = d.position.to_numpy()

        def col(x):
            return np.asarray(x, float)[:, None]

        u = rng.random((n, N))
        cls = np.where(u < col(ing.p0), 0, np.where(u < col(ing.p0 + ing.p1), 1, 2))
        frac = np.where(cls == 2, col(ing.m2), np.where(cls == 1, col(ing.m1), 0.0)).clip(0, 90) / 90
        g_for = rng.poisson(np.broadcast_to(col(ing.lam_for), (n, N)))
        g_ag = rng.poisson(np.broadcast_to(col(ing.lam_against), (n, N)))
        on_for = rng.binomial(g_for, frac)
        on_ag = rng.binomial(g_ag, frac)
        goals = rng.binomial(on_for, col(ing.q_g))
        q_a_adj = np.clip(col(ing.q_a) / np.clip(1 - col(ing.q_g), 0.1, None), 0, 0.95)
        assists = rng.binomial(on_for - goals, q_a_adj)
        played = cls > 0
        cs = ((cls == 2) & (on_ag == 0)).astype(int)
        dc_hit = (rng.random((n, N)) < np.where(cls == 2, col(ing.p_dc2), col(ing.p_dc1))) & played
        mu_sv = np.where(cls == 2, col(ing.mu_sv2), col(ing.mu_sv1)) * played
        r = float(ing.sv_size.iloc[0]) if len(ing) else 1e6
        saves = rng.negative_binomial(r, r / (r + np.clip(mu_sv, 1e-9, None))) * played
        yellow = (rng.random((n, N)) < np.where(cls == 2, col(ing.p_y2), col(ing.p_y1))) & played
        save_pts = np.floor(saves / 3)

        # bonus: expected value for this draw's events, then a discrete draw
        grid = self.bonus.grid_expectations(d)                                  # (n, G)
        gi = SimBonusModel.grid_index(np.maximum(cls, 1), goals, assists, cs, dc_hit.astype(int),
                                      np.where(pos[:, None] == "GKP", save_pts, 0).astype(int))
        e_bonus = np.take_along_axis(grid, gi, axis=1) * played
        bonus = rng.binomial(3, np.clip(e_bonus / 3, 0, 1))

        def p(event):
            return np.array([R.p(event, x) for x in pos])[:, None]

        parts = {
            "appearance": np.where(cls == 2, p("appearance_60"), np.where(cls == 1, p("appearance_1_59"), 0)),
            "goals": goals * p("goal"), "assists": assists * p("assist"), "clean_sheet": cs * p("clean_sheet"),
            "goals_conceded": np.floor(on_ag / 2) * p("goals_conceded") * played, "saves": save_pts * p("saves"),
            "cards": yellow * p("yellow_card"), "defensive": dc_hit * p("defensive_contribution"),
            "bonus": bonus * p("bonus")}
        pts = sum(parts.values()).astype(np.float32)
        self._last_parts = {k: v.mean(axis=1) for k, v in parts.items()}      # expected points per component
        return pts

    def predict(self, d: pd.DataFrame, minutes_probs=None, extra_goal_rate=None, chunk: int = 250,
                return_draws: bool = False):
        rng = np.random.default_rng(self.seed)
        ing = self.ingredients(d, minutes_probs, extra_goal_rate)
        blocks, parts = [], []
        for i in range(0, len(d), chunk):
            blocks.append(self.simulate_points(d.iloc[i:i + chunk], ing.iloc[i:i + chunk], rng))
            parts.append(pd.DataFrame(self._last_parts))
        draws = np.vstack(blocks) if blocks else np.zeros((0, self.n_sims), np.float32)
        breakdown = (pd.concat(parts, ignore_index=True).add_prefix("xp_").set_index(d.index) if parts
                     else pd.DataFrame(index=d.index))
        out = pd.DataFrame({"xpts": draws.mean(axis=1), "xmins": (ing.p1 * ing.m1 + ing.p2 * ing.m2).to_numpy(),
                            "p_plays": (1 - ing.p0).to_numpy(), "p_60": ing.p2.to_numpy()}, index=d.index)
        # gameweek-level distribution: sum draws over a player's fixtures in the same gameweek
        key = d.player_id.astype(str) + "_" + d.gw_key.astype(str)
        codes, uniq = pd.factorize(key)
        gw_draws = np.zeros((len(uniq), draws.shape[1]), np.float32)
        np.add.at(gw_draws, codes, draws)
        gw_stats = pd.DataFrame({
            "p_return": (gw_draws >= RETURN_PTS).mean(axis=1), "p_haul": (gw_draws >= HAUL_PTS).mean(axis=1),
            "p10": np.percentile(gw_draws, 10, axis=1), "p50": np.percentile(gw_draws, 50, axis=1),
            "p90": np.percentile(gw_draws, 90, axis=1)})
        out = pd.concat([out, gw_stats.iloc[codes].set_index(d.index), breakdown], axis=1)
        return (out, draws) if return_draws else out
