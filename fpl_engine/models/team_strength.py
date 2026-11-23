"""Team strength & fixture difficulty.

Each team gets an ATTACK and a DEFENCE rating, re-estimated before every gameweek deadline from recent matches,
with older matches down-weighted exponentially. For a fixture:

    expected goals for team A  =  exp(base + home·[A at home] + attack_A − defence_B + promoted terms)

fitted as a weighted Poisson regression on a target that blends expected goals (xG) and actual goals.
Clean-sheet probability for team B = P(A scores 0) = exp(−λ_A).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import psycopg
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

MATCH_SQL = """
WITH pa AS (
  SELECT fs.fixture_id,
         max((i->>'home')::float) AS h_xg_pa, max((i->>'away')::float) AS a_xg_pa
  FROM raw.provider_records r
  JOIN core.fixture_source_ids fs ON fs.source_name = 'pitchapi' AND fs.source_fixture_id = r.entity_id
  CROSS JOIN jsonb_array_elements(r.payload->'periods') p
  CROSS JOIN jsonb_array_elements(p->'groups') g
  CROSS JOIN jsonb_array_elements(g->'items') i
  WHERE r.source_name = 'pitchapi' AND r.entity = 'match_stats' AND p->>'period' = 'All'
    AND i->>'key' = 'expected_goals'
  GROUP BY 1
), fx AS (
  SELECT s.fixture_id,
         sum(s.expected_goals) FILTER (WHERE s.was_home)     AS h_xg_fpl,
         sum(s.expected_goals) FILTER (WHERE NOT s.was_home) AS a_xg_fpl
  FROM core.player_match_stats s GROUP BY 1
)
SELECT f.fixture_id, f.season, f.gameweek, f.kickoff_time, gw.deadline_time, f.home_team_id, f.away_team_id,
       f.finished, f.home_score, f.away_score, f.official_home_fdr, f.official_away_fdr, pa.h_xg_pa, pa.a_xg_pa, fx.h_xg_fpl, fx.a_xg_fpl
FROM core.fixtures f
LEFT JOIN core.gameweeks gw ON gw.season = f.season AND gw.gameweek = f.gameweek
LEFT JOIN pa USING (fixture_id)
LEFT JOIN fx USING (fixture_id)
WHERE f.season >= %s AND f.gameweek IS NOT NULL
"""


def load_matches(conn: psycopg.Connection, since: str = "2021-22") -> pd.DataFrame:
    cur = conn.execute(MATCH_SQL, (since,))
    m = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    for c in ("home_score", "away_score", "h_xg_pa", "a_xg_pa", "h_xg_fpl", "a_xg_fpl"):
        m[c] = pd.to_numeric(m[c], errors="coerce").astype(float)
    m["kickoff_time"] = pd.to_datetime(m.kickoff_time, utc=True)
    first_ko = m.groupby(["season", "gameweek"]).kickoff_time.transform("min")
    m["deadline"] = pd.to_datetime(m.deadline_time, utc=True).fillna(first_ko - pd.Timedelta(minutes=90))
    m["gw_key"] = m.season.str[:4].astype(int) * 100 + m.gameweek.astype(int)
    m["played"] = m.finished.fillna(False) & m.home_score.notna()
    # promoted = not in the league the previous season
    teams_by_season = pd.concat([m[["season", "home_team_id"]].rename(columns={"home_team_id": "t"}),
                                 m[["season", "away_team_id"]].rename(columns={"away_team_id": "t"})]).drop_duplicates()
    prev = {s: set(g.t) for s, g in teams_by_season.groupby("season")}
    seasons = sorted(prev)
    promoted = {(s, t) for i, s in enumerate(seasons[1:], 1) for t in prev[s] - prev[seasons[i - 1]]}
    m["home_promoted"] = [(s, t) in promoted for s, t in zip(m.season, m.home_team_id)]
    m["away_promoted"] = [(s, t) in promoted for s, t in zip(m.season, m.away_team_id)]
    return m.sort_values(["kickoff_time", "fixture_id"]).reset_index(drop=True)


@dataclass
class TeamStrengthConfig:
    source: str = "pa"            # which xG: "pa" (PitchAPI team xG) or "fpl" (sum of FPL player xG)
    xg_weight: float = 1.0        # target = xg_weight * xG + (1 - xg_weight) * goals (notebook 05: xG only)
    half_life_days: float = 365.0
    window_days: float = 1095.0     # 3 seasons
    alpha: float = 0.002          # ridge penalty: pulls ratings toward average when data is thin
    team_home: bool = False       # experiment: each team gets its own home bonus (on top of the league-wide one)
    team_home_scale: float = 1.0  # smaller = shrink team-specific home bonuses harder toward the league average


def _long(m: pd.DataFrame, cfg: TeamStrengthConfig) -> pd.DataFrame:
    """Two rows per match: one per team attacking."""
    h = pd.DataFrame({"att": m.home_team_id, "def": m.away_team_id, "home": 1.0,
                      "att_promoted": m.home_promoted.astype(float), "def_promoted": m.away_promoted.astype(float),
                      "goals": m.home_score, "xg": m[f"h_xg_{cfg.source}"], "kickoff_time": m.kickoff_time,
                      "fixture_id": m.fixture_id})
    a = pd.DataFrame({"att": m.away_team_id, "def": m.home_team_id, "home": 0.0,
                      "att_promoted": m.away_promoted.astype(float), "def_promoted": m.home_promoted.astype(float),
                      "goals": m.away_score, "xg": m[f"a_xg_{cfg.source}"], "kickoff_time": m.kickoff_time,
                      "fixture_id": m.fixture_id})
    return pd.concat([h, a], ignore_index=True)


class TeamStrength:
    def __init__(self, cfg: TeamStrengthConfig | None = None):
        self.cfg = cfg or TeamStrengthConfig()

    def _design(self, d: pd.DataFrame) -> pd.DataFrame:
        # build in numpy: with pandas copy-on-write, writing through DataFrame.values is silently lost
        col_idx = {c: i for i, c in enumerate(self.columns)}
        X = np.zeros((len(d), len(self.columns)))
        X[:, col_idx["home"]] = d.home.to_numpy()
        X[:, col_idx["att_promoted"]] = d.att_promoted.to_numpy()
        X[:, col_idx["def_promoted"]] = d.def_promoted.to_numpy()
        if self.cfg.team_home:
            # attacking team at home -> its own extra home bonus; defending team at home -> its own defensive one
            for row, (t_att, t_def, home) in enumerate(zip(d.att.to_numpy(), d["def"].to_numpy(), d.home.to_numpy())):
                key = f"homeatt_{t_att}" if home else f"homedef_{t_def}"
                j = col_idx.get(key)
                if j is not None:
                    X[row, j] = self.cfg.team_home_scale
        for kind in ("att", "def"):
            for row, t in enumerate(d[kind].to_numpy()):
                j = col_idx.get(f"{kind}_{t}")
                if j is not None:                     # unseen team (e.g. newly promoted): average + promoted term
                    X[row, j] = 1.0
        return pd.DataFrame(X, index=d.index, columns=self.columns)

    def fit(self, history: pd.DataFrame, as_of: pd.Timestamp) -> "TeamStrength":
        cfg = self.cfg
        h = history[history.played & (history.kickoff_time < as_of)
                    & (history.kickoff_time >= as_of - pd.Timedelta(days=cfg.window_days))]
        d = _long(h, cfg)
        xg = d.xg.fillna(d.goals)                      # matches without xG fall back to goals
        d["y"] = cfg.xg_weight * xg + (1 - cfg.xg_weight) * d.goals
        age = (as_of - d.kickoff_time).dt.total_seconds() / 86400
        w = 0.5 ** (age / cfg.half_life_days)
        teams = sorted(set(d.att) | set(d["def"]))
        self.columns = ["home", "att_promoted", "def_promoted"] + [f"att_{t}" for t in teams] + [f"def_{t}" for t in teams]
        if cfg.team_home:
            self.columns += [f"homeatt_{t}" for t in teams] + [f"homedef_{t}" for t in teams]
        self.reg = PoissonRegressor(alpha=cfg.alpha, max_iter=1000).fit(self._design(d), d.y, sample_weight=w / w.mean())
        self.as_of = as_of
        return self

    def ratings(self) -> pd.DataFrame:
        coef = pd.Series(self.reg.coef_, index=self.columns)
        att = coef[coef.index.str.startswith("att_") & ~coef.index.str.contains("promoted")]
        dfn = coef[coef.index.str.startswith("def_") & ~coef.index.str.contains("promoted")]
        return pd.DataFrame({"attack": att.set_axis(att.index.str[4:].astype(int)),
                             # defence: positive = concedes MORE (weaker); flip so higher = better defence
                             "defence": -dfn.set_axis(dfn.index.str[4:].astype(int))})

    @property
    def home_advantage(self) -> float:
        return float(np.exp(self.reg.coef_[0]))

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        d = _long(fixtures, self.cfg)
        lam = self.reg.predict(self._design(d))
        n = len(fixtures)
        out = pd.DataFrame({"fixture_id": fixtures.fixture_id.to_numpy(),
                            "lam_home": lam[:n], "lam_away": lam[n:]})
        out["p_cs_home"] = np.exp(-out.lam_away)
        out["p_cs_away"] = np.exp(-out.lam_home)
        return out


def rolling_predictions(matches: pd.DataFrame, cfg: TeamStrengthConfig | None = None,
                        from_season: str = "2022-23") -> pd.DataFrame:
    """For every gameweek from `from_season`, fit on matches before its deadline and predict its fixtures
    (including unplayed future gameweeks, which use the latest fit)."""
    out = []
    model = TeamStrength(cfg)
    for gw_key, fx in matches[matches.season >= from_season].groupby("gw_key"):
        as_of = min(fx.deadline.min(), pd.Timestamp.now(tz="UTC"))
        model.fit(matches, as_of)
        p = model.predict(fx).assign(gw_key=gw_key, home_adv=model.home_advantage)
        out.append(p)
    return pd.concat(out, ignore_index=True)


def team_view(matches: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """One row per (fixture, team): expected goals for/against and clean-sheet probability."""
    m = matches.merge(preds, on="fixture_id")
    h = pd.DataFrame({"fixture_id": m.fixture_id, "team_id": m.home_team_id, "opponent_team_id": m.away_team_id,
                      "was_home": True, "team_lam_for": m.lam_home, "team_lam_against": m.lam_away,
                      "team_p_cs": m.p_cs_home})
    a = pd.DataFrame({"fixture_id": m.fixture_id, "team_id": m.away_team_id, "opponent_team_id": m.home_team_id,
                      "was_home": False, "team_lam_for": m.lam_away, "team_lam_against": m.lam_home,
                      "team_p_cs": m.p_cs_away})
    return pd.concat([h, a], ignore_index=True)


class FdrLookup:
    """Baseline: average goals scored / clean-sheet rate by the OFFICIAL FPL difficulty rating and home/away."""

    def fit(self, history: pd.DataFrame, as_of: pd.Timestamp, window_days: float = 730) -> "FdrLookup":
        h = history[history.played & (history.kickoff_time < as_of)
                    & (history.kickoff_time >= as_of - pd.Timedelta(days=window_days))]
        rows = pd.concat([
            pd.DataFrame({"fdr": h.official_home_fdr, "home": 1, "goals_for": h.home_score, "cs": h.away_score == 0}),
            pd.DataFrame({"fdr": h.official_away_fdr, "home": 0, "goals_for": h.away_score, "cs": h.home_score == 0})])
        self.table = rows.groupby(["fdr", "home"]).agg(goals_for=("goals_for", "mean"), cs=("cs", "mean"))
        self.default = rows[["goals_for", "cs"]].mean()
        return self

    def predict(self, fx: pd.DataFrame) -> pd.DataFrame:
        def look(fdr, home, col):
            return [self.table[col].get((f, home), self.default[col]) for f in fdr]
        return pd.DataFrame({"fixture_id": fx.fixture_id.to_numpy(),
                             "lam_home": look(fx.official_home_fdr, 1, "goals_for"),
                             "lam_away": look(fx.official_away_fdr, 0, "goals_for"),
                             "p_cs_home": look(fx.official_home_fdr, 1, "cs"),
                             "p_cs_away": look(fx.official_away_fdr, 0, "cs")})


def evaluate(matches: pd.DataFrame, preds: pd.DataFrame) -> dict:
    """Goal log-likelihood (higher = better) and clean-sheet Brier (lower = better) on played fixtures."""
    e = matches[matches.played].merge(preds, on="fixture_id")
    goals = pd.concat([e.home_score, e.away_score])
    lam = pd.concat([e.lam_home, e.lam_away])
    cs_p = pd.concat([e.p_cs_home, e.p_cs_away])
    cs_y = pd.concat([e.away_score == 0, e.home_score == 0]).astype(float)
    return {"goal log-lik": goal_log_likelihood(goals, lam), "goal MAE": float((goals - lam).abs().mean()),
            "CS Brier": float(((cs_p - cs_y) ** 2).mean()), "fixtures": len(e)}


def rolling_baseline(matches: pd.DataFrame, model, from_season: str = "2024-25") -> pd.DataFrame:
    out = []
    for gw_key, fx in matches[matches.season >= from_season].groupby("gw_key"):
        out.append(model.fit(matches, min(fx.deadline.min(), pd.Timestamp.now(tz="UTC"))).predict(fx))
    return pd.concat(out, ignore_index=True)


class LeagueAverage:
    """Baseline: every home team scores the league-average home goals; same for away."""

    def fit(self, history, as_of, window_days: float = 365):
        h = history[history.played & (history.kickoff_time < as_of)
                    & (history.kickoff_time >= as_of - pd.Timedelta(days=window_days))]
        self.h, self.a = h.home_score.mean(), h.away_score.mean()
        self.csh, self.csa = (h.away_score == 0).mean(), (h.home_score == 0).mean()
        return self

    def predict(self, fx):
        n = len(fx)
        return pd.DataFrame({"fixture_id": fx.fixture_id.to_numpy(), "lam_home": [self.h] * n, "lam_away": [self.a] * n,
                             "p_cs_home": [self.csh] * n, "p_cs_away": [self.csa] * n})


def goal_log_likelihood(goals: pd.Series, lam: pd.Series) -> float:
    return float(poisson.logpmf(goals.astype(int), lam).mean())
