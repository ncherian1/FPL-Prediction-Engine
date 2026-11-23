"""Cross-provider identity matching: PitchAPI teams, fixtures and players -> canonical FPL IDs.

Players are matched on evidence, not names alone:
  1. teams    — a PitchAPI team and an FPL team that played on the same dates are the same club;
  2. fixtures — same season, same home and away club;
  3. players  — candidates are only FPL players in the SAME team in the SAME fixture; each candidate pair is
                scored on name similarity AND on whether their per-match minutes / goals / assists agree.
Every decision is stored with its method, confidence and evidence. Manual overrides in
config/manual_player_mappings.csv are applied last and always win.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd
import psycopg
from rapidfuzz import fuzz

from . import db
from .config import ROOT

log = logging.getLogger(__name__)
SOURCE = "pitchapi"
MANUAL_FILE = ROOT / "config" / "manual_player_mappings.csv"

# --- thresholds (documented in notebooks/01b_player_matching.ipynb) ----------------------------
MIN_AGREE_RATE = 0.90      # share of shared appearances where minutes agree
NAME_FUZZY_OK = 80         # rapidfuzz score treated as a name match
MIN_SHARED_FOR_STATS_ONLY = 5


def norm_name(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("-", " ").replace("'", "").replace(".", " ")
    return re.sub(r"[^a-z ]", "", re.sub(r"\s+", " ", s)).strip()


def name_variants(first: str | None, second: str | None, web: str | None) -> dict[str, set[str]]:
    """Full-name forms, plus single names that are only valid as an exact mononym match."""
    f, s, w = norm_name(first), norm_name(second), norm_name(web)
    full = {f"{f} {s}".strip(), f"{s} {f}".strip(), w}
    if f and s:
        full.add(f"{f.split()[0]} {s.split()[-1]}")          # "ruben dias" from a long legal name
        full.add(f"{f.split()[0]} {s.split()[0]}")           # "marc cucurella" from "cucurella saseta"
    if " " in f:
        full.add(f)                                          # compound given name: "john victor", "deivid washington"
    return {"full": {v for v in full if v}, "mononym": {v for v in (w, f, s) if v and " " not in v},
            "surname_tokens": set((s + " " + w).split())}


def name_scores(pa_name: str, variants: dict[str, set[str]]) -> tuple[bool, float]:
    """(exact, score 0-100). A first name alone never counts as a match."""
    n = norm_name(pa_name)
    if n in variants["full"] or (" " not in n and n in variants["mononym"]):
        return True, 100.0
    score = max((fuzz.ratio(n, v) for v in variants["full"]), default=0.0)
    # a long legal name may contain the short name ("ruben dias" within "ruben dos santos gato alves dias"),
    # but only if the surname matches too (stops "jack jenkins" matching "jack harrison")
    last = n.split()[-1] if n else ""
    if any(fuzz.ratio(last, t) >= 85 for t in variants["surname_tokens"]):
        score = max(score, max(fuzz.token_set_ratio(n, v) for v in variants["full"]))
    return False, float(score)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_pitchapi(conn: psycopg.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = pd.DataFrame(conn.execute("""
        SELECT entity_id AS match_id, season, (payload->>'date')::date AS date, payload->>'status' AS status,
               payload->'home_team'->>'id' AS home_pid, payload->'home_team'->>'name' AS home_name,
               payload->'away_team'->>'id' AS away_pid, payload->'away_team'->>'name' AS away_name
        FROM raw.provider_records WHERE source_name = 'pitchapi' AND entity = 'match'""").fetchall(),
        columns=["match_id", "season", "date", "status", "home_pid", "home_name", "away_pid", "away_name"])
    apps = pd.DataFrame(conn.execute("""
        SELECT r.entity_id AS match_id, x->'player'->>'id' AS pid, x->'player'->>'name' AS name,
               x->>'team_id' AS team_pid,
               max((e.v->'stat'->>'value')::float) FILTER (WHERE e.k = 'Minutes played') AS minutes,
               max((e.v->'stat'->>'value')::float) FILTER (WHERE e.k = 'Goals') AS goals,
               max((e.v->'stat'->>'value')::float) FILTER (WHERE e.k = 'Assists') AS assists
        FROM raw.provider_records r
        CROSS JOIN jsonb_array_elements(r.payload) x
        LEFT JOIN LATERAL jsonb_array_elements(x->'stats') g ON true
        LEFT JOIN LATERAL jsonb_each(g->'stats') e(k, v) ON true
        WHERE r.source_name = 'pitchapi' AND r.entity = 'match_players'
        GROUP BY 1, 2, 3, 4""").fetchall(),
        columns=["match_id", "pid", "name", "team_pid", "minutes", "goals", "assists"])
    return matches, apps


def load_fpl(conn: psycopg.Connection, seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixtures = pd.DataFrame(conn.execute("""
        SELECT fixture_id, season, kickoff_time::date AS date, home_team_id, away_team_id
        FROM core.fixtures WHERE season = ANY(%s) AND kickoff_time IS NOT NULL""", (seasons,)).fetchall(),
        columns=["fixture_id", "season", "date", "home_team_id", "away_team_id"])
    rows = pd.DataFrame(conn.execute("""
        SELECT s.fixture_id, s.player_id, s.team_id, s.minutes, s.goals, s.assists,
               p.first_name, p.second_name, p.web_name, ps.position
        FROM core.player_match_stats s
        JOIN core.players p USING (player_id)
        LEFT JOIN core.player_seasons ps ON ps.player_id = s.player_id AND ps.season = s.season
        WHERE s.season = ANY(%s)""", (seasons,)).fetchall(),
        columns=["fixture_id", "player_id", "team_id", "minutes", "goals", "assists",
                 "first_name", "second_name", "web_name", "position"])
    return fixtures, rows


# ---------------------------------------------------------------------------
# teams + fixtures
# ---------------------------------------------------------------------------
def match_teams(matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Map (season, PitchAPI team) -> FPL team by overlap of (date, home/away) fixture keys."""
    pa = pd.concat([
        matches.assign(team_pid=matches.home_pid, team_name=matches.home_name, home=True),
        matches.assign(team_pid=matches.away_pid, team_name=matches.away_name, home=False)])
    fp = pd.concat([fixtures.assign(team_id=fixtures.home_team_id, home=True),
                    fixtures.assign(team_id=fixtures.away_team_id, home=False)])
    pa_keys = pa.groupby(["season", "team_pid", "team_name"]).apply(
        lambda g: set(zip(g.date, g.home)), include_groups=False)
    fp_keys = fp.groupby(["season", "team_id"]).apply(lambda g: set(zip(g.date, g.home)), include_groups=False)
    out = []
    for (season, pid, pname), keys in pa_keys.items():
        cands = fp_keys.loc[season] if season in fp_keys.index.get_level_values(0) else pd.Series(dtype=object)
        scored = sorted(((len(keys & k) / len(keys), tid) for tid, k in cands.items()), reverse=True)
        best, second = (scored + [(0, None), (0, None)])[:2]
        out.append({"season": season, "team_pid": pid, "team_name": pname, "team_id": best[1],
                    "confidence": round(best[0], 3), "runner_up": round(second[0], 3)})
    return pd.DataFrame(out)


def match_fixtures(matches: pd.DataFrame, teams: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    t = teams.set_index(["season", "team_pid"]).team_id
    m = matches.assign(
        home_team_id=[t.get((s, h)) for s, h in zip(matches.season, matches.home_pid)],
        away_team_id=[t.get((s, a)) for s, a in zip(matches.season, matches.away_pid)])
    return m.merge(fixtures[["season", "home_team_id", "away_team_id", "fixture_id"]],
                   on=["season", "home_team_id", "away_team_id"], how="left")


# ---------------------------------------------------------------------------
# players
# ---------------------------------------------------------------------------
def candidate_evidence(apps: pd.DataFrame, fx: pd.DataFrame, teams: pd.DataFrame,
                       fpl_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (PitchAPI player, FPL player) pair that shared a team in at least one fixture."""
    t = teams.set_index(["season", "team_pid"]).team_id
    a = apps.merge(fx[["match_id", "season", "fixture_id"]], on="match_id").dropna(subset=["fixture_id"])
    a["team_id"] = [t.get((s, p)) for s, p in zip(a.season, a.team_pid)]
    pairs = a.merge(fpl_rows, on=["fixture_id", "team_id"], suffixes=("_pa", "_fpl"))
    pa_played = pairs.minutes_pa.fillna(0) > 0
    fpl_played = pairs.minutes_fpl.fillna(0) > 0
    pairs["either_played"] = pa_played | fpl_played
    pairs["both_played"] = pa_played & fpl_played
    # played/not-played, goals and minutes (±2) agree for true pairs; assists are NOT compared because
    # FPL awards assists more generously than the official stat (they differ in ~2% of true pairs).
    pairs["agree"] = (pa_played == fpl_played) & (
        ~pairs.both_played | ((pairs.minutes_pa - pairs.minutes_fpl).abs() <= 2)
        & (pairs.goals_pa.fillna(0) == pairs.goals_fpl.fillna(0)))
    pairs["agree_played"] = pairs.agree & pairs.either_played
    pairs["status_agree"] = pa_played == fpl_played          # incl. "both unused that day"
    ev = pairs.groupby(["pid", "player_id"]).agg(
        pa_name=("name", "first"), first_name=("first_name", "first"), second_name=("second_name", "first"),
        web_name=("web_name", "first"), position=("position", "last"),
        shared_fixtures=("fixture_id", "nunique"), either_played=("either_played", "sum"),
        both_played=("both_played", "sum"), agree_played=("agree_played", "sum"),
        status_agree=("status_agree", "mean"),
        seasons=("season", lambda s: ",".join(sorted(set(s))))).reset_index()
    ev["agree_rate"] = (ev.agree_played / ev.either_played).where(ev.either_played > 0)
    names = [name_scores(n, name_variants(f, s, w))
             for n, f, s, w in zip(ev.pa_name, ev.first_name, ev.second_name, ev.web_name)]
    ev["name_exact"] = [x[0] for x in names]
    ev["name_score"] = [x[1] for x in names]
    # combined score: stats agreement dominates; names break ties. For players who never played, agreement on
    # played/not-played status is the evidence (an unused sub matches an unused sub, not a starter).
    ev["score"] = ev.agree_rate.fillna(ev.status_agree) * 70 + ev.name_score * 0.3 + ev.name_exact * 5
    return ev


def decide(ev: pd.DataFrame) -> pd.DataFrame:
    """Pick the best FPL candidate per PitchAPI player and assign a confidence tier."""
    ev = ev.sort_values(["pid", "score"], ascending=[True, False])
    best = ev.groupby("pid").head(1).set_index("pid")
    second = ev.groupby("pid").nth(1).set_index("pid")
    best["runner_up_score"] = second.score.reindex(best.index)
    best["margin"] = best.score - best.runner_up_score.fillna(0)

    def tier(r) -> str:
        has_stats = r.either_played > 0
        stats_ok = has_stats and r.agree_rate >= MIN_AGREE_RATE
        if r.name_exact and (stats_ok or not has_stats):
            return "A_exact_name"
        if r.name_score >= NAME_FUZZY_OK and stats_ok:
            return "B_fuzzy_name_stats"
        if stats_ok and r.both_played >= MIN_SHARED_FOR_STATS_ONLY:
            return "C_stats_only"          # names differ but stats agree: review
        return "D_review"

    best["tier"] = best.apply(tier, axis=1)
    # one-to-one: an FPL player claimed by several PitchAPI IDs is a conflict to review
    # one-to-one: an FPL player claimed by several PitchAPI IDs is a conflict to review, unless the
    # claimants carry the identical name (PitchAPI occasionally has duplicate IDs for one person)
    confident = best.tier.ne("D_review")
    acc = best[confident]
    names_per_player = acc.groupby("player_id").pa_name.agg(lambda s: s.map(norm_name).nunique())
    clash = names_per_player[names_per_player > 1].index
    best["conflict"] = confident & best.player_id.isin(clash)
    best["confidence"] = (
        best.agree_rate.fillna(0.5) * 0.7 + best.name_score / 100 * 0.3).clip(0, 1).round(3)
    return best.reset_index()


def apply_manual(decisions: pd.DataFrame, path: Path = MANUAL_FILE) -> pd.DataFrame:
    if not path.exists():
        return decisions
    manual = pd.read_csv(path, dtype=str).fillna("")
    manual = manual[manual.source_name == SOURCE]
    d = decisions.set_index("pid")
    for _, m in manual.iterrows():
        pid, target = m.source_player_id, m.canonical_player_id.strip()
        if pid not in d.index:
            continue
        if target.lower() in ("", "none", "unmatched"):
            d.loc[pid, ["tier", "player_id"]] = ["X_manual_unmatched", None]
        else:
            d.loc[pid, ["tier", "player_id", "confidence", "conflict"]] = ["M_manual", int(target), 1.0, False]
    return d.reset_index()


ACCEPT = {"A_exact_name", "B_fuzzy_name_stats", "M_manual"}


def run(conn: psycopg.Connection, write: bool = True) -> dict[str, pd.DataFrame]:
    matches, apps = load_pitchapi(conn)
    matches = matches[matches.status == "finished"]
    seasons = sorted(matches.season.unique())
    fixtures, fpl_rows = load_fpl(conn, seasons)
    teams = match_teams(matches, fixtures)
    fx = match_fixtures(matches, teams, fixtures)
    ev = candidate_evidence(apps, fx, teams, fpl_rows)
    decisions = apply_manual(decide(ev))
    # PitchAPI players who never shared a team-fixture with any FPL player
    all_pids = apps.groupby("pid").name.first()
    missing = all_pids[~all_pids.index.isin(decisions.pid)]
    decisions = pd.concat([decisions, pd.DataFrame({"pid": missing.index, "pa_name": missing.values,
                                                     "tier": "D_review"})], ignore_index=True)
    if write:
        _write(conn, teams, fx, decisions)
    return {"teams": teams, "fixtures": fx, "evidence": ev, "decisions": decisions}


def _num(v):
    return None if v is None or pd.isna(v) else float(v)


def _write(conn, teams, fx, decisions) -> None:
    with db.track_run(conn, "match_pitchapi") as run, conn.transaction():
        run.add("teams", db.upsert(conn, "core.team_source_ids", [{
            "source_name": SOURCE, "source_team_id": r.team_pid, "season": r.season, "team_id": int(r.team_id),
            "source_team_name": r.team_name, "match_method": "fixture_dates", "match_confidence": r.confidence,
        } for r in teams.dropna(subset=["team_id"]).itertuples()], ["source_name", "source_team_id", "season"],
            touch_updated_at=True))
        run.add("fixtures", db.upsert(conn, "core.fixture_source_ids", [{
            "source_name": SOURCE, "source_fixture_id": r.match_id, "fixture_id": int(r.fixture_id),
        } for r in fx.dropna(subset=["fixture_id"]).itertuples()], ["source_name", "source_fixture_id"],
            touch_updated_at=True))
        conn.execute("DELETE FROM core.player_source_ids WHERE source_name = %s", (SOURCE,))
        acc = decisions[decisions.tier.isin(ACCEPT) & ~decisions.conflict.fillna(False).astype(bool)]
        run.add("players", db.insert(conn, "core.player_source_ids", [{
            "canonical_player_id": int(r.player_id), "source_name": SOURCE, "source_player_id": r.pid,
            "season": None, "match_method": r.tier, "match_confidence": float(r.confidence),
            "evidence": {"pa_name": r.pa_name, "name_score": _num(r.name_score), "agree_rate": _num(r.agree_rate),
                         "both_played": _num(r.both_played), "seasons": r.seasons},
        } for r in acc.itertuples()]))


# ---------------------------------------------------------------------------
# defensive back-fill (uses the links above)
# ---------------------------------------------------------------------------
DEFENSIVE_STATS = {"Clearances": "clearances", "Blocks": "blocks", "Interceptions": "interceptions",
                   "Tackles": "tackles", "Recoveries": "recoveries", "Minutes played": "minutes"}


def build_defensive(conn: psycopg.Connection) -> int:
    """PitchAPI per-player defensive actions -> core.player_match_defensive, for linked players."""
    rows = conn.execute("""
        SELECT i.canonical_player_id, fs.fixture_id, r.season, e.k, sum((e.v->'stat'->>'value')::float)
        FROM raw.provider_records r
        JOIN core.fixture_source_ids fs ON fs.source_name = 'pitchapi' AND fs.source_fixture_id = r.entity_id
        CROSS JOIN jsonb_array_elements(r.payload) x
        JOIN core.player_source_ids i ON i.source_name = 'pitchapi' AND i.source_player_id = x->'player'->>'id'
        CROSS JOIN jsonb_array_elements(x->'stats') g
        CROSS JOIN jsonb_each(g->'stats') e(k, v)
        WHERE r.source_name = 'pitchapi' AND r.entity = 'match_players' AND e.k = ANY(%s)
        GROUP BY 1, 2, 3, 4""", (list(DEFENSIVE_STATS),)).fetchall()
    df = pd.DataFrame(rows, columns=["player_id", "fixture_id", "season", "stat", "value"])
    if df.empty:
        return 0
    wide = df.pivot_table(index=["player_id", "fixture_id", "season"], columns="stat", values="value",
                          aggfunc="sum").rename(columns=DEFENSIVE_STATS).reset_index()
    for c in DEFENSIVE_STATS.values():
        if c not in wide:
            wide[c] = None
    # PitchAPI omits a stat when the player recorded none of it; for a player who played, that's an observed 0
    played = wide.minutes.fillna(0) > 0
    for c in ("clearances", "blocks", "interceptions", "tackles", "recoveries"):
        wide.loc[played, c] = wide.loc[played, c].fillna(0)
    wide["cbit"] = wide.clearances + wide.blocks + wide.interceptions + wide.tackles
    wide["cbirt"] = wide.cbit + wide.recoveries
    observed = pd.Timestamp.now(tz="UTC")
    out = [{"player_id": int(r.player_id), "fixture_id": int(r.fixture_id), "season": r.season,
            "source": SOURCE, "observed_at": observed,
            **{c: (None if pd.isna(getattr(r, c)) else int(getattr(r, c)))
               for c in ("minutes", "clearances", "blocks", "interceptions", "tackles", "recoveries", "cbit", "cbirt")}}
           for r in wide.itertuples()]
    with db.track_run(conn, "pitchapi_defensive") as run, conn.transaction():
        run.add("player_match_defensive", db.upsert(conn, "core.player_match_defensive", out,
                                                    ["player_id", "fixture_id", "source"]))
    return len(out)


REFERENCE_SEASON = "2025-26"   # first season FPL scored defensive contributions; today's definitions


def build_adjustments(conn: psycopg.Connection, reference: str = REFERENCE_SEASON) -> pd.DataFrame:
    """Per-season factors that rescale PitchAPI defensive stats to `reference`-season definitions.

    Seasons at or after the reference keep factor 1 (current definitions)."""
    stats = ["clearances", "blocks", "interceptions", "tackles", "recoveries"]
    means = pd.DataFrame(conn.execute(f"""
        SELECT season, {', '.join(f'sum({s})::float / (count(DISTINCT fixture_id) * 2) AS {s}' for s in stats)}
        FROM core.player_match_defensive WHERE source = %s GROUP BY season""", (SOURCE,)).fetchall(),
        columns=["season", *stats]).set_index("season")
    ref = means.loc[reference]
    rows = []
    for season, m in means.iterrows():
        for s in stats:
            factor = 1.0 if season >= reference else float(ref[s] / m[s])
            rows.append({"source": SOURCE, "stat": s, "season": season, "reference_season": reference,
                         "season_mean": round(float(m[s]), 3), "reference_mean": round(float(ref[s]), 3),
                         "factor": round(factor, 4)})
    with conn.transaction():
        db.upsert(conn, "core.stat_season_adjustments", rows, ["source", "stat", "season"])
    return pd.DataFrame(rows)
