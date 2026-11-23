"""Pure transformation helpers shared by the FPL API and Vaastav importers."""
from __future__ import annotations

import hashlib
import math
from datetime import datetime
from typing import Any

# core.player_match_stats column -> FPL field name (FPL API element-summary history and
# Vaastav merged_gw.csv share these names).
MATCH_STAT_FIELDS: dict[str, str] = {
    "minutes": "minutes",
    "starts": "starts",
    "goals": "goals_scored",
    "assists": "assists",
    "clean_sheet": "clean_sheets",
    "goals_conceded": "goals_conceded",
    "own_goals": "own_goals",
    "penalties_saved": "penalties_saved",
    "penalties_missed": "penalties_missed",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "saves": "saves",
    "bonus": "bonus",
    "bps": "bps",
    "fpl_points": "total_points",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",
    "ict_index": "ict_index",
    "expected_goals": "expected_goals",
    "expected_assists": "expected_assists",
    "expected_goal_involvements": "expected_goal_involvements",
    "expected_goals_conceded": "expected_goals_conceded",
    "clearances_blocks_interceptions": "clearances_blocks_interceptions",
    "recoveries": "recoveries",
    "tackles": "tackles",
    "defensive_contribution": "defensive_contribution",
    "key_passes": "key_passes",
    "big_chances_created": "big_chances_created",
    "big_chances_missed": "big_chances_missed",
    "fouls": "fouls",
    "selected": "selected",
    "transfers_in": "transfers_in",
    "transfers_out": "transfers_out",
    "transfers_balance": "transfers_balance",
}
NUMERIC_FIELDS = {
    "influence", "creativity", "threat", "ict_index", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
}

FPL_STATUS = {
    "a": "available",
    "d": "doubtful",
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",      # left club / long-term unavailable
    "n": "not_in_squad",     # e.g. on loan, ineligible
}


def season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_start_year(season: str) -> int:
    return int(season[:4])


def is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def to_int(v: Any) -> int | None:
    if is_missing(v):
        return None
    if isinstance(v, bool):
        return int(v)
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def to_float(v: Any) -> float | None:
    if is_missing(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_bool(v: Any) -> bool | None:
    if is_missing(v):
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "t", "yes"}:
        return True
    if s in {"false", "0", "f", "no"}:
        return False
    return None


def to_ts(v: Any) -> datetime | None:
    if is_missing(v):
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


def price(v: Any) -> float | None:
    """FPL prices are integers in tenths (e.g. 105 -> 10.5)."""
    n = to_int(v)
    return None if n is None else n / 10


def match_stats(row: dict[str, Any]) -> dict[str, Any]:
    """Map an FPL-style per-fixture row onto core.player_match_stats stat columns.

    A field absent from the source row becomes None (unknown), never 0."""
    out: dict[str, Any] = {}
    for col, src in MATCH_STAT_FIELDS.items():
        v = row.get(src)
        out[col] = to_float(v) if col in NUMERIC_FIELDS else to_int(v)
    out["price"] = price(row.get("value"))
    return out


def availability_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join("" if p is None else str(p) for p in parts).encode()).hexdigest()
