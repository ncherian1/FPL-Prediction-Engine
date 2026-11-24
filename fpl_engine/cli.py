"""Command line entry point:  python -m fpl_engine <command> [options]"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import db, reports
from .sources import fpl_api, football_data, pitchapi, strikerapi, vaastav
from .transform import season_start_year

log = logging.getLogger("fpl_engine")


def _current(conn) -> str:
    season = fpl_api.current_season(conn)
    if not season:
        season = fpl_api.sync_bootstrap(conn)
    return season


def cmd_migrate(conn, args):
    ran = db.migrate(conn)
    print("applied:", ran or "nothing (up to date)")


def cmd_sync_fpl(conn, args):
    """Every 6h: players, teams, gameweeks, prices, ownership, injuries, fixtures."""
    season = fpl_api.sync_bootstrap(conn)
    fpl_api.sync_fixtures(conn, season)
    if args.with_history:
        fpl_api.sync_player_histories(conn, season)


def cmd_sync_histories(conn, args):
    fpl_api.sync_player_histories(conn, _current(conn), element_ids=args.elements or None)


def cmd_sync_live(conn, args):
    fpl_api.sync_live(conn, args.gameweek)


def cmd_import_vaastav(conn, args):
    current = season_start_year(_current(conn))
    seasons = args.seasons or vaastav.available_seasons(current)
    for s in seasons:
        vaastav.import_season(conn, s)


def cmd_import_football_data(conn, args):
    current = season_start_year(_current(conn))
    years = [int(s[:4]) for s in args.seasons] if args.seasons else range(args.since, current + 1)
    for y in years:
        football_data.import_season(conn, y, is_current=(y == current))


def cmd_sync_pitchapi(conn, args):
    season = args.season or _current(conn)
    pitchapi.sync_season(conn, season, refresh=args.refresh, limit=args.limit)
    print(reports.provider_coverage(conn, "pitchapi").to_string(index=False))


def cmd_strikerapi_probe(conn, args):
    year = args.year or season_start_year(_current(conn))
    print(strikerapi.probe(conn, year))


def cmd_strikerapi_shots(conn, args):
    year = args.year or season_start_year(_current(conn))
    print(strikerapi.sync_shot_maps(conn, year, args.teams, reserve=args.reserve))


def cmd_daily(conn, args):
    """Daily job: FPL core + per-player histories + current Football-Data season (+PitchAPI if keyed)."""
    season = fpl_api.sync_bootstrap(conn)
    fpl_api.sync_fixtures(conn, season)
    fpl_api.sync_player_histories(conn, season)
    football_data.import_season(conn, season_start_year(season), is_current=True)
    from .config import get_settings
    if get_settings().pitchapi_key:
        pitchapi.sync_season(conn, season)
        from . import matching
        matching.run(conn, write=True)          # link any new players / fixtures
        matching.build_defensive(conn)
        matching.build_adjustments(conn)
    from . import live
    live.predict_gameweek(conn)
    paths = reports.write_all(conn)
    print("reports:", *paths.values(), sep="\n  ")


def cmd_backfill(conn, args):
    """One-off: full historical load from every keyless source."""
    season = fpl_api.sync_bootstrap(conn)
    fpl_api.sync_fixtures(conn, season)
    current = season_start_year(season)
    for s in vaastav.available_seasons(current):
        vaastav.import_season(conn, s)
    for y in range(vaastav.FIRST_SEASON, current + 1):
        football_data.import_season(conn, y, is_current=(y == current))
    fpl_api.sync_player_histories(conn, season)


def cmd_seed_scoring_rules(conn, args):
    from .scoring import seed_rules
    with fpl_api.fetcher(conn) as f:
        scoring = f.get("/bootstrap-static/").json().get("game_config", {}).get("scoring")
    print("scoring rules stored for:", ", ".join(seed_rules(conn, scoring)))


def cmd_match_players(conn, args):
    from . import matching
    r = matching.run(conn, write=True)
    d = r["decisions"]
    print(d.tier.value_counts().to_string())
    out = reports.REPORT_DIR / "player_matching_review.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    review = d[(~d.tier.isin(matching.ACCEPT) & d.tier.ne("X_manual_unmatched"))
               | d.conflict.fillna(False).astype(bool)]
    review.to_csv(out, index=False)
    print(f"review list ({len(review)}):", out)
    print("defensive rows from PitchAPI:", matching.build_defensive(conn))
    matching.build_adjustments(conn)


def cmd_predict(conn, args):
    from . import live
    out = live.predict_gameweek(conn, gameweek=args.gameweek, n_sims=args.sims)
    top = out.groupby("player_id").agg(xpts=("gw_xpts", "first"), p_haul=("p_haul", "first")).nlargest(15, "xpts")
    names = dict(conn.execute("SELECT player_id, web_name FROM core.players").fetchall())
    print(top.assign(name=top.index.map(names)).round(2).to_string())


def cmd_report(conn, args):
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n== Freshness ==")
    print(reports.freshness(conn).to_string(index=False))
    cov = reports.field_coverage(conn)
    print("\n== % non-null in core.player_match_stats by season (selected fields) ==")
    print(cov[["season", "n_rows", "minutes", "expected_goals", "defensive_contribution", "recoveries",
               "key_passes", "starts", "price"]].to_string(index=False))
    players = reports.consolidated_players(conn)
    print(f"\n== Current-season players: {len(players)} ==")
    print(players.isna().sum().rename("missing").to_frame().T.to_string())
    paths = reports.write_all(conn)
    print("\nCSV reports:", *paths.values(), sep="\n  ")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fpl_engine")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("migrate", help="create/upgrade the database schema").set_defaults(fn=cmd_migrate)
    s = sub.add_parser("sync-fpl", help="FPL bootstrap + fixtures (run every 6h)")
    s.add_argument("--with-history", action="store_true", help="also fetch per-player histories (~700 requests)")
    s.set_defaults(fn=cmd_sync_fpl)
    s = sub.add_parser("sync-histories", help="FPL element-summary for every (or given) player")
    s.add_argument("elements", nargs="*", type=int)
    s.set_defaults(fn=cmd_sync_histories)
    s = sub.add_parser("sync-live", help="live gameweek stats (run every 1-5 min during matches)")
    s.add_argument("--gameweek", type=int)
    s.set_defaults(fn=cmd_sync_live)
    s = sub.add_parser("import-vaastav", help="historical FPL seasons from the Vaastav archive")
    s.add_argument("seasons", nargs="*", help="e.g. 2023-24 (default: 2016-17 .. last season)")
    s.set_defaults(fn=cmd_import_vaastav)
    s = sub.add_parser("import-football-data", help="Football-Data.co.uk Premier League CSVs")
    s.add_argument("seasons", nargs="*", help="e.g. 2023-24")
    s.add_argument("--since", type=int, default=2016)
    s.set_defaults(fn=cmd_import_football_data)
    s = sub.add_parser("sync-pitchapi", help="PitchAPI matches + per-match detail (needs key)")
    s.add_argument("--season", help="e.g. 2025-26 (default: current)")
    s.add_argument("--limit", type=int, help="max matches to detail (for testing)")
    s.add_argument("--refresh", action="store_true")
    s.set_defaults(fn=cmd_sync_pitchapi)
    s = sub.add_parser("strikerapi-probe", help="~3 credits: teams + games for a season (needs key)")
    s.add_argument("--year", type=int, help="season start year, e.g. 2025")
    s.set_defaults(fn=cmd_strikerapi_probe)
    s = sub.add_parser("strikerapi-shots", help="10 credits per team (~201 per season): shot maps with xG (needs key)")
    s.add_argument("teams", nargs="*", help="team codes, e.g. ARS LIV (default: all clubs that season)")
    s.add_argument("--year", type=int)
    s.add_argument("--reserve", type=int, default=50, help="credits to keep unspent")
    s.set_defaults(fn=cmd_strikerapi_shots)
    sub.add_parser("daily", help="scheduled daily job").set_defaults(fn=cmd_daily)
    sub.add_parser("backfill", help="one-off full historical load (keyless sources)").set_defaults(fn=cmd_backfill)
    sub.add_parser("report", help="freshness + missing-field reports").set_defaults(fn=cmd_report)
    s = sub.add_parser("predict", help="predict the next (or given) gameweek and store the run")
    s.add_argument("--gameweek", type=int)
    s.add_argument("--sims", type=int, default=10_000)
    s.set_defaults(fn=cmd_predict)
    sub.add_parser("match-players", help="link PitchAPI teams/fixtures/players to FPL IDs").set_defaults(
        fn=cmd_match_players)
    sub.add_parser("seed-scoring-rules", help="store FPL scoring rules for every season").set_defaults(
        fn=cmd_seed_scoring_rules)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    with db.connect() as conn:
        if args.cmd != "migrate":
            db.migrate(conn)
        args.fn(conn, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
