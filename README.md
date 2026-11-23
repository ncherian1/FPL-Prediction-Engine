# FPL Prediction Engine

A data warehouse for Fantasy Premier League predictions: it ingests data from several providers into PostgreSQL. This repository currently covers Phase 1, the data infrastructure. Feature engineering, models and the app will be built on top of this warehouse.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env              # add the optional PitchAPI / StrikerAPI keys here
docker compose up -d --wait       # local Postgres 16 (or point DATABASE_URL at Supabase)
.venv/bin/python -m fpl_engine migrate
.venv/bin/python -m fpl_engine backfill   # one-off: 2016-17 → now from the keyless sources (~10 min)
.venv/bin/python -m fpl_engine report     # freshness + missing-field reports
```

## Commands (`python -m fpl_engine …`)

| Command | What it does | Cadence |
|---|---|---|
| `sync-fpl` | FPL bootstrap (players, teams, gameweeks, prices, ownership, injuries) + fixtures | every 6h |
| `sync-histories [ids…]` | FPL `element-summary` for each player: per-fixture stats and past-season totals | daily |
| `sync-live [--gameweek N]` | `/event/{gw}/live/` | every 1–5 min during matches |
| `import-vaastav [2019-20 …]` | historical FPL seasons from the Vaastav archive | one-off |
| `import-football-data [2023-24 …]` | Football-Data.co.uk results, shots, cards, corners, odds and team xG | daily (current season) |
| `sync-pitchapi [--season 2025-26] [--limit N]` | PitchAPI matches, shots, player stats, lineups and advanced stats | daily (needs key) |
| `strikerapi-probe [--year 2025]` | about 3 credits: checks StrikerAPI coverage | manual (needs key) |
| `strikerapi-shots ARS LIV … [--year]` | 10 credits per team: shot maps with xG | manual (needs key) |
| `daily` | runs everything above that is scheduled daily, then writes reports | daily |
| `report` | freshness by job, % non-null per stat per season, consolidated current-player table | any time |

`.github/workflows/ingest.yml` runs `sync-fpl` every 6h and `daily` once a day once the `DATABASE_URL` repo secret is set.

## Hosting (Supabase)

The production database is Supabase Postgres. Use the **Session pooler** URI from Connect → Direct connection string → Session pooler; the direct connection is IPv6-only. Set it as `DATABASE_URL` in `.env` and as a GitHub Actions secret. To copy a local warehouse up:

```bash
scripts/push_to_remote.sh "postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres"
```

## Data sources

| Source | Key? | Status | Lands in |
|---|---|---|---|
| Official FPL API | no | ✅ live | `core.*` (authoritative for FPL data) |
| Vaastav archive (2016-17 → 2025-26) | no | ✅ | `core.*` with `source='vaastav'` |
| Football-Data.co.uk | no | ✅ | `raw.football_data_matches` |
| PitchAPI (2021-22 → now) | free key | ✅ shots + xG, player match stats, lineups, team advanced stats | `raw.provider_records` |
| StrikerAPI | free key, 500 credits/mo | ✅ 2025-26 shot maps loaded (201 credits) | `raw.provider_records` |
| Understat | no | ✖ skipped: PitchAPI covers xG | — |

The FPL API publishes per-match xG, xA, xGI, xGC, CBI, recoveries, tackles and `defensive_contribution` itself, so these exist without a third-party provider from 2022-23 (xG) and 2025-26 (defensive stats). Key passes and big chances are only in the FPL archive for 2016-17 to 2018-19. Run `report` to see the exact coverage.

## Database layout

See **[docs/schema.md](docs/schema.md)** for the entity-relationship diagram and a guide to every table.

`sql/001_init.sql` holds the full schema and its comments.

- **raw**: `source_snapshots` records every HTTP request: endpoint, params, status, sha256 and archive path. Response bodies are stored gzipped and content-addressed under `ARCHIVE_DIR`. `fpl_live_gameweek`, `football_data_matches` and `provider_records` hold provider data whose coverage hasn't been validated yet.
- **core**: `players`, `teams`, `team_seasons`, `player_seasons`, `player_source_ids`, `gameweeks`, `fixtures`, `player_match_stats`, `player_season_history`, `player_price_history` (append-only), `player_availability` (append-only, written only when something changes) and `lineups`. There are also the views `player_defensive_stats`, `goalkeeper_stats` and `v_latest_availability`.
- **features**, **predictions**: reserved for Phases 2–4.
- **ops**: `ingestion_runs` stores the status, row counts and errors of every job. This is how the last successful update is tracked.

Rules this layout enforces:
- **Canonical IDs.** `player_id` is the FPL element `code` and `team_id` is the FPL team `code`; both stay stable across seasons. The per-season element and team IDs are mapped in `player_seasons` / `team_seasons` / `player_source_ids`.
- **NULL means unknown, 0 means an observed zero.** A stat a season doesn't publish stays NULL. An FPL player with no injury flag gets `availability_probability = NULL`, not 100%.
- **No overwriting of history.** Prices and availability are append-only, and raw payloads are archived.
- **Retries and rate limits.** Every request goes through `fpl_engine/http.py`: retries with exponential backoff (respecting `Retry-After`), per-provider rate limiting, and TTL caching for immutable historical files.

## Tests

```bash
.venv/bin/python -m pytest -q
```
