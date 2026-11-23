-- FPL Prediction Engine — initial schema (Phase 1: data infrastructure)
--
-- Layers
--   raw          provider payload metadata + provider data landed as-is (unvalidated fields)
--   core         standardised, consolidated entities keyed on canonical IDs
--   features     derived predictive features            (Phase 2 — empty for now)
--   predictions  model outputs and evaluation           (Phase 3/4 — empty for now)
--   ops          ingestion run bookkeeping / freshness
--
-- Canonical IDs
--   core.players.player_id = FPL element "code"  (stable across seasons; the per-season
--                            element "id" lives in core.player_source_ids / player_seasons)
--   core.teams.team_id     = FPL team "code"     (stable across seasons; 3 = Arsenal)
--   core.fixtures.fixture_id is a surrogate; (season, fpl_fixture_id) is the natural key.
--
-- NULL means "unknown / not provided by the source". Zero means a real observed zero.

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS predictions;
CREATE SCHEMA IF NOT EXISTS ops;

-- ---------------------------------------------------------------------------
-- ops
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ops.schema_migrations (
    version     text PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ops.ingestion_runs (
    id            bigserial PRIMARY KEY,
    job           text NOT NULL,
    started_at    timestamptz NOT NULL DEFAULT now(),
    finished_at   timestamptz,
    status        text NOT NULL DEFAULT 'running',   -- running | success | failed
    rows_written  integer,
    details       jsonb,
    error_message text
);
CREATE INDEX IF NOT EXISTS ingestion_runs_job_idx ON ops.ingestion_runs (job, started_at DESC);

-- ---------------------------------------------------------------------------
-- raw
-- ---------------------------------------------------------------------------
-- Every HTTP request made to a provider. Payload bodies are archived gzipped and
-- content-addressed on disk (archive_path); identical bodies are stored once.
CREATE TABLE IF NOT EXISTS raw.source_snapshots (
    id                 bigserial PRIMARY KEY,
    source_name        text NOT NULL,
    endpoint           text NOT NULL,
    params             jsonb,
    snapshot_timestamp timestamptz NOT NULL DEFAULT now(),
    http_status        integer,
    success            boolean NOT NULL,
    error_message      text,
    content_sha256     text,
    archive_path       text,
    bytes              integer,
    from_cache         boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS source_snapshots_lookup_idx
    ON raw.source_snapshots (source_name, endpoint, snapshot_timestamp DESC);

-- Live gameweek data from /event/{gw}/live/ — latest observation per player/gameweek.
CREATE TABLE IF NOT EXISTS raw.fpl_live_gameweek (
    season        text NOT NULL,
    gameweek      integer NOT NULL,
    fpl_element_id integer NOT NULL,
    stats         jsonb NOT NULL,
    explain       jsonb,
    observed_at   timestamptz NOT NULL,
    snapshot_id   bigint REFERENCES raw.source_snapshots(id),
    PRIMARY KEY (season, gameweek, fpl_element_id)
);

-- Football-Data.co.uk match rows. Column sets vary by season, so the full row is kept
-- in `data`; frequently used fields are typed alongside. Team names are provider names.
CREATE TABLE IF NOT EXISTS raw.football_data_matches (
    season        text NOT NULL,
    division      text NOT NULL,
    match_date    date NOT NULL,
    home_team     text NOT NULL,
    away_team     text NOT NULL,
    fthg integer, ftag integer, ftr text,
    hthg integer, htag integer,
    home_shots integer, away_shots integer,
    home_shots_on_target integer, away_shots_on_target integer,
    home_fouls integer, away_fouls integer,
    home_corners integer, away_corners integer,
    home_yellow integer, away_yellow integer,
    home_red integer, away_red integer,
    home_xg numeric, away_xg numeric,
    data          jsonb NOT NULL,
    observed_at   timestamptz NOT NULL,
    snapshot_id   bigint REFERENCES raw.source_snapshots(id),
    PRIMARY KEY (season, division, match_date, home_team, away_team)
);

-- Generic landing table for supplementary providers (PitchAPI, StrikerAPI, ...) whose
-- field coverage is still being validated (spec rule 10). Promote to typed core tables
-- once coverage is confirmed.
CREATE TABLE IF NOT EXISTS raw.provider_records (
    source_name   text NOT NULL,
    entity        text NOT NULL,          -- e.g. match, match_players, match_shots, lineups
    entity_id     text NOT NULL,
    season        text,
    parent_id     text,
    payload       jsonb NOT NULL,
    observed_at   timestamptz NOT NULL,
    snapshot_id   bigint REFERENCES raw.source_snapshots(id),
    PRIMARY KEY (source_name, entity, entity_id)
);
CREATE INDEX IF NOT EXISTS provider_records_season_idx ON raw.provider_records (source_name, entity, season);

-- ---------------------------------------------------------------------------
-- core
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.seasons (
    season      text PRIMARY KEY,           -- '2026-27'
    start_year  integer NOT NULL UNIQUE,
    is_current  boolean NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS core.teams (
    team_id     integer PRIMARY KEY,        -- FPL team code
    team_name   text NOT NULL,
    short_name  text,
    stadium     text,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS core.team_seasons (
    season                 text NOT NULL REFERENCES core.seasons(season),
    fpl_team_id            integer NOT NULL,      -- 1..20, season-specific
    team_id                integer NOT NULL REFERENCES core.teams(team_id),
    strength               integer,
    strength_overall_home  integer,
    strength_overall_away  integer,
    strength_attack_home   integer,
    strength_attack_away   integer,
    strength_defence_home  integer,
    strength_defence_away  integer,
    source                 text NOT NULL,
    updated_at             timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (season, fpl_team_id),
    UNIQUE (season, team_id)
);

CREATE TABLE IF NOT EXISTS core.players (
    player_id      integer PRIMARY KEY,       -- FPL element code
    first_name     text,
    second_name    text,
    web_name       text,
    position       text,                      -- GKP/DEF/MID/FWD (latest known)
    team_id        integer REFERENCES core.teams(team_id),  -- latest known
    date_of_birth  date,
    nationality    text,                      -- not provided by FPL; NULL until sourced
    current_price  numeric(4,1),
    ownership      numeric(5,2),              -- selected_by_percent
    opta_code      text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS core.player_seasons (
    season          text NOT NULL REFERENCES core.seasons(season),
    fpl_element_id  integer NOT NULL,
    player_id       integer NOT NULL REFERENCES core.players(player_id),
    team_id         integer REFERENCES core.teams(team_id),
    position        text,
    start_price     numeric(4,1),
    source          text NOT NULL,
    updated_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (season, fpl_element_id),
    UNIQUE (season, player_id)
);

CREATE TABLE IF NOT EXISTS core.player_source_ids (
    id                  bigserial PRIMARY KEY,
    canonical_player_id integer NOT NULL REFERENCES core.players(player_id),
    source_name         text NOT NULL,
    source_player_id    text NOT NULL,
    season              text,                -- NULL when the source ID is season-independent
    match_method        text NOT NULL DEFAULT 'exact',  -- exact | name_team | manual
    match_confidence    numeric(4,3),
    valid_from          timestamptz NOT NULL DEFAULT now(),
    valid_to            timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS player_source_ids_uq
    ON core.player_source_ids (source_name, source_player_id, COALESCE(season, ''));

CREATE TABLE IF NOT EXISTS core.gameweeks (
    season          text NOT NULL REFERENCES core.seasons(season),
    gameweek        integer NOT NULL,
    name            text,
    deadline_time   timestamptz,
    finished        boolean,
    data_checked    boolean,
    is_current      boolean,
    is_next         boolean,
    average_score   integer,
    highest_score   integer,
    source          text NOT NULL,
    updated_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (season, gameweek)
);

CREATE TABLE IF NOT EXISTS core.fixtures (
    fixture_id         bigserial PRIMARY KEY,
    season             text NOT NULL REFERENCES core.seasons(season),
    fpl_fixture_id     integer NOT NULL,
    fpl_code           bigint,
    gameweek           integer,              -- NULL while unscheduled / postponed
    home_team_id       integer NOT NULL REFERENCES core.teams(team_id),
    away_team_id       integer NOT NULL REFERENCES core.teams(team_id),
    kickoff_time       timestamptz,
    started            boolean,
    finished           boolean,
    finished_provisional boolean,
    postponed          boolean,              -- derived: no gameweek/kickoff assigned
    home_score         integer,
    away_score         integer,
    official_home_fdr  integer,
    official_away_fdr  integer,
    source             text NOT NULL,
    updated_at         timestamptz NOT NULL DEFAULT now(),
    UNIQUE (season, fpl_fixture_id)
);
CREATE INDEX IF NOT EXISTS fixtures_gw_idx ON core.fixtures (season, gameweek);

-- One row per player per fixture (double gameweeks produce two rows).
-- FPL-sourced fields only. Provider stats (shots, key passes, ...) for seasons where FPL
-- does not publish them are NULL here and will be joined from validated provider tables.
CREATE TABLE IF NOT EXISTS core.player_match_stats (
    id                     bigserial PRIMARY KEY,
    player_id              integer NOT NULL REFERENCES core.players(player_id),
    fixture_id             bigint  NOT NULL REFERENCES core.fixtures(fixture_id),
    season                 text NOT NULL,
    gameweek               integer,
    team_id                integer REFERENCES core.teams(team_id),
    opponent_team_id       integer REFERENCES core.teams(team_id),
    was_home               boolean,
    kickoff_time           timestamptz,
    minutes                integer,
    starts                 integer,
    goals                  integer,
    assists                integer,
    clean_sheet            integer,
    goals_conceded         integer,
    own_goals              integer,
    penalties_saved        integer,
    penalties_missed       integer,
    yellow_cards           integer,
    red_cards              integer,
    saves                  integer,
    bonus                  integer,
    bps                    integer,
    fpl_points             integer,
    influence              numeric,
    creativity             numeric,
    threat                 numeric,
    ict_index              numeric,
    expected_goals         numeric,
    expected_assists       numeric,
    expected_goal_involvements numeric,
    expected_goals_conceded numeric,
    clearances_blocks_interceptions integer,
    recoveries             integer,
    tackles                integer,
    defensive_contribution integer,
    -- only published by FPL in 2016-17 .. 2018-19; NULL elsewhere
    shots                  integer,
    key_passes             integer,
    big_chances_created    integer,
    big_chances_missed     integer,
    fouls                  integer,
    -- market state at the time of the fixture
    price                  numeric(4,1),
    selected               bigint,
    transfers_in           integer,
    transfers_out          integer,
    transfers_balance      integer,
    source                 text NOT NULL,
    observed_at            timestamptz NOT NULL,
    UNIQUE (player_id, fixture_id)
);
CREATE INDEX IF NOT EXISTS pms_season_gw_idx ON core.player_match_stats (season, gameweek);

CREATE OR REPLACE VIEW core.player_defensive_stats AS
SELECT player_id, fixture_id, season, gameweek,
       clearances_blocks_interceptions, recoveries, tackles, defensive_contribution,
       source, observed_at
FROM core.player_match_stats;

CREATE OR REPLACE VIEW core.goalkeeper_stats AS
SELECT s.player_id, s.fixture_id, s.season, s.gameweek, s.saves, s.penalties_saved,
       s.goals_conceded, s.clean_sheet, s.expected_goals_conceded, s.source, s.observed_at
FROM core.player_match_stats s
JOIN core.player_seasons ps ON ps.player_id = s.player_id AND ps.season = s.season
WHERE ps.position = 'GKP';

-- Season totals per player as published in FPL element-summary history_past.
CREATE TABLE IF NOT EXISTS core.player_season_history (
    player_id     integer NOT NULL REFERENCES core.players(player_id),
    season        text NOT NULL,
    start_price   numeric(4,1),
    end_price     numeric(4,1),
    total_points  integer,
    minutes       integer,
    data          jsonb NOT NULL,
    observed_at   timestamptz NOT NULL,
    PRIMARY KEY (player_id, season)
);

-- Append-only market observations.
CREATE TABLE IF NOT EXISTS core.player_price_history (
    id                 bigserial PRIMARY KEY,
    player_id          integer NOT NULL REFERENCES core.players(player_id),
    season             text NOT NULL,
    price              numeric(4,1),
    ownership          numeric(5,2),
    transfers_in_event integer,
    transfers_out_event integer,
    transfers_in       integer,
    transfers_out      integer,
    cost_change_event  integer,
    observed_at        timestamptz NOT NULL,
    source             text NOT NULL,
    snapshot_id        bigint REFERENCES raw.source_snapshots(id)
);
CREATE INDEX IF NOT EXISTS price_history_player_idx ON core.player_price_history (player_id, observed_at DESC);

-- Append-only availability observations; a new row is written only when the content changes.
CREATE TABLE IF NOT EXISTS core.player_availability (
    id                        bigserial PRIMARY KEY,
    player_id                 integer NOT NULL REFERENCES core.players(player_id),
    source                    text NOT NULL,
    status_code               text,          -- raw provider code (FPL: a/d/i/s/u/n)
    injury_status             text,          -- normalised: available/doubtful/injured/suspended/unavailable/not_in_squad
    chance_of_playing_this_round integer,    -- NULL = no statement (not "100%")
    chance_of_playing_next_round integer,
    availability_probability  numeric(4,3),
    injury_description        text,
    expected_return_date      date,
    suspension_status         text,
    news                      text,
    news_added                timestamptz,
    confidence                numeric(4,3),
    content_hash              text NOT NULL,
    observed_at               timestamptz NOT NULL,
    snapshot_id               bigint REFERENCES raw.source_snapshots(id)
);
CREATE INDEX IF NOT EXISTS availability_player_idx ON core.player_availability (player_id, source, observed_at DESC);

CREATE TABLE IF NOT EXISTS core.lineups (
    id                bigserial PRIMARY KEY,
    fixture_id        bigint NOT NULL REFERENCES core.fixtures(fixture_id),
    player_id         integer NOT NULL REFERENCES core.players(player_id),
    is_starting       boolean,
    is_substitute     boolean,
    lineup_confirmed  boolean NOT NULL,
    expected_minutes  numeric,
    source            text NOT NULL,
    observed_at       timestamptz NOT NULL
);

-- Latest availability per player per source.
CREATE OR REPLACE VIEW core.v_latest_availability AS
SELECT DISTINCT ON (player_id, source) *
FROM core.player_availability
ORDER BY player_id, source, observed_at DESC;
