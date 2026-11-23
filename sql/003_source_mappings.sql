-- Cross-provider identity mapping (teams, fixtures) + match evidence for players.
CREATE TABLE IF NOT EXISTS core.team_source_ids (
    source_name      text NOT NULL,
    source_team_id   text NOT NULL,
    season           text NOT NULL,
    team_id          integer NOT NULL REFERENCES core.teams(team_id),
    source_team_name text,
    match_method     text NOT NULL,          -- fixture_dates | manual
    match_confidence numeric(4,3),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source_name, source_team_id, season)
);

CREATE TABLE IF NOT EXISTS core.fixture_source_ids (
    source_name       text NOT NULL,
    source_fixture_id text NOT NULL,
    fixture_id        bigint NOT NULL REFERENCES core.fixtures(fixture_id),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source_name, source_fixture_id),
    UNIQUE (source_name, fixture_id)
);

ALTER TABLE core.player_source_ids ADD COLUMN IF NOT EXISTS evidence jsonb;
