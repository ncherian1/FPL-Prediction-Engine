-- Model outputs (spec §3 layer 4). Append-only: every prediction run is kept, never overwritten.
CREATE TABLE IF NOT EXISTS predictions.model_runs (
    run_id            bigserial PRIMARY KEY,
    created_at        timestamptz NOT NULL DEFAULT now(),
    model_name        text NOT NULL,
    model_version     text NOT NULL,
    target_season     text NOT NULL,
    target_gameweek   integer NOT NULL,
    trained_through   timestamptz,
    n_train_rows      integer,
    config            jsonb,
    notes             text
);

CREATE TABLE IF NOT EXISTS predictions.player_fixture_predictions (
    run_id            bigint NOT NULL REFERENCES predictions.model_runs(run_id),
    player_id         integer NOT NULL REFERENCES core.players(player_id),
    fixture_id        bigint NOT NULL REFERENCES core.fixtures(fixture_id),
    season            text NOT NULL,
    gameweek          integer NOT NULL,
    xpts              numeric NOT NULL,          -- expected points for this fixture
    xmins             numeric,
    p_plays           numeric,
    p_60              numeric,
    gw_xpts           numeric,                   -- sum over the player's fixtures this gameweek (double gameweeks)
    p_return          numeric,                   -- P(gameweek points >= 6)
    p_haul            numeric,                   -- P(gameweek points >= 10)
    p10 numeric, p50 numeric, p90 numeric,       -- gameweek points percentiles
    breakdown         jsonb,                     -- expected points by component
    availability      jsonb,                     -- FPL status / chance of playing applied
    penalty_adjustment numeric,                  -- extra goals per 90 from the penalty-taker overlay
    PRIMARY KEY (run_id, player_id, fixture_id)
);
CREATE INDEX IF NOT EXISTS pfp_gw_idx ON predictions.player_fixture_predictions (season, gameweek, run_id);

CREATE TABLE IF NOT EXISTS predictions.backtest_metrics (
    id          bigserial PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    label       text NOT NULL,
    model       text NOT NULL,
    population  text NOT NULL,
    metric      text NOT NULL,
    value       numeric,
    config      jsonb
);

-- Latest run's predictions per gameweek, one row per player (double gameweeks summed).
CREATE OR REPLACE VIEW predictions.v_latest_gameweek AS
WITH latest AS (
    SELECT DISTINCT ON (target_season, target_gameweek) run_id, target_season, target_gameweek
    FROM predictions.model_runs ORDER BY target_season, target_gameweek, created_at DESC
)
SELECT p.season, p.gameweek, p.player_id, pl.web_name, ps.position, t.short_name AS team,
       count(*) AS fixtures, sum(p.xpts) AS xpts, max(p.p_return) AS p_return, max(p.p_haul) AS p_haul,
       max(p.p10) AS p10, max(p.p50) AS p50, max(p.p90) AS p90, sum(p.xmins) AS xmins, min(p.p_plays) AS p_plays,
       p.run_id
FROM predictions.player_fixture_predictions p
JOIN latest l ON l.run_id = p.run_id
JOIN core.players pl USING (player_id)
LEFT JOIN core.player_seasons ps ON ps.player_id = p.player_id AND ps.season = p.season
LEFT JOIN core.teams t ON t.team_id = ps.team_id
GROUP BY p.season, p.gameweek, p.player_id, pl.web_name, ps.position, t.short_name, p.run_id;
