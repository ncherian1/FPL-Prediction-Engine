-- Definition-drift adjustment for defensive stats (decided 2026-11-23, notebook 01b §7).
-- Provider stat definitions shifted between 2023 and 2025 (e.g. recoveries ~61 -> ~47 per team-match).
-- factor = league mean per team-match in the reference season / mean in this season, per stat.
-- Raw counts are never modified; adjusted values are computed in the view.
CREATE TABLE IF NOT EXISTS core.stat_season_adjustments (
    source            text    NOT NULL,
    stat              text    NOT NULL,
    season            text    NOT NULL,
    reference_season  text    NOT NULL,
    season_mean       numeric NOT NULL,
    reference_mean    numeric NOT NULL,
    factor            numeric NOT NULL,
    method            text    NOT NULL DEFAULT 'mean_ratio_per_team_match',
    computed_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, stat, season)
);

DROP VIEW IF EXISTS core.v_player_match_stats;
CREATE VIEW core.v_player_match_stats AS
WITH adj AS (
    SELECT season,
           max(factor) FILTER (WHERE stat = 'clearances')    AS f_cl,
           max(factor) FILTER (WHERE stat = 'blocks')        AS f_bl,
           max(factor) FILTER (WHERE stat = 'interceptions') AS f_it,
           max(factor) FILTER (WHERE stat = 'tackles')       AS f_tk,
           max(factor) FILTER (WHERE stat = 'recoveries')    AS f_rc
    FROM core.stat_season_adjustments WHERE source = 'pitchapi' GROUP BY season
)
SELECT s.*,
       ps.position,
       COALESCE(s.clearances_blocks_interceptions, d.clearances + d.blocks + d.interceptions,
                CASE WHEN s.minutes = 0 THEN 0 END)                            AS cbi_filled,
       COALESCE(s.tackles,    d.tackles,    CASE WHEN s.minutes = 0 THEN 0 END) AS tackles_filled,
       COALESCE(s.recoveries, d.recoveries, CASE WHEN s.minutes = 0 THEN 0 END) AS recoveries_filled,
       -- raw count, best available source
       COALESCE(s.defensive_contribution,
                CASE WHEN s.minutes = 0 OR ps.position = 'GKP' THEN 0
                     WHEN ps.position = 'DEF' THEN d.cbit
                     ELSE d.cbirt END)                                          AS defensive_contribution_filled,
       -- count on today's stat definitions (equals the raw count wherever FPL publishes it)
       COALESCE(s.defensive_contribution::numeric,
                CASE WHEN s.minutes = 0 OR ps.position = 'GKP' THEN 0
                     ELSE d.clearances * COALESCE(a.f_cl, 1) + d.blocks * COALESCE(a.f_bl, 1)
                        + d.interceptions * COALESCE(a.f_it, 1) + d.tackles * COALESCE(a.f_tk, 1)
                        + CASE WHEN ps.position = 'DEF' THEN 0 ELSE d.recoveries * COALESCE(a.f_rc, 1) END
                END)                                                            AS defensive_contribution_adjusted,
       CASE WHEN s.defensive_contribution IS NOT NULL THEN s.source
            WHEN s.minutes = 0 OR ps.position = 'GKP' THEN 'not_applicable'
            WHEN d.player_id IS NOT NULL THEN d.source END                      AS defensive_source
FROM core.player_match_stats s
LEFT JOIN core.player_seasons ps ON ps.player_id = s.player_id AND ps.season = s.season
LEFT JOIN core.player_match_defensive d
       ON d.player_id = s.player_id AND d.fixture_id = s.fixture_id AND d.source = 'pitchapi'
LEFT JOIN adj a ON a.season = s.season;
