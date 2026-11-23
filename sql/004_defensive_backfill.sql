-- Per-player defensive actions from PitchAPI (validated against FPL's own counts in 2025-26:
-- 95.5% identical, 99.7% same 2-point outcome — notebooks/01b_player_matching.ipynb).
CREATE TABLE IF NOT EXISTS core.player_match_defensive (
    player_id      integer NOT NULL REFERENCES core.players(player_id),
    fixture_id     bigint  NOT NULL REFERENCES core.fixtures(fixture_id),
    season         text    NOT NULL,
    source         text    NOT NULL,             -- pitchapi
    minutes        integer,
    clearances     integer,
    blocks         integer,
    interceptions  integer,
    tackles        integer,
    recoveries     integer,
    cbit           integer,                      -- clearances + blocks + interceptions + tackles (DEF count)
    cbirt          integer,                      -- cbit + recoveries (MID/FWD count)
    observed_at    timestamptz NOT NULL,
    PRIMARY KEY (player_id, fixture_id, source)
);

-- player_match_stats with defensive fields completed from the best available source:
-- FPL's own values when published (2025-26 →), otherwise PitchAPI-derived (2021-22 → 2024-25).
-- A player who did not play has an observed 0. Anything else unknown stays NULL.
CREATE OR REPLACE VIEW core.v_player_match_stats AS
SELECT s.*,
       ps.position,
       COALESCE(s.clearances_blocks_interceptions, d.clearances + d.blocks + d.interceptions,
                CASE WHEN s.minutes = 0 THEN 0 END)                            AS cbi_filled,
       COALESCE(s.tackles,    d.tackles,    CASE WHEN s.minutes = 0 THEN 0 END) AS tackles_filled,
       COALESCE(s.recoveries, d.recoveries, CASE WHEN s.minutes = 0 THEN 0 END) AS recoveries_filled,
       COALESCE(s.defensive_contribution,
                CASE WHEN s.minutes = 0 OR ps.position = 'GKP' THEN 0
                     WHEN ps.position = 'DEF' THEN d.cbit
                     ELSE d.cbirt END)                                          AS defensive_contribution_filled,
       CASE WHEN s.defensive_contribution IS NOT NULL THEN s.source
            WHEN s.minutes = 0 OR ps.position = 'GKP' THEN 'not_applicable'
            WHEN d.player_id IS NOT NULL THEN d.source END                      AS defensive_source
FROM core.player_match_stats s
LEFT JOIN core.player_seasons ps ON ps.player_id = s.player_id AND ps.season = s.season
LEFT JOIN core.player_match_defensive d
       ON d.player_id = s.player_id AND d.fixture_id = s.fixture_id AND d.source = 'pitchapi';
