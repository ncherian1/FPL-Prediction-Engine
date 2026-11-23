-- Season-configurable FPL scoring rules (spec §5). One row per season x event x position.
-- threshold: appearance_60 = minute cut-off; saves / goals_conceded = "per N";
--            defensive_contribution = count needed to score.
CREATE TABLE IF NOT EXISTS core.scoring_rules (
    season           text NOT NULL REFERENCES core.seasons(season),
    event_type       text NOT NULL,
    player_position  text NOT NULL,
    points_value     numeric NOT NULL,
    threshold        numeric,
    effective_from   timestamptz,
    effective_to     timestamptz,
    source           text NOT NULL DEFAULT 'derived',   -- game_config | derived (validated against official points)
    updated_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (season, event_type, player_position)
);
