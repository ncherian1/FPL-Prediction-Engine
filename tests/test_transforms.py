from datetime import datetime, timezone

from fpl_engine.sources import fpl_api, football_data, vaastav
from fpl_engine.transform import match_stats, season_label, to_bool, to_int

OBS = datetime(2026, 11, 23, tzinfo=timezone.utc)


def test_season_label():
    assert season_label(2026) == "2026-27"
    assert season_label(2099) == "2099-00"


def test_missing_is_none_not_zero():
    stats = match_stats({"minutes": "90", "goals_scored": "0", "expected_goals": ""})
    assert stats["minutes"] == 90
    assert stats["goals"] == 0            # observed zero stays zero
    assert stats["expected_goals"] is None  # blank -> unknown
    assert stats["key_passes"] is None    # absent column -> unknown
    assert stats["price"] is None


def test_scalar_parsers():
    assert to_int("3.0") == 3
    assert to_int(float("nan")) is None
    assert to_bool("True") is True and to_bool("False") is False and to_bool("") is None


def test_expected_return_parsing():
    assert str(fpl_api.parse_expected_return("Hamstring - Expected back 30 Nov", OBS)) == "2026-11-30"
    # A January date seen in November is next year
    assert str(fpl_api.parse_expected_return("Knee - Expected back 04 Jan", OBS)) == "2027-01-04"
    assert fpl_api.parse_expected_return("Knock - 75% chance of playing", OBS) is None


def _bootstrap(status="d", chance=None):
    return {
        "events": [{"id": 1, "deadline_time": "2026-08-21T17:30:00Z"}],
        "element_types": [{"id": 3, "singular_name_short": "MID"}],
        "teams": [{"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS"}],
        "elements": [{"id": 10, "code": 9999, "team": 1, "element_type": 3, "now_cost": 105,
                      "cost_change_start": 5, "selected_by_percent": "12.3", "status": status,
                      "chance_of_playing_next_round": chance, "news": "Knock", "news_added": None}],
    }


def test_bootstrap_rows():
    r = fpl_api.bootstrap_rows(_bootstrap(chance=75), OBS, 1)
    assert r["season"] == "2026-27"
    p = r["players"][0]
    assert p["player_id"] == 9999 and p["team_id"] == 3 and p["current_price"] == 10.5
    assert r["player_seasons"][0]["start_price"] == 10.0
    a = r["availability"][0]
    assert a["injury_status"] == "doubtful" and a["availability_probability"] == 0.75


def test_no_chance_statement_is_not_full_fitness():
    a = fpl_api.bootstrap_rows(_bootstrap(status="a", chance=None), OBS, 1)["availability"][0]
    assert a["availability_probability"] is None


def test_history_rows_resolve_team_from_fixture():
    fixtures = {7: (100, 3, 8, 2, None)}  # fpl fixture 7: home=3, away=8, gw2
    rows = fpl_api.history_rows([{"fixture": 7, "was_home": False, "round": 2, "total_points": 5}],
                                player_id=9999, season="2026-27", fixtures=fixtures, observed=OBS)
    assert rows[0]["fixture_id"] == 100
    assert rows[0]["team_id"] == 8 and rows[0]["opponent_team_id"] == 3
    assert rows[0]["fpl_points"] == 5


def test_derive_fixtures_from_player_rows():
    rows = [
        {"fixture": "1", "round": "1", "was_home": "True", "opponent_team": "5", "kickoff_time": "k",
         "team_h_score": "2", "team_a_score": "1"},
        {"fixture": "1", "round": "1", "was_home": "False", "opponent_team": "9", "kickoff_time": "k",
         "team_h_score": "2", "team_a_score": "1"},
    ]
    (f,) = vaastav.derive_fixtures(rows)
    assert f["team_h"] == "9" and f["team_a"] == "5"


def test_football_data_parse():
    csv = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HxG,AxG\nE0,21/08/2026,Arsenal,Coventry,3,0,H,1.88,\n,,,,,,,,\n"
    (row,) = football_data.parse_rows(csv, "2026-27", "E0", OBS, 1)
    assert str(row["match_date"]) == "2026-08-21"
    assert row["fthg"] == 3 and row["home_xg"] == 1.88 and row["away_xg"] is None
    assert football_data.season_code(2026) == "2627"
