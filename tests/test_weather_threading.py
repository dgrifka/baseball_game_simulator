"""Weather extraction + threading tests (no network)."""
import pandas as pd
import pytest

from Simulator.get_game_information import parse_weather
from Simulator.game_simulator import outcomes, prepare_batted_ball_features


def test_parse_weather_present():
    game = {"gameData": {"weather": {"condition": "Sunny", "temp": "88", "wind": "10 mph"}}}
    temp_f, condition = parse_weather(game)
    assert temp_f == pytest.approx(88.0)
    assert condition == "Sunny"


def test_parse_weather_missing_block():
    temp_f, condition = parse_weather({"gameData": {}})
    assert pd.isna(temp_f)
    assert condition is None


def test_parse_weather_unparseable_temp():
    game = {"gameData": {"weather": {"condition": "Dome", "temp": "n/a"}}}
    temp_f, condition = parse_weather(game)
    assert pd.isna(temp_f)
    assert condition == "Dome"


def _game_data(weather_temp=91.0, condition="Sunny"):
    return pd.DataFrame([{
        "isTopInning": False, "eventType": "single", "batter.fullName": "A B",
        "hitData.launchSpeed": 101.0, "hitData.launchAngle": 22.0,
        "hitData.totalDistance": 300.0, "venue.name": "Wrigley Field",
        "hitData.coordinates.coordX": 120.0, "hitData.coordinates.coordY": 95.0,
        "batSide.code": "R", "pitchHand.code": "L", "pitcher.id": 1,
        "batter.id": 2, "playId": "p1", "inning": 3,
        "weather_temp_f": weather_temp, "weather_condition": condition,
    }])


def _empty_steals():
    # outcomes() boolean-indexes on isTopInning, so an empty frame still
    # needs the column
    return pd.DataFrame(columns=["isTopInning", "play", "batter.fullName"])


def test_outcomes_carries_weather():
    out = outcomes(_game_data(), _empty_steals(), "home")
    batted = [o for o in out if isinstance(o[0], dict)]
    assert batted
    assert batted[0][0]["temp_f"] == pytest.approx(91.0)
    assert batted[0][0]["roof_closed"] is False


def test_outcomes_roof_closed_flag():
    out = outcomes(_game_data(condition="Roof Closed"), _empty_steals(), "home")
    batted = [o for o in out if isinstance(o[0], dict)]
    assert batted[0][0]["roof_closed"] is True


def test_outcomes_without_weather_columns_defaults():
    gd = _game_data().drop(columns=["weather_temp_f", "weather_condition"])
    out = outcomes(gd, _empty_steals(), "home")
    batted = [o for o in out if isinstance(o[0], dict)]
    assert pd.isna(batted[0][0]["temp_f"])
    assert batted[0][0]["roof_closed"] is False


def test_prepare_features_accepts_temp():
    df = prepare_batted_ball_features(101.0, 22.0, "Wrigley Field",
                                      coord_x=120.0, coord_y=95.0, bat_side="R",
                                      temp_f=91.0)
    assert df["temp_f"].iloc[0] == pytest.approx(91.0)


def test_prepare_features_default_temp():
    df = prepare_batted_ball_features(101.0, 22.0, "Wrigley Field")
    assert df["temp_f"].iloc[0] == pytest.approx(70.0)


def test_prepare_features_nan_bat_side_coerced_not_raised():
    import numpy as np
    df = prepare_batted_ball_features(101.0, 22.0, "Wrigley Field",
                                      coord_x=120.0, coord_y=95.0,
                                      bat_side=np.nan)
    r_df = prepare_batted_ball_features(101.0, 22.0, "Wrigley Field",
                                        coord_x=120.0, coord_y=95.0,
                                        bat_side="R")
    assert df["carry_ft_spin"].iloc[0] == r_df["carry_ft_spin"].iloc[0]
