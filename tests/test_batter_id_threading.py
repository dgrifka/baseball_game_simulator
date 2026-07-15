"""
Regression test: create_detailed_outcomes_df() must carry the batter's
numeric MLB id through to its output, mirroring the existing pitcher_id
threading.

Background: outcomes() captured 'pitcher_id': row.get('pitcher.id') into its
batted-ball dict but only captured the batter as a NAME
(row['batter.fullName']) — no id. calculate_total_bases() and
create_detailed_outcomes_df() propagated that gap, so the batter's numeric id
never reached downstream consumers (e.g. the private repo's batted-ball
parquet export), even though the raw game_data frame carries 'batter.id'
(see Simulator/get_game_information.py's selected columns) right next to
'pitcher.id'.
"""
import pandas as pd

from Simulator.game_simulator import create_detailed_outcomes_df


def _put_in_play_row(batter_id, batter_name, pitcher_id=555, is_top_inning=False):
    return {
        "isTopInning": is_top_inning,
        "eventType": "single",
        "hitData.launchSpeed": 100.0,
        "hitData.launchAngle": 25.0,
        "hitData.totalDistance": 350.0,
        "venue.name": "Yankee Stadium",
        "hitData.coordinates.coordX": 100.0,
        "hitData.coordinates.coordY": 150.0,
        "batSide.code": "R",
        "pitchHand.code": "R",
        "pitcher.id": pitcher_id,
        "batter.id": batter_id,
        "playId": "abc123",
        "inning": 3,
        "batter.fullName": batter_name,
        "pitcher.fullName": "Test Pitcher",
    }


def _empty_steals_and_pickoffs():
    return pd.DataFrame(columns=["isTopInning", "batter.fullName", "pitcher.fullName", "play"])


class TestBatterIdThreading:
    def test_create_detailed_outcomes_df_includes_batter_id(self):
        game_data = pd.DataFrame([
            _put_in_play_row(660271, "Test Hitter Home"),
        ])

        result = create_detailed_outcomes_df(game_data, _empty_steals_and_pickoffs(), "home")

        assert "batter_id" in result.columns
        assert list(result["batter_id"]) == [660271]

    def test_batter_id_distinguishes_two_rows(self):
        game_data = pd.DataFrame([
            _put_in_play_row(660271, "Test Hitter A"),
            _put_in_play_row(592450, "Test Hitter B"),
        ])

        result = create_detailed_outcomes_df(game_data, _empty_steals_and_pickoffs(), "home")

        assert set(result["batter_id"]) == {660271, 592450}
