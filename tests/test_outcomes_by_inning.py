"""
Ordering contract for ``outcomes_by_inning``.

The per-inning win-probability simulator replays events in list order, so the
sort is the behavior: within an inning, a steal or pickoff must land BEFORE the
plate appearance sharing its ``ab_num`` (a baserunning event is attempted
mid-at-bat, before that at-bat resolves). The missing-``ab_num`` fallback shares
one counter across both loops so synthetic values can never collide -- and warns,
because production data should always carry ``ab_num``.
"""
import warnings

import pandas as pd
import pytest

from Simulator.game_simulator import outcomes_by_inning


def _pa(inning, ab_num, event_type, launch_speed=None, top=False):
    return {
        "inning": inning,
        "ab_num": ab_num,
        "isTopInning": top,
        "eventType": event_type,
        "batter.fullName": f"Batter {ab_num}",
        "hitData.launchSpeed": launch_speed,
        "hitData.launchAngle": 20.0 if launch_speed is not None else None,
        "hitData.totalDistance": None,
        "venue.name": "Fenway Park",
        "hitData.coordinates.coordX": 120.0,
        "hitData.coordinates.coordY": 95.0,
        "batSide.code": "R",
        "pitchHand.code": "R",
        "pitcher.id": 100,
        "playId": f"play-{ab_num}",
        "weather_temp_f": 72.0,
        "weather_condition": "Clear",
    }


def _baserunning(inning, ab_num, play, top=False):
    return {
        "inning": inning,
        "ab_num": ab_num,
        "isTopInning": top,
        "play": play,
        "batter.fullName": f"Runner {ab_num}",
    }


@pytest.fixture
def game_frames():
    plate_appearances = pd.DataFrame(
        [
            _pa(1, 1, "out"),                        # strikeout-like
            _pa(1, 2, "walk"),
            _pa(1, 3, "single", launch_speed=95.0),   # batted ball
            _pa(2, 4, "out"),
        ]
    )
    baserunning = pd.DataFrame(
        [
            _baserunning(1, 3, "stolen_base"),        # shares ab_num 3 with the PA
            _baserunning(2, 4, "pickoff"),            # shares ab_num 4 with the PA
        ]
    )
    return plate_appearances, baserunning


def test_baserunning_sorts_before_the_plate_appearance_sharing_its_ab_num(game_frames):
    plate_appearances, baserunning = game_frames
    result = outcomes_by_inning(plate_appearances, baserunning, "home")

    kinds = [
        outcome if isinstance(outcome, str) else "batted_ball"
        for outcome, _ in result
    ]
    assert kinds == [
        "strikeout",       # ab_num 1
        "walk",            # ab_num 2
        "stolen_base",     # ab_num 3, baserunning -> before the PA
        "batted_ball",     # ab_num 3, the plate appearance
        "pickoff",         # ab_num 4, baserunning -> before the PA
        "strikeout",       # ab_num 4, the plate appearance
    ]


def test_innings_are_tagged_and_ordered(game_frames):
    plate_appearances, baserunning = game_frames
    result = outcomes_by_inning(plate_appearances, baserunning, "home")
    innings = [inning for _, inning in result]
    assert innings == [1, 1, 1, 1, 2, 2]
    assert innings == sorted(innings)


def test_home_and_away_split_on_is_top_inning(game_frames):
    plate_appearances, baserunning = game_frames
    # Every synthetic row is a bottom-inning (home) event.
    assert outcomes_by_inning(plate_appearances, baserunning, "away") == []
    assert len(outcomes_by_inning(plate_appearances, baserunning, "home")) == 6


def test_unmodeled_event_types_are_dropped(game_frames):
    """HBP / interference have no launch data and are not walks or outs."""
    _, baserunning = game_frames
    with_hbp = pd.DataFrame(
        [
            _pa(1, 1, "out"),
            _pa(1, 2, "walk"),
            _pa(1, 3, "single", launch_speed=95.0),
            _pa(2, 4, "out"),
            _pa(2, 5, "hit_by_pitch"),
        ]
    )
    assert len(outcomes_by_inning(with_hbp, baserunning, "home")) == 6


def test_rows_without_an_inning_are_dropped(game_frames):
    _, baserunning = game_frames
    no_inning = pd.DataFrame(
        [
            _pa(1, 1, "out"),
            _pa(1, 2, "walk"),
            _pa(1, 3, "single", launch_speed=95.0),
            _pa(2, 4, "out"),
            _pa(None, 6, "walk"),
        ]
    )
    assert len(outcomes_by_inning(no_inning, baserunning, "home")) == 6


def test_missing_ab_num_warns_on_the_plate_appearance_path(game_frames):
    plate_appearances, baserunning = game_frames
    broken = plate_appearances.copy()
    broken.loc[1, "ab_num"] = None

    with pytest.warns(RuntimeWarning, match="plate-appearance row missing ab_num"):
        result = outcomes_by_inning(broken, baserunning, "home")
    assert len(result) == 6


def test_missing_ab_num_warns_on_the_baserunning_path(game_frames):
    plate_appearances, baserunning = game_frames
    broken = baserunning.copy()
    broken.loc[0, "ab_num"] = None

    with pytest.warns(RuntimeWarning, match="baserunning row missing ab_num"):
        result = outcomes_by_inning(plate_appearances, broken, "home")
    assert len(result) == 6


def test_synthetic_ab_nums_never_collide_between_the_two_loops(game_frames):
    """One shared counter, not two independent ones -- a collision would make the
    sort order depend on pandas row order rather than on the game."""
    plate_appearances, baserunning = game_frames
    broken_pa = plate_appearances.copy()
    broken_pa["ab_num"] = None
    broken_br = baserunning.copy()
    broken_br["ab_num"] = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = outcomes_by_inning(broken_pa, broken_br, "home")

    # Every event survives, and the result is deterministic across repeat calls.
    assert len(result) == 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        again = outcomes_by_inning(broken_pa, broken_br, "home")
    assert [
        o if isinstance(o, str) else "batted_ball" for o, _ in result
    ] == [o if isinstance(o, str) else "batted_ball" for o, _ in again]


def test_clean_data_emits_no_warning(game_frames):
    plate_appearances, baserunning = game_frames
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        outcomes_by_inning(plate_appearances, baserunning, "home")
