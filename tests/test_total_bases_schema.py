"""
Exported data contract for ``calculate_total_bases``.

This frame is the scoring standard for every estimated-bases data product (the
display table, the batted-balls export, player evaluations), and it is consumed
downstream by the private orchestration repo. A renamed, dropped or reordered
column is a silent breaking change, so the column list is pinned as a literal.

The probabilities here are RAW calibrated model output -- HR_TAIL_CORRECTIONS is
deliberately not applied on this path. That is asserted below so a well-meaning
"consistency" fix trips a test.
"""
import pandas as pd
import pytest

from Simulator.game_simulator import calculate_total_bases

# Emitted order, copied verbatim from a run of the shipped implementation.
# `play_id` and `pitcher` are appended only when present in the input.
EXPECTED_COLUMNS = [
    "player",
    "launch_speed",
    "launch_angle",
    "stadium",
    "event_type",
    "original_event_type",
    "estimated_bases",
    "out_prob",
    "single_prob",
    "double_prob",
    "triple_prob",
    "hr_prob",
    "coord_x",
    "coord_y",
    "bat_side",
    "pitcher_hand",
    "pitcher_id",
    "batter_id",
    "inning",
    "is_top_inning",
    "play_id",
    "pitcher",
]

PROB_COLUMNS = ["out_prob", "single_prob", "double_prob", "triple_prob", "hr_prob"]


@pytest.fixture(scope="module")
def scored(synthetic_ball):
    outcomes_list = [
        (synthetic_ball(103.4, 27, 140.2, 90.7, "R"), "home_run", "Slug Ger", "Ace Arm"),
        (synthetic_ball(88.0, 5, 110.0, 95.0, "L"), "single", "Sing Les", "Ace Arm"),
        ("walk", "walk", "Pat Ient", "Ace Arm"),
        ("strikeout", "strikeout", "Whi Ff", "Ace Arm"),
        ("stolen_base", "stolen_base", "Speed Y", "Ace Arm"),
        ("pickoff", "pickoff", "Care Less", "Ace Arm"),
    ]
    return calculate_total_bases(outcomes_list)


def test_column_list_is_frozen(scored):
    assert list(scored.columns) == EXPECTED_COLUMNS


def test_one_row_per_outcome(scored):
    assert len(scored) == 6


def test_probability_rows_sum_to_one(scored):
    for _, row in scored.iterrows():
        assert sum(row[c] for c in PROB_COLUMNS) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "event_type, bases, probs",
    [
        ("strikeout", 0, [1, 0, 0, 0, 0]),
        ("walk", 1, [0, 1, 0, 0, 0]),
        ("stolen_base", 1, [0, 1, 0, 0, 0]),
        ("pickoff", 0, [1, 0, 0, 0, 0]),
    ],
)
def test_non_batted_events_are_deterministic(scored, event_type, bases, probs):
    row = scored[scored["event_type"] == event_type].iloc[0]
    assert row["estimated_bases"] == bases
    assert [row[c] for c in PROB_COLUMNS] == probs
    # pandas coerces the None the function emits to NaN in the float column.
    assert pd.isna(row["launch_speed"])
    assert pd.isna(row["stadium"])


def test_estimated_bases_matches_the_probability_weighting(scored):
    """estimated_bases is 1*single + 2*double + 3*triple + 4*hr, by definition."""
    in_play = scored[scored["event_type"] == "in_play"]
    assert len(in_play) == 2
    for _, row in in_play.iterrows():
        expected = (
            row["single_prob"] * 1
            + row["double_prob"] * 2
            + row["triple_prob"] * 3
            + row["hr_prob"] * 4
        )
        assert row["estimated_bases"] == pytest.approx(expected, rel=1e-12)
        assert 0.0 <= row["estimated_bases"] <= 4.0


def test_exported_probabilities_are_raw_not_hr_tail_corrected(synthetic_ball):
    """HR_TAIL_CORRECTIONS is simulation-only; this export must stay raw."""
    import joblib
    import os

    from Simulator.game_simulator import prepare_batted_ball_features

    ball = synthetic_ball(103.4, 27, 140.2, 90.7, "R")
    row = calculate_total_bases([(ball, "home_run", "Slug Ger", "Ace Arm")]).iloc[0]

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline = joblib.load(os.path.join(repo_root, "Model", "batted_ball_model.pkl"))
    raw = pipeline.predict_proba(
        prepare_batted_ball_features(
            launch_speed=ball["launch_speed"],
            launch_angle=ball["launch_angle"],
            venue_name=ball["venue_name"],
            coord_x=ball["coord_x"],
            coord_y=ball["coord_y"],
            bat_side=ball["bat_side"],
            temp_f=ball["temp_f"],
            roof_closed=ball["roof_closed"],
        )
    )[0]
    assert row["hr_prob"] == pytest.approx(raw[4], rel=1e-12)
