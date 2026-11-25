"""Documentation/make_article_figures.py: the pure helpers, no network.

Figure 1 needs the MLB Stats API and the model; Figure 2 needs the model
metadata. These tests pin the helpers that shape their inputs, so a renamed
metadata key or a second matching batted ball breaks here rather than in a
silently wrong README chart.
"""
import importlib.util
import os

import pandas as pd
import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), os.pardir,
                       "Documentation", "make_article_figures.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("make_article_figures", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata():
    # Deliberately out of order: the helper must sort by exit velocity, not
    # trust the JSON's key order.
    return {"tail_metrics": {"hr_rmse_by_ev_bin": {
        "95-100": {"count": 300, "actual_hr_rate": 0.10, "model_hr_rate": 0.08, "gap": 0.02},
        "85-90": {"count": 100, "actual_hr_rate": 0.001, "model_hr_rate": 0.002, "gap": -0.001},
        "115-120": {"count": 20, "actual_hr_rate": 0.45, "model_hr_rate": 0.52, "gap": -0.07},
        "90-95": {"count": 200, "actual_hr_rate": 0.01, "model_hr_rate": 0.01, "gap": 0.0},
    }}}


def test_hr_check_rows_sorts_by_speed_and_drops_the_top_bins(mod):
    rows = mod.hr_check_rows(_metadata(), max_bins=3)
    assert rows == [
        ("85-90", 0.002, 0.001, 100),
        ("90-95", 0.01, 0.01, 200),
        ("95-100", 0.08, 0.10, 300),
    ]


def test_hr_check_subtitle_counts_the_drawn_balls_and_names_the_dropped_bin(mod):
    subtitle = mod.hr_check_subtitle(_metadata(), max_bins=3)
    assert "600 batted balls from the model's own home run check" in subtitle
    assert "115-120 mph (20 balls) not shown" in subtitle


def test_hr_check_subtitle_has_no_dropped_clause_when_every_bin_is_drawn(mod):
    subtitle = mod.hr_check_subtitle(_metadata(), max_bins=10)
    assert "620 batted balls" in subtitle
    assert "not shown" not in subtitle


def _frame():
    return pd.DataFrame({
        "player": ["Tommy Edman", "Tommy Edman", "Max Muncy", "Tommy Edman"],
        "launch_speed": [101.3, 78.9, 101.3, None],
        "launch_angle": [42.0, 10.0, 42.0, None],
        "estimated_bases": [2.01, 1.36, 3.0, 0.0],
        "out_prob": [0.49, 0.6, 0.2, 1.0],
        "single_prob": [0.004, 0.3, 0.1, 0.0],
        "double_prob": [0.004, 0.1, 0.1, 0.0],
        "triple_prob": [0.002, 0.0, 0.0, 0.0],
        "hr_prob": [0.50, 0.0, 0.6, 0.0],
    })


def test_pick_ball_finds_the_one_row_by_name_speed_and_angle(mod):
    assert mod.pick_ball(_frame(), "Tommy Edman", 101.3, 42) == 0


def test_pick_ball_tolerates_display_rounding(mod):
    frame = _frame()
    frame.loc[0, "launch_speed"] = 101.27
    frame.loc[0, "launch_angle"] = 41.6
    assert mod.pick_ball(frame, "Tommy Edman", 101.3, 42) == 0


@pytest.mark.parametrize("player,speed,angle", [
    ("Tommy Edman", 99.9, 42),     # no such ball
    ("Shohei Ohtani", 101.3, 42),  # no such batter
])
def test_pick_ball_refuses_when_nothing_matches(mod, player, speed, angle):
    with pytest.raises(ValueError, match="expected exactly one"):
        mod.pick_ball(_frame(), player, speed, angle)


def test_pick_ball_refuses_two_matches(mod):
    frame = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="expected exactly one"):
        mod.pick_ball(frame, "Tommy Edman", 101.3, 42)


def test_outcome_bars_reads_the_five_probabilities_in_order(mod):
    bars = mod.outcome_bars(_frame().iloc[0])
    assert [label for label, _, _ in bars] == ["Out", "Single", "Double", "Triple", "Home run"]
    assert [p for _, p, _ in bars] == [0.49, 0.004, 0.004, 0.002, 0.50]
    assert [key for _, _, key in bars] == ["out", "single", "xbh", "xbh", "hr"]


@pytest.mark.parametrize("event,label", [
    ("out", "Out"),
    ("single", "Single"),
    ("double", "Double"),
    ("triple", "Triple"),
    ("home_run", "Home run"),
    ("walk", None),
])
def test_actual_label_maps_the_feed_event_to_its_bar(mod, event, label):
    assert mod.actual_label(event) == label


@pytest.mark.parametrize("p,label", [
    (0.4951, "50%"),
    (0.01, "1%"),
    (0.004, "<1%"),
    (0.0, "0%"),
])
def test_pct_label(mod, p, label):
    assert mod.pct_label(p) == label
