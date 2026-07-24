"""Temp guard + F6 column emission tests."""
import math

import pytest

from Model.feature_engineering import (
    create_features_for_prediction,
    create_features_for_prediction_fallback,
    is_roof_closed,
    sanitize_temp,
)

F6_NEW_COLS = [
    "carry_ft_spin", "over_fence_margin_spin", "total_spin_rpm",
    "sidespin_abs_rpm", "carry_ft_spin_temp", "over_fence_margin_spin_temp",
    "temp_f",
]


@pytest.mark.parametrize("temp,roof,expected", [
    (None, False, 70.0),          # missing -> default
    (float("nan"), True, 70.0),   # NaN -> default (before roof logic)
    (0.0, True, 72.0),            # dome reporting 0F -> indoor default
    (108.0, True, 72.0),          # roof closed, implausible outdoor temp
    (72.0, True, 72.0),           # plausible indoor reading kept
    (50.0, True, 50.0),           # roof boundary kept
    (95.0, True, 95.0),           # roof boundary kept
    (108.0, False, 105.0),        # open air clip high
    (30.0, False, 35.0),          # open air clip low
    (40.0, False, 40.0),          # open air plausible kept
])
def test_sanitize_temp(temp, roof, expected):
    assert sanitize_temp(temp, roof) == pytest.approx(expected)


def test_is_roof_closed():
    assert is_roof_closed("Dome") and is_roof_closed("Roof Closed")
    assert not is_roof_closed("Clear")
    assert not is_roof_closed(None)


def test_emits_29_columns_with_defaults():
    df = create_features_for_prediction(103.0, 28.0, 125.0, 100.0, "R", "15")
    assert len(df.columns) == 29
    for col in F6_NEW_COLS:
        assert col in df.columns
    assert df["temp_f"].iloc[0] == pytest.approx(70.0)
    # at the default 70F the temp carry equals the non-temp carry
    assert df["carry_ft_spin_temp"].iloc[0] == pytest.approx(df["carry_ft_spin"].iloc[0])
    assert df["over_fence_margin_spin"].iloc[0] == pytest.approx(
        df["carry_ft_spin"].iloc[0] - df["wall_distance_ft"].iloc[0])


def test_existing_f3_columns_unchanged():
    base = create_features_for_prediction(103.0, 28.0, 125.0, 100.0, "R", "15")
    hot = create_features_for_prediction(103.0, 28.0, 125.0, 100.0, "R", "15",
                                         temp_f=95.0)
    for col in ["hitData_launchSpeed", "carry_ft", "over_fence_margin",
                "wall_distance_ft", "altitude_ft", "venue_id"]:
        assert base[col].iloc[0] == hot[col].iloc[0]


def test_hot_day_increases_temp_carry_only():
    base = create_features_for_prediction(103.0, 28.0, 125.0, 100.0, "R", "15")
    hot = create_features_for_prediction(103.0, 28.0, 125.0, 100.0, "R", "15",
                                         temp_f=95.0)
    assert hot["temp_f"].iloc[0] == pytest.approx(95.0)
    assert hot["carry_ft_spin_temp"].iloc[0] > base["carry_ft_spin_temp"].iloc[0]
    assert hot["carry_ft_spin"].iloc[0] == pytest.approx(base["carry_ft_spin"].iloc[0])


def test_roof_closed_guard_applies_inside_features():
    df = create_features_for_prediction(103.0, 28.0, 125.0, 100.0, "R", "15",
                                        temp_f=108.0, roof_closed=True)
    assert df["temp_f"].iloc[0] == pytest.approx(72.0)


def test_fallback_emits_same_columns():
    full = create_features_for_prediction(103.0, 28.0, 125.0, 100.0, "R", "15")
    fb = create_features_for_prediction_fallback(103.0, 28.0, "15", temp_f=90.0)
    assert list(fb.columns) == list(full.columns)
    assert fb["temp_f"].iloc[0] == pytest.approx(90.0)


def test_spin_columns_consistent():
    df = create_features_for_prediction(103.0, 28.0, 110.0, 90.0, "L", "15")
    total = df["total_spin_rpm"].iloc[0]
    side = df["sidespin_abs_rpm"].iloc[0]
    assert total >= side >= 0.0
    assert math.isfinite(total)
