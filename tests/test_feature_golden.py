"""
Golden-output guard for ``Model.feature_engineering.create_features_for_prediction``.

Training and inference share this one function, so a reordered dict literal or a
silently changed constant is train/serve skew. The existing F6 test only asserts
``len(columns) == 29``; these tests pin the emitted *order* and a handful of
values so any drift is a loud failure rather than a quiet one.

Values below were captured from the shipped implementation and are a
characterization baseline, not an independently derived expectation. If a
deliberate model change moves them, retrain + re-bake-off first, then update
this file in the same commit.
"""
import pytest

# The fixed input and the `build_golden` factory live in tests/conftest.py, so
# tests/test_model_predict.py can share them without a cross-module import.

# Emitted order, copied verbatim from a run of the shipped implementation.
# Column order is load-bearing: the pickled ColumnTransformer selects by name,
# but the exported per-ball frames and the training path both rely on this order.
EXPECTED_COLUMNS = [
    "hitData_launchSpeed",
    "hitData_launchAngle",
    "distance_proxy",
    "hr_distance_proxy",
    "launch_speed_squared",
    "spray_angle_adj",
    "spray_angle_abs",
    "is_barrel",
    "is_pulled",
    "is_opposite",
    "pulled_hard",
    "oppo_hard",
    "spray_ev_interaction",
    "pulled_ground_ball",
    "oppo_line_drive",
    "altitude_ft",
    "wall_distance_ft",
    "carry_ft",
    "over_fence_margin",
    "carry_ft_spin",
    "over_fence_margin_spin",
    "total_spin_rpm",
    "sidespin_abs_rpm",
    "carry_ft_spin_temp",
    "over_fence_margin_spin_temp",
    "temp_f",
    "launch_angle_category",
    "spray_direction",
    "venue_id",
]

# Captured from the shipped implementation at the golden input
# (103.4 mph, 27 deg, coords 140.2/90.7, R, Coors Field, 85F).
GOLDEN_VALUES = {
    "spray_angle_adj": 7.769886810339523,
    "carry_ft": 390.04960086297257,
    "wall_distance_ft": 421.344651441818,
    "total_spin_rpm": 3076.506566461019,
    "carry_ft_spin_temp": 437.8739321828653,
    "over_fence_margin_spin_temp": 16.52928074104733,
}


def test_column_order_is_frozen(golden_frame):
    assert list(golden_frame.columns) == EXPECTED_COLUMNS


def test_single_row_output(golden_frame):
    assert len(golden_frame) == 1


@pytest.mark.parametrize("column, expected", sorted(GOLDEN_VALUES.items()))
def test_golden_feature_value(golden_frame, column, expected):
    assert golden_frame[column].iloc[0] == pytest.approx(expected, rel=1e-9)


def test_spin_carry_is_temperature_neutral_at_70f(build_golden):
    """At the 70F reference temperature the temp-adjusted carry is the identity.

    ``carry_ft_spin_temp`` applies a multiplicative temperature correction on top
    of ``carry_ft_spin`` anchored at 70F. Anything that shifts the anchor breaks
    the F6 feature's meaning without changing its name, so pin the identity here.
    """
    df = build_golden(temp_f=70)
    assert df["carry_ft_spin"].iloc[0] == df["carry_ft_spin_temp"].iloc[0]
    assert (
        df["over_fence_margin_spin"].iloc[0]
        == df["over_fence_margin_spin_temp"].iloc[0]
    )
