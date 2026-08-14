"""Spray-angle vertex guards.

Two vertices coexist on purpose:

* ``Model.feature_engineering.HOME_PLATE_X/Y`` (125.42, 199.02) is the **frozen**
  model-feature vertex. Every shipped model (feature sets <= F6) was trained on
  angles measured from it, so it must not move without a retrain.
* ``Simulator.visualizations.VERTEX_CALIBRATED_X/Y`` (127.4, 215.0) is the
  **rendering** vertex, calibrated so drawn batted balls land on the correct side
  of the foul lines.

These tests pin both: a silent swap of the frozen vertex would invalidate the
model without any other test noticing.
"""
import importlib
import os

import numpy as np
import pytest

from Model.feature_engineering import (
    HOME_PLATE_X,
    HOME_PLATE_Y,
    create_features_for_prediction,
)
from Simulator.visualizations import (
    VERTEX_CALIBRATED_X,
    VERTEX_CALIBRATED_Y,
    calculate_spray_angle_calibrated,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARQUET_2024 = os.path.join(_REPO_ROOT, "Data", "batted_balls",
                             "batted_balls_2024.parquet")


def _parquet_readable():
    """Both the file and a pandas parquet engine have to be present.

    CI installs requirements.txt, which carries neither pyarrow nor fastparquet
    (nothing in the production path reads parquet), so the file being committed
    is not enough — without an engine pandas raises ImportError at read time.
    """
    if not os.path.exists(_PARQUET_2024):
        return False
    for engine in ("pyarrow", "fastparquet"):
        try:
            importlib.import_module(engine)
            return True
        except ImportError:
            continue
    return False

# A pulled ball for a right-handed batter: left of the frozen vertex, well into
# the outfield.
COORD_X, COORD_Y = 80.0, 90.0


def _arctan2_degrees(coord_x, coord_y, vertex_x, vertex_y):
    """Reference implementation, written out so a bug in the module can't hide."""
    return float(np.degrees(np.arctan2(coord_x - vertex_x, vertex_y - coord_y)))


def test_model_features_use_the_frozen_vertex():
    """The model path must keep measuring angles from (125.42, 199.02)."""
    assert (HOME_PLATE_X, HOME_PLATE_Y) == (125.42, 199.02)

    expected = _arctan2_degrees(COORD_X, COORD_Y, 125.42, 199.02)
    df = create_features_for_prediction(103.0, 28.0, COORD_X, COORD_Y, "R", "15")

    # bat_side 'R' leaves the handedness adjustment as a no-op, so the emitted
    # spray_angle_adj is the raw angle from the frozen vertex.
    assert df["spray_angle_adj"].iloc[0] == pytest.approx(expected)
    assert df["spray_angle_abs"].iloc[0] == pytest.approx(abs(expected))


def test_calibrated_vertex_constants_and_angle():
    assert (VERTEX_CALIBRATED_X, VERTEX_CALIBRATED_Y) == (127.4, 215.0)

    expected = _arctan2_degrees(COORD_X, COORD_Y, 127.4, 215.0)
    assert calculate_spray_angle_calibrated(COORD_X, COORD_Y) == pytest.approx(expected)


def test_calibrated_angle_differs_from_frozen_angle():
    """The two vertices are not interchangeable — that's the whole point."""
    frozen = _arctan2_degrees(COORD_X, COORD_Y, HOME_PLATE_X, HOME_PLATE_Y)
    calibrated = calculate_spray_angle_calibrated(COORD_X, COORD_Y)
    assert abs(frozen - calibrated) > 1.0


@pytest.mark.skipif(not _parquet_readable(),
                    reason="2024 batted-ball parquet or a parquet engine is unavailable")
def test_calibrated_vertex_keeps_home_runs_fair():
    """A home run cannot land foul, so |raw spray| > 45 deg is impossible by rule.

    On 2024 batted balls the frozen vertex puts ~4.9% of home runs outside the
    foul lines; the calibrated vertex drops that to ~0.2%.
    """
    import pandas as pd

    df = pd.read_parquet(_PARQUET_2024)
    hr = df[df.eventType.eq("home_run") & df.hitData_coordinates_coordX.notna()]
    x = hr.hitData_coordinates_coordX.to_numpy()
    y = hr.hitData_coordinates_coordY.to_numpy()

    frozen = np.degrees(np.arctan2(x - HOME_PLATE_X, HOME_PLATE_Y - y))
    calibrated = np.degrees(np.arctan2(x - VERTEX_CALIBRATED_X,
                                       VERTEX_CALIBRATED_Y - y))

    assert (np.abs(frozen) > 45).mean() > 0.04
    assert (np.abs(calibrated) > 45).mean() < 0.005
