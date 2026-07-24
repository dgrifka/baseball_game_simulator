"""Tests for Model/bbe_physics.py — Nathan spin + temp-aware carry (F6 port).

Pinned values generated from the private harness module
(baseball_simulator_model/Content/bbe_physics.py) on 2026-07-24; the private
repo's Gate 1 (Content/f6_deploy_gates.py) additionally proves exact parity
on the full frozen 2026 frame."""
import numpy as np
import pytest

from Model.bbe_physics import air_density_K, nathan_spin, spin_aware_carry


def test_nathan_spin_pinned_rhb_pull():
    bs, ss = nathan_spin(25.0, -10.0, "R")
    assert bs == pytest.approx(2027.0)
    assert ss == pytest.approx(91.0)


def test_nathan_spin_pinned_lhb_oppo():
    bs, ss = nathan_spin(25.0, 10.0, "L")
    assert bs == pytest.approx(2447.0)
    assert ss == pytest.approx(1789.0)


def test_carry_pinned_sea_level_70f():
    assert spin_aware_carry(103.0, 28.0, 0.0, "R", 0.0, 70.0) == pytest.approx(
        414.87773681144023, abs=1e-9)


def test_temp_increases_carry_pinned():
    c95 = spin_aware_carry(103.0, 28.0, 0.0, "R", 0.0, 95.0)
    assert c95 == pytest.approx(423.25348291457516, abs=1e-9)
    assert c95 > spin_aware_carry(103.0, 28.0, 0.0, "R", 0.0, 70.0)


def test_altitude_increases_carry_pinned():
    assert spin_aware_carry(103.0, 28.0, 0.0, "R", 5280.0, 70.0) == pytest.approx(
        445.4046810898481, abs=1e-9)


def test_default_temp_is_70():
    assert spin_aware_carry(101.0, 25.0, -12.0, "L", 600.0) == spin_aware_carry(
        101.0, 25.0, -12.0, "L", 600.0, 70.0)


def test_air_density_pinned():
    assert air_density_K(0.0, 70.0) == pytest.approx(0.005355777902676589, abs=1e-12)
    assert air_density_K(0.0, 95.0) == pytest.approx(0.0050842827292621766, abs=1e-12)


def test_vector_shapes_match_scalar():
    ev = np.array([95.0, 105.0])
    la = np.array([20.0, 30.0])
    spray = np.array([-5.0, 15.0])
    side = np.array(["R", "L"])
    out = spin_aware_carry(ev, la, spray, side, 0.0, 70.0)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(spin_aware_carry(95.0, 20.0, -5.0, "R", 0.0, 70.0))


def test_nan_spray_treated_as_center():
    bs_nan, ss_nan = nathan_spin(20.0, float("nan"), "R")
    bs_c, ss_c = nathan_spin(20.0, 0.0, "R")
    assert bs_nan == pytest.approx(bs_c)
    assert ss_nan == pytest.approx(ss_c)


def test_bad_bat_side_raises():
    with pytest.raises(ValueError):
        nathan_spin(20.0, 0.0, "S")
