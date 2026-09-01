"""
Boundary tests for ``_apply_hr_tail_correction``.

This correction is simulation-only by design: it bumps HR probability at 100+ mph
inside ``simulator()`` during resampling, while the *exported* per-ball
probabilities stay raw calibrated output. That asymmetry is deliberate -- these
tests pin the correction's edges without touching it.
"""
import numpy as np
import pytest

from Simulator.game_simulator import HR_TAIL_CORRECTIONS, _apply_hr_tail_correction

# A plausible hard-hit probability vector: [out, single, double, triple, hr].
BASE_PROBS = [0.50, 0.20, 0.10, 0.02, 0.18]


def probs():
    return np.array(BASE_PROBS, dtype=float)


@pytest.mark.parametrize("launch_speed", [0.0, 60.0, 95.0, 99.9, 99.999])
def test_below_100_mph_is_untouched(launch_speed):
    """Under 100 mph the original array comes back, not a copy."""
    p = probs()
    result = _apply_hr_tail_correction(p, launch_speed)
    assert result is p
    assert np.array_equal(result, BASE_PROBS)


@pytest.mark.parametrize("launch_speed", [100.0, 102.5, 104.999, 105.0, 130.0, 199.999])
def test_corrected_rows_still_sum_to_one(launch_speed):
    result = _apply_hr_tail_correction(probs(), launch_speed)
    assert result.sum() == pytest.approx(1.0, abs=1e-12)
    assert (result >= 0).all()


@pytest.mark.parametrize(
    "launch_speed, factor",
    [(100.0, 1.02), (104.999, 1.02), (105.0, 1.05), (150.0, 1.05)],
)
def test_hr_probability_is_boosted_by_the_bucket_factor(launch_speed, factor):
    result = _apply_hr_tail_correction(probs(), launch_speed)
    assert result[4] == pytest.approx(BASE_PROBS[4] * factor, rel=1e-12)
    # The boost is taken out of out_prob, nothing else moves.
    assert result[0] == pytest.approx(
        BASE_PROBS[0] - BASE_PROBS[4] * (factor - 1.0), rel=1e-12
    )
    assert list(result[1:4]) == BASE_PROBS[1:4]


def test_input_array_is_never_mutated():
    """The caller's array is reused for the raw export; mutating it would leak."""
    for launch_speed in (99.0, 100.0, 105.0, 250.0):
        p = probs()
        _apply_hr_tail_correction(p, launch_speed)
        assert np.array_equal(p, BASE_PROBS), f"mutated at {launch_speed} mph"


def test_hr_probability_is_non_decreasing_across_the_bucket_boundaries():
    speeds = [99.9, 100.0, 104.999, 105.0, 150.0, 199.999]
    hr = [_apply_hr_tail_correction(probs(), s)[4] for s in speeds]
    assert hr == sorted(hr), f"HR probability dipped across buckets: {hr}"


def test_bucket_table_covers_100_mph_and_up_without_gaps():
    edges = sorted(HR_TAIL_CORRECTIONS)
    assert edges[0][0] == 100
    for (_, hi), (lo, _) in zip(edges, edges[1:]):
        assert hi == lo, f"gap between HR tail buckets at {hi}/{lo}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Latent boundary gap: HR_TAIL_CORRECTIONS' top bucket is (105, 200) and the "
        "lookup is `lo <= launch_speed < hi`, so at exactly 200 mph and above the "
        "correction silently stops applying and HR probability drops back to raw -- "
        "non-monotonic in EV. Not reachable in production (the Statcast record is "
        "~122 mph), and fixing it means editing Simulator/, which wave A3 forbids. "
        "Documented here so wave B can close the top bucket."
    ),
)
def test_hr_correction_still_applies_at_and_above_200_mph():
    hr_195 = _apply_hr_tail_correction(probs(), 199.999)[4]
    hr_200 = _apply_hr_tail_correction(probs(), 200.0)[4]
    assert hr_200 >= hr_195
