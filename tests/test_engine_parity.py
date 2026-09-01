"""
Scalar-vs-vector engine parity on a fixed synthetic game.

`vector_engine.py` holds no baserunning rules of its own: its transition tables
are built at import time by calling the scalar functions in `game_simulator.py`
with the single stochastic branch forced each way. That auto-follow rests on one
assumption -- every event consumes at most ONE `random.random()` branch draw. Add
a second draw and the vectorized engine resolves the first, never sees the
second, and raises nothing. Run totals just drift.

This test is the drift alarm. It compares distributions, not individual games:
the two engines consume randomness in a different order, so identical per-game
results are not expected and would not be meaningful.

TOLERANCES ARE STATISTICAL until wave B seeds the generators. The scalar path
draws from the seedable `random` / `np.random` globals, but the vectorized path
builds an unseeded `np.random.default_rng()` inside `simulator()`, so its output
varies run to run and cannot be pinned from a test. Observed spread over 10 runs
at n=4000 was <0.07 runs and <2.2 percentage points; the bounds below sit ~4
sigma out. A *shifted mean* is the signature of a dropped branch draw -- if this
test fails, suspect that before suspecting the tolerance.
"""
import random

import numpy as np
import pytest

from Simulator.game_simulator import simulator

N_SIMS = 4_000  # 4k, not 10k: the whole suite has to stay under 30 s.
SEED = 20260901

# Statistical until wave B seeds the generators (see module docstring).
MEAN_RUNS_TOLERANCE = 0.25          # runs per team
WIN_PROBABILITY_TOLERANCE = 4.0     # percentage points; simulator() returns 0-100


def _run(outcomes, scalar, monkeypatch):
    home, away = outcomes
    if scalar:
        monkeypatch.setenv("DTW_SCALAR_SIM", "1")
    else:
        monkeypatch.delenv("DTW_SCALAR_SIM", raising=False)
    # Seeds the scalar path (global `random` + `np.random`); the vector path's
    # own default_rng() ignores them, which is exactly why this stays statistical.
    random.seed(SEED)
    np.random.seed(SEED)
    return simulator(N_SIMS, home, away)


@pytest.fixture(scope="module")
def parity_runs(request):
    """Both engines run once on the same game, so the comparison is apples to apples."""
    outcomes = request.getfixturevalue("synthetic_outcomes")
    mp = pytest.MonkeyPatch()
    try:
        vector = _run(outcomes, scalar=False, monkeypatch=mp)
        scalar = _run(outcomes, scalar=True, monkeypatch=mp)
    finally:
        mp.undo()
    return vector, scalar


def test_mean_runs_agree(parity_runs):
    vector, scalar = parity_runs
    for team, v, s in (("home", vector[0], scalar[0]), ("away", vector[1], scalar[1])):
        assert abs(v.mean() - s.mean()) < MEAN_RUNS_TOLERANCE, (
            f"{team} mean runs drifted: vector={v.mean():.4f} scalar={s.mean():.4f}"
        )


def test_win_probabilities_agree(parity_runs):
    vector, scalar = parity_runs
    for label, i in (("home_win", 2), ("away_win", 3), ("tie", 4)):
        assert abs(vector[i] - scalar[i]) < WIN_PROBABILITY_TOLERANCE, (
            f"{label} drifted: vector={vector[i]:.3f} scalar={scalar[i]:.3f}"
        )


def test_run_distribution_shape_agrees(parity_runs):
    """A dropped branch draw shifts the spread as well as the mean."""
    vector, scalar = parity_runs
    assert abs(vector[0].std() - scalar[0].std()) < 0.35
    assert abs(np.median(vector[0]) - np.median(scalar[0])) <= 1


def test_win_percentages_sum_to_one_hundred(parity_runs):
    for result in parity_runs:
        assert result[2] + result[3] + result[4] == pytest.approx(100.0, abs=1e-9)


def test_both_engines_return_the_requested_number_of_games(parity_runs):
    for result in parity_runs:
        assert result[0].shape == (N_SIMS,)
        assert result[1].shape == (N_SIMS,)
        assert (result[0] >= 0).all() and (result[1] >= 0).all()


def test_transition_tables_still_build():
    """Import-time table build is the forced-draw trick; a signature change breaks it."""
    from Simulator.game_simulator import _get_transition_tables

    tables = _get_transition_tables()
    assert tables, "transition tables came back empty"
    for name, table in tables.items():
        assert getattr(table, "shape", None) is not None, f"{name} is not an array"
