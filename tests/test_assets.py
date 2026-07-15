"""
Asset presence: feature_engineering loads park-wall geometry and altitude CSVs
at import time inside a scoped try/except that falls back to EMPTY dicts on a
missing file. That fallback silently degrades the batted-ball model to
league-median walls / zero altitude with no error.

We deliberately keep the runtime swallow (a transient read failure shouldn't
hard-fail the daily bake), and instead assert here — at CI time, before any SHA
is pinned into the daily pipeline — that both assets actually loaded. A moved or
renamed CSV becomes a red test, not a quietly worse model in production.
"""
from Model import feature_engineering


def test_park_walls_loaded_non_empty():
    assert feature_engineering._WALLS, (
        "feature_engineering._WALLS is empty — mlb_park_walls.csv failed to load; "
        "the model would silently fall back to league-median walls."
    )


def test_park_altitudes_loaded_non_empty():
    assert feature_engineering._ALTITUDES, (
        "feature_engineering._ALTITUDES is empty — park_altitudes.csv failed to load; "
        "the model would silently treat every park as sea level."
    )
