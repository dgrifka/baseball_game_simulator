"""
Ensures the repo root is importable as `Simulator.*` regardless of how
pytest's import mode resolves test-file paths (there's no top-level
conftest.py/pyproject.toml pinning rootdir insertion in this repo).
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402  (must follow the sys.path insertion above)

# Everything shared between test modules lives here as a fixture on purpose.
# A test module must never do `from tests.conftest import ...`: this repo has no
# pytest ini file, so pytest picks its rootdir by walking up to the enclosing
# .git, which resolves `tests.` against the MAIN checkout even when the suite is
# running inside a worktree. Fixtures are discovered by pytest itself and are
# immune to that.


def _synthetic_ball(launch_speed, launch_angle, coord_x, coord_y, bat_side):
    """One batted-ball outcome dict in the shape ``outcomes()`` emits."""
    return {
        "launch_speed": launch_speed,
        "launch_angle": launch_angle,
        "total_distance": None,
        "venue_name": "Fenway Park",
        "coord_x": coord_x,
        "coord_y": coord_y,
        "bat_side": bat_side,
        "pitcher_hand": "R",
        "pitcher_id": 100,
        "batter_id": 200,
        "play_id": "synthetic-play",
        "inning": 1,
        "is_top_inning": False,
        "temp_f": 72.0,
        "roof_closed": False,
    }


def _synthetic_team(ev_offset):
    """10 batted balls spanning the EV/LA grid, 2 walks, 3 Ks, a steal, a pickoff."""
    balls = [
        _synthetic_ball(
            launch_speed=86.0 + ev_offset + 3.0 * i,
            launch_angle=5.0 + 7.0 * (i % 5),
            coord_x=100.0 + 4.0 * i,
            coord_y=90.0 + 2.0 * i,
            bat_side="RL"[i % 2],
        )
        for i in range(10)
    ]
    return balls + [
        "walk",
        "walk",
        "strikeout",
        "strikeout",
        "strikeout",
        "stolen_base",
        "pickoff",
    ]


@pytest.fixture(scope="session")
def synthetic_outcomes():
    """A fixed synthetic game: (home_outcomes, away_outcomes).

    Deliberately hand-built rather than fetched, so the parity and schema tests
    never touch the MLB API and never vary between runs.
    """
    return _synthetic_team(0.0), _synthetic_team(1.0)


@pytest.fixture(scope="session")
def synthetic_ball():
    """Factory for one batted-ball outcome dict, in the shape ``outcomes()`` emits."""
    return _synthetic_ball


# --- Golden feature-frame input, shared by the feature and model-smoke tests ---

# One fixed batted ball: a hard, well-struck fly ball to the pull side at Coors
# Field (venue id 19, the altitude outlier) on a warm night.
GOLDEN_INPUT = dict(
    launch_speed=103.4,
    launch_angle=27,
    coord_x=140.2,
    coord_y=90.7,
    bat_side="R",
    venue_id="19",
    temp_f=85,
    roof_closed=False,
)


@pytest.fixture(scope="session")
def build_golden():
    """Factory returning the single-row feature frame for the golden input."""
    from Model.feature_engineering import create_features_for_prediction

    def _build(**overrides):
        kwargs = dict(GOLDEN_INPUT)
        kwargs.update(overrides)
        return create_features_for_prediction(**kwargs)

    return _build


@pytest.fixture(scope="session")
def golden_frame(build_golden):
    return build_golden()
