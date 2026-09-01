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
