"""
Import-smoke: every module the simulator/daily pipeline depends on must import
cleanly. This is the guard that would have caught the dead `best_batted_balls.py`
landmine — a top-level `FileNotFoundError` on a deleted model path made that
module unimportable while nothing flagged it (it was imported by nothing).

Any module that loads a model/CSV or runs a self-test at import time (e.g.
game_simulator, feature_engineering) is exercised here, so a future stale-path
or missing-asset break turns into a red test instead of a silent production
failure.
"""
import importlib

import pytest

# Modules loaded by the live simulation / daily-bake path (repo root is on
# sys.path via tests/conftest.py). Kept explicit rather than globbed so a new
# module is a deliberate addition and a deleted one fails loudly here.
MODULES = [
    "Simulator.constants",
    "Simulator.team_mapping",
    "Simulator.style",
    "Simulator.utils",
    "Simulator.get_game_information",
    "Simulator.game_simulator",
    "Simulator.vector_engine",
    "Simulator.visualizations",
    "Model.feature_engineering",
    "Model.data_loader",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports_cleanly(module_name):
    importlib.import_module(module_name)
