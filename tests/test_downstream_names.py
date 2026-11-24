"""
Names the downstream pipeline's scheduled runs import from this repo.

The downstream pipeline pins this repo by commit. If one of these names is
renamed or deleted here, nothing in this repo used to notice, and the break
surfaced only at the pipeline's next pin bump. `test_import_smoke.py` checks
that each module imports; this file checks that each listed name is still in
its module, and the failure message names what went missing.

Names used only by that pipeline's own tests or one-off scripts are left out on
purpose, so an internal helper here can still be renamed freely. The list
changes only together with the pipeline code that imports these names.
"""
import importlib

import pytest

DOWNSTREAM_NAMES = {
    "Simulator.constants": [
        "DEFAULT_VENUE_ID", "NEUTRAL_SITE_VENUES", "VENUE_NAME_TO_ID",
        "espn_logo_urls", "mlb_team_logos", "season", "team_colors",
        "team_palette", "venue_names",
    ],
    "Simulator.game_simulator": [
        "calculate_total_bases", "create_detailed_outcomes_df", "outcome_rankings",
        "outcomes", "outcomes_by_inning", "simulator",
    ],
    "Simulator.get_game_information": ["fetch_games", "get_game_info", "team_info"],
    "Simulator.style": ["PALETTE"],
    "Simulator.visualizations": [
        "create_estimated_bases_table", "la_ev_graph", "player_contribution_chart",
        "run_dist", "spray_chart",
    ],
}


@pytest.mark.parametrize("module_name", sorted(DOWNSTREAM_NAMES))
def test_downstream_names_still_exist(module_name):
    module = importlib.import_module(module_name)
    missing = [n for n in DOWNSTREAM_NAMES[module_name] if not hasattr(module, n)]
    assert missing == [], f"{module_name} no longer exposes {missing}"
