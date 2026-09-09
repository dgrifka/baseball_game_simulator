"""Documentation/make_readme_images.py: the pure helpers, no network.

The script itself needs the MLB Stats API; these tests pin the two
payload-shaping helpers so a schedule-format change or a renamed column breaks
here rather than halfway through a README regeneration.
"""
import importlib.util
import os

import pandas as pd
import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), os.pardir,
                       "Documentation", "make_readme_images.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("make_readme_images", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _schedule():
    return {"dates": [{"games": [{
        "gamePk": 813024,
        "officialDate": "2025-11-01",
        "venue": {"name": "Rogers Centre", "id": 14},
        "teams": {
            "away": {"team": {"id": 119, "name": "Los Angeles Dodgers"}, "score": 5},
            "home": {"team": {"id": 141, "name": "Toronto Blue Jays"}, "score": 4},
        },
    }]}]}


def _teams():
    return pd.DataFrame({"team.id": [119, 141], "teamName": ["Dodgers", "Blue Jays"]})


def test_game_context_uses_short_names_scores_venue_and_date(mod):
    ctx = mod.game_context(_schedule(), _teams())
    assert ctx == {
        "home_team": "Blue Jays", "away_team": "Dodgers",
        "home_score": 4, "away_score": 5,
        "venue": "Rogers Centre", "formatted_date": "11/01/2025",
    }


def test_game_context_falls_back_to_full_name_for_unknown_club(mod):
    sched = _schedule()
    sched["dates"][0]["games"][0]["teams"]["away"]["team"] = {"id": 159, "name": "American League All-Stars"}
    ctx = mod.game_context(sched, _teams())
    assert ctx["away_team"] == "American League All-Stars"


def test_player_id_map_merges_batters_and_pitchers(mod):
    game_data = pd.DataFrame({
        "batter.fullName": ["Shohei Ohtani", "Shohei Ohtani", None],
        "batter.id": [660271, 660271, None],
        "pitcher.fullName": ["Shohei Ohtani", "Max Scherzer", "Max Scherzer"],
        "pitcher.id": [660271, 453286, 453286],
    })
    assert mod.player_id_map(game_data) == {"Shohei Ohtani": 660271, "Max Scherzer": 453286}


def test_player_id_map_tolerates_missing_columns(mod):
    assert mod.player_id_map(pd.DataFrame({"eventType": ["out"]})) == {}
