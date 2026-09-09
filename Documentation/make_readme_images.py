#!/usr/bin/env python3
"""Render the four README example charts for one MLB game.

Usage, from the repo root (network access required):

    python Documentation/make_readme_images.py                # default game
    python Documentation/make_readme_images.py --game-pk 813024 --seed 7

Fetches the game's play-by-play from the MLB Stats API, runs the simulator
with a fixed seed, and writes the spray chart, run distribution, estimated
bases table and player contribution chart to ``Documentation/Images/``. This
is the same fetch -> outcomes -> simulate -> render chain the daily pipeline
runs; there is no separate rendering path, so the README pictures always
reflect the shipped chart code.

The two validation plots in the README (``spray_angle_validation.png`` and
``ev_la_spray_angle.png``) come from ``readme_image_generator.ipynb`` instead.
"""
import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 2025 World Series Game 7: Dodgers 5, Blue Jays 4 (11 innings) at Rogers Centre.
DEFAULT_GAME_PK = 813024
DEFAULT_SEED = 20251101
DEFAULT_SIMS = 10000
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "Documentation", "Images")


def game_context(schedule_json, teams_df):
    """Team short names, scores, venue and display date from a schedule payload.

    ``teams_df`` is ``team_info()``'s frame; a participant that is not one of
    the 30 clubs (the All-Star squads) falls back to the API's full team name,
    which is the key the logo and color tables use for them.
    """
    game = schedule_json["dates"][0]["games"][0]
    short_names = dict(zip(teams_df["team.id"], teams_df["teamName"]))

    def side(key):
        team = game["teams"][key]
        name = short_names.get(team["team"]["id"], team["team"]["name"])
        return name, int(team["score"])

    home_team, home_score = side("home")
    away_team, away_score = side("away")
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "venue": game["venue"]["name"],
        "formatted_date": pd.to_datetime(game["officialDate"]).strftime("%m/%d/%Y"),
    }


def player_id_map(game_data):
    """``{full_name: player_id}`` for headshots; batters win a name collision."""
    mapping = {}
    for role in ("batter", "pitcher"):
        name_col, id_col = f"{role}.fullName", f"{role}.id"
        if name_col not in game_data.columns or id_col not in game_data.columns:
            continue
        pairs = game_data[[name_col, id_col]].dropna().drop_duplicates()
        for name, pid in zip(pairs[name_col], pairs[id_col]):
            mapping.setdefault(name, int(pid))
    return mapping


def render(game_pk, out_dir, num_simulations, seed):
    """Fetch one game, simulate it, write the four charts. Returns their paths.

    Only ``spray_chart`` returns its path; the other three return None, so the
    paths are recovered from the shared filename prefix every chart uses.
    """
    # Imported here so the pure helpers above stay importable without loading
    # the model pickle (game_simulator loads it at import time).
    from Simulator.constants import base_url, mlb_team_logos, schedule_ver
    from Simulator.get_game_information import get_game_info, response_code, team_info
    from Simulator.game_simulator import (
        calculate_total_bases, create_detailed_outcomes_df, outcome_rankings,
        outcomes, simulator,
    )
    from Simulator.visualizations import (
        create_estimated_bases_table, player_contribution_chart, run_dist,
        spray_chart,
    )

    schedule = response_code(base_url, schedule_ver, f"schedule?sportId=1&gamePk={game_pk}")
    ctx = game_context(schedule, team_info())
    print(f"{ctx['away_team']} {ctx['away_score']} @ {ctx['home_team']} {ctx['home_score']}, "
          f"{ctx['venue']}, {ctx['formatted_date']}")

    game_data, _all_pbp, steals_and_pickoffs = get_game_info(game_pk, all_columns=False)
    game_data["venue.name"] = ctx["venue"]
    steals_and_pickoffs = steals_and_pickoffs.drop_duplicates(subset=["startTime"], keep="first")

    home_outcomes = outcomes(game_data, steals_and_pickoffs, "home")
    away_outcomes = outcomes(game_data, steals_and_pickoffs, "away")

    home_bases = calculate_total_bases(home_outcomes)
    away_bases = calculate_total_bases(away_outcomes)

    def bases_lookup(df):
        if "play_id" not in df.columns:
            return {}
        valid = df.dropna(subset=["play_id"])
        return dict(zip(valid["play_id"], valid["estimated_bases"]))

    home_detailed = create_detailed_outcomes_df(game_data, steals_and_pickoffs, "home")
    home_detailed["team"] = ctx["home_team"]
    away_detailed = create_detailed_outcomes_df(game_data, steals_and_pickoffs, "away")
    away_detailed["team"] = ctx["away_team"]
    top_15 = outcome_rankings(home_detailed, away_detailed)

    home_runs, away_runs, home_wp, away_wp, tie_pct = simulator(
        num_simulations,
        [o[0] for o in home_outcomes],
        [o[0] for o in away_outcomes],
        seed=seed,
    )
    print(f"DTW: {ctx['away_team']} {away_wp}% - {home_wp}% {ctx['home_team']} (tie {tie_pct}%)")

    common = (ctx["home_team"], ctx["away_team"], ctx["home_score"], ctx["away_score"],
              home_wp, away_wp, tie_pct)
    charts = [
        spray_chart(home_outcomes, away_outcomes, *common, mlb_team_logos,
                    ctx["formatted_date"], venue_name=ctx["venue"], images_dir=out_dir),
        run_dist(num_simulations, home_runs, away_runs, *common,
                 ctx["formatted_date"], out_dir),
        create_estimated_bases_table(top_15, ctx["away_team"], ctx["home_team"],
                                     ctx["away_score"], ctx["home_score"], away_wp, home_wp,
                                     ctx["formatted_date"], mlb_team_logos, out_dir),
        player_contribution_chart(home_outcomes, away_outcomes, *common, mlb_team_logos,
                                  ctx["formatted_date"], out_dir,
                                  player_id_map=player_id_map(game_data),
                                  precomputed_home=bases_lookup(home_bases),
                                  precomputed_away=bases_lookup(away_bases)),
    ]
    del charts  # only spray_chart returns a path; recover all four by prefix
    prefix = f"{ctx['away_team']}_{ctx['home_team']}_{ctx['away_score']}-{ctx['home_score']}--"
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir)
                  if f.startswith(prefix) and f.endswith(".png"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--game-pk", type=int, default=DEFAULT_GAME_PK)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    for path in render(args.game_pk, args.out_dir, args.sims, args.seed):
        print("wrote", path)


if __name__ == "__main__":
    main()
