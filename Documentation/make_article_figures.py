#!/usr/bin/env python3
"""Render the two README article figures.

Usage, from the repo root:

    python Documentation/make_article_figures.py                    # both figures
    python Documentation/make_article_figures.py --only hr_check    # no network

``article_one_ball.png`` draws one batted ball from the default game as five
bars, one per outcome the model gives it a chance of. It fetches the game from
the MLB Stats API and scores it with the shipped model through
``calculate_total_bases()``, the same call the estimated bases table uses, so
the five bars are the numbers behind that table's row. It also prints the
ball's full feature row and the raw weather the feed reported.

``article_hr_check.png`` puts the model's home run rate beside the actual one
for each exit-velocity bin in ``Model/model_metadata.json``. It reads only
that file, so it needs no network.

Both are drawn with the same title strip, fonts and background as the four
game charts (``Simulator/style.py``). The four game charts themselves come
from ``make_readme_images.py``.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Same game, seed and simulation count as make_readme_images.py.
# 2025 World Series Game 7: Dodgers 5, Blue Jays 4 (11 innings) at Rogers Centre.
DEFAULT_GAME_PK = 813024
DEFAULT_SEED = 20251101
DEFAULT_SIMS = 10000
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "Documentation", "Images")
METADATA_PATH = os.path.join(REPO_ROOT, "Model", "model_metadata.json")

# Figure 1's ball: rank 6 in the default game's estimated bases table.
ONE_BALL_PLAYER = "Tommy Edman"
ONE_BALL_SPEED = 101.3   # mph, as the table shows it
ONE_BALL_ANGLE = 42      # degrees, as the table shows it
ONE_BALL_GAME = "2025 World Series Game 7"

# Figure 2 draws the first six exit-velocity bins. The seventh (115-120 mph)
# holds 20 balls, too few for a rate worth drawing; the subtitle says so.
HR_CHECK_MAX_BINS = 6
# How the subtitle names the balls in the metadata's home run table.
HR_CHECK_SAMPLE = "batted balls from the model's own home run check"

# (column from calculate_total_bases, bar label, PALETTE key, feed eventType)
OUTCOMES = [
    ("out_prob", "Out", "out", "out"),
    ("single_prob", "Single", "single", "single"),
    ("double_prob", "Double", "xbh", "double"),
    ("triple_prob", "Triple", "xbh", "triple"),
    ("hr_prob", "Home run", "hr", "home_run"),
]

FILE_ONE_BALL = "article_one_ball.png"
FILE_HR_CHECK = "article_hr_check.png"


# ---------------------------------------------------------------------------
# Pure helpers (no model, no network; pinned by tests)
# ---------------------------------------------------------------------------

def _bin_start(label):
    return float(label.split("-")[0])


def _sorted_bins(metadata):
    bins = metadata["tail_metrics"]["hr_rmse_by_ev_bin"]
    return sorted(bins.items(), key=lambda kv: _bin_start(kv[0]))


def hr_check_rows(metadata, max_bins=HR_CHECK_MAX_BINS):
    """``[(bin label, model HR rate, actual HR rate, count), ...]``, slowest first.

    Only the first ``max_bins`` bins by exit velocity are returned.
    """
    return [(label, b["model_hr_rate"], b["actual_hr_rate"], b["count"])
            for label, b in _sorted_bins(metadata)[:max_bins]]


def hr_check_subtitle(metadata, max_bins=HR_CHECK_MAX_BINS):
    """One subtitle line: the drawn sample and any bin left out.

    The title and the axis labels say what is plotted, which keeps this line
    short enough to clear the stamp on the right of the same row.
    """
    kept = hr_check_rows(metadata, max_bins)
    dropped = _sorted_bins(metadata)[max_bins:]
    n = sum(count for *_, count in kept)
    parts = [f"{n:,} {HR_CHECK_SAMPLE}"]
    for label, b in dropped:
        parts.append(f"{label} mph ({b['count']:,} balls) not shown")
    return ". ".join(parts)


def pick_ball(frame, player, launch_speed, launch_angle):
    """Row position of the one ball by ``player`` at this speed and angle.

    Matches the rounding the estimated bases table displays (0.1 mph, whole
    degrees). Raises ``ValueError`` unless exactly one row matches.
    """
    speed = frame["launch_speed"].astype(float)
    angle = frame["launch_angle"].astype(float)
    hit = ((frame["player"] == player)
           & ((speed - launch_speed).abs() < 0.05 + 1e-9)
           & ((angle - launch_angle).abs() < 0.5))
    positions = [i for i, flag in enumerate(hit.fillna(False)) if flag]
    if len(positions) != 1:
        raise ValueError(f"expected exactly one ball for {player} at "
                         f"{launch_speed} mph and {launch_angle} degrees, "
                         f"found {len(positions)}")
    return positions[0]


def outcome_bars(row):
    """``[(label, probability, PALETTE key), ...]`` in OUTCOMES order."""
    return [(label, float(row[col]), key) for col, label, key, _ in OUTCOMES]


def actual_label(event_type):
    """The bar label for the feed's ``eventType``, or None if it has no bar."""
    for _, label, _, event in OUTCOMES:
        if event == event_type:
            return label
    return None


def pct_label(p):
    """Whole-percent label; a chance under 1% but above 0 reads ``<1%``."""
    if 0 < p < 0.005:
        return "<1%"
    return f"{p:.0%}"


# ---------------------------------------------------------------------------
# Drawing (imports the model and the shipped chart style)
# ---------------------------------------------------------------------------

def _stamp():
    """The top-right stamp the four game charts carry: (logo, handle, site)."""
    # Imported here: Simulator.visualizations loads the model pickle.
    from Simulator.visualizations import (
        WATERMARK_HANDLE, WATERMARK_SITE, _watermark_logo_rgba,
    )
    return _watermark_logo_rgba(), WATERMARK_HANDLE, WATERMARK_SITE


def build_one_ball_figure(bars, title, subtitle, actual=None):
    """Five horizontal bars, Out on top, with each chance beside its bar.

    ``actual`` is the label of the bar that matches what really happened; it
    gets a muted "what happened" note after its percentage.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from Simulator.style import PALETTE, apply_base_style, draw_title_block, title_axes

    apply_base_style()
    fig = plt.figure(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax = fig.add_axes((0.12, 0.11, 0.84, 0.66))
    ax.set_facecolor(PALETTE["bg"])

    labels = [label for label, _, _ in bars]
    probs = [p for _, p, _ in bars]
    ys = list(range(len(bars)))[::-1]   # first bar on top
    ax.barh(ys, probs, height=0.62, color=[PALETTE[key] for _, _, key in bars],
            zorder=3)
    for y, label, p in zip(ys, labels, probs):
        pct = ax.text(p + 0.012, y, pct_label(p), va="center", ha="left",
                      fontsize=13, fontweight="bold", color=PALETTE["text"],
                      zorder=4)
        if label == actual:
            ax.annotate("what happened", xy=(1, 0.5), xycoords=pct,
                        xytext=(8, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=12,
                        fontstyle="italic", color=PALETTE["text_muted"])

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=13, color=PALETTE["text"])
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Model's chance of each result", fontsize=12, labelpad=8,
                  color=PALETTE["text"])
    ax.grid(True, axis="x", linestyle="--", alpha=0.45, color=PALETTE["grid"])
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["spine"])

    tax = title_axes(fig, height_frac=0.17, top_pad=0.02)
    logo, handle, site = _stamp()
    draw_title_block(tax, title, [subtitle], title_size=20, subtitle_size=11,
                     logo=logo, handle=handle, site=site)
    return fig


def build_hr_check_figure(rows, title, subtitle):
    """Model and actual home run rate side by side for each exit-velocity bin."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter
    from Simulator.style import PALETTE, apply_base_style, draw_title_block, title_axes

    apply_base_style()
    fig = plt.figure(figsize=(12, 7.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax = fig.add_axes((0.08, 0.11, 0.89, 0.69))
    ax.set_facecolor(PALETTE["bg"])

    xs = np.arange(len(rows))
    width = 0.38
    model = [m for _, m, _, _ in rows]
    actual = [a for _, _, a, _ in rows]
    ax.bar(xs - width / 2, model, width, color=PALETTE["hr"], label="Model",
           zorder=3)
    ax.bar(xs + width / 2, actual, width, color=PALETTE["text_muted"],
           label="What happened", zorder=3)
    top = max(model + actual)
    for x, m, a in zip(xs, model, actual):
        for dx, v in ((-width / 2, m), (width / 2, a)):
            ax.text(x + dx, v + top * 0.015, f"{v:.1%}", ha="center", va="bottom",
                    fontsize=10, color=PALETTE["text"], zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels([label for label, *_ in rows], fontsize=11,
                       color=PALETTE["text"])
    ax.set_xlabel("Exit velocity (mph)", fontsize=12, labelpad=8,
                  color=PALETTE["text"])
    ax.set_ylabel("Share that were home runs", fontsize=12, labelpad=8,
                  color=PALETTE["text"])
    ax.set_ylim(0, top * 1.12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(True, axis="y", linestyle="--", alpha=0.45, color=PALETTE["grid"])
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_color(PALETTE["spine"])
    ax.spines["left"].set_color(PALETTE["spine"])
    ax.legend(loc="upper left", frameon=False, fontsize=11)

    tax = title_axes(fig, height_frac=0.15, top_pad=0.02)
    logo, handle, site = _stamp()
    draw_title_block(tax, title, [subtitle], title_size=20, subtitle_size=11,
                     logo=logo, handle=handle, site=site)
    return fig


def render_hr_check(out_dir, metadata_path=METADATA_PATH):
    """Write ``article_hr_check.png`` from the metadata file. Returns its path."""
    from Simulator.style import finalize

    with open(metadata_path) as fh:
        metadata = json.load(fh)
    rows = hr_check_rows(metadata)
    subtitle = hr_check_subtitle(metadata)
    print("Figure 2 rows (bin, model HR rate, actual HR rate, count):")
    for row in rows:
        print("  ", row)
    print("Figure 2 subtitle:", subtitle)
    fig = build_hr_check_figure(
        rows, "Does the model's home run rate match what happened?", subtitle)
    path = os.path.join(out_dir, FILE_HR_CHECK)
    finalize(fig, path, dpi=200)
    return path


def render_one_ball(game_pk, out_dir, num_simulations, seed):
    """Fetch the game, score every ball, draw the one. Returns the PNG's path.

    Also prints the game's win shares from the same seeded run as
    make_readme_images.py, so this figure's game state can be checked
    against the README's four game charts.
    """
    import pandas as pd
    from Simulator.constants import base_url, schedule_ver
    from Simulator.get_game_information import get_game_info, response_code, team_info
    from Simulator.game_simulator import (
        calculate_total_bases, outcomes, prepare_batted_ball_features, simulator,
    )
    from Simulator.style import finalize

    from make_readme_images import game_context   # same folder

    schedule = response_code(base_url, schedule_ver, f"schedule?sportId=1&gamePk={game_pk}")
    ctx = game_context(schedule, team_info())
    print(f"{ctx['away_team']} {ctx['away_score']} @ {ctx['home_team']} {ctx['home_score']}, "
          f"{ctx['venue']}, {ctx['formatted_date']}")

    game_data, _all_pbp, steals_and_pickoffs = get_game_info(game_pk, all_columns=False)
    game_data["venue.name"] = ctx["venue"]
    steals_and_pickoffs = steals_and_pickoffs.drop_duplicates(subset=["startTime"], keep="first")

    home_outcomes = outcomes(game_data, steals_and_pickoffs, "home")
    away_outcomes = outcomes(game_data, steals_and_pickoffs, "away")

    _, _, home_wp, away_wp, tie_pct = simulator(
        num_simulations,
        [o[0] for o in home_outcomes],
        [o[0] for o in away_outcomes],
        seed=seed,
    )
    print(f"DTW: {ctx['away_team']} {away_wp}% - {home_wp}% {ctx['home_team']} (tie {tie_pct}%)")

    # One scoring call over both teams: row i of the frame is outcome i.
    all_outcomes = away_outcomes + home_outcomes
    frame = calculate_total_bases(all_outcomes)
    pos = pick_ball(frame, ONE_BALL_PLAYER, ONE_BALL_SPEED, ONE_BALL_ANGLE)
    row = frame.iloc[pos]
    ball = all_outcomes[pos][0]
    bars = outcome_bars(row)

    print(f"Figure 1 ball: {ONE_BALL_PLAYER}, {row['launch_speed']} mph, "
          f"{row['launch_angle']} degrees, inning {ball.get('inning')}, "
          f"top of inning {ball.get('is_top_inning')}, result {all_outcomes[pos][1]}")
    play = game_data[game_data["playId"] == ball.get("play_id")]
    for col in ("description", "outs", "hitData.totalDistance", "batSide.code",
                "weather_temp_f", "weather_condition"):
        if col in play.columns and len(play):
            print(f"  feed {col}: {play.iloc[0][col]}")
    print("  roof_closed passed to the model:", ball.get("roof_closed"))
    for label, p, _ in bars:
        print(f"  {label}: {p:.4f}")
    print(f"  estimated bases: {row['estimated_bases']:.4f}")

    features = prepare_batted_ball_features(
        launch_speed=ball["launch_speed"], launch_angle=ball["launch_angle"],
        venue_name=ball["venue_name"], coord_x=ball.get("coord_x"),
        coord_y=ball.get("coord_y"), bat_side=ball.get("bat_side"),
        temp_f=ball.get("temp_f"), roof_closed=ball.get("roof_closed", False),
    )
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print("Figure 1 ball, full feature row:")
        print(features.T.to_string(header=False))

    speed = f"{float(row['launch_speed']):.1f}"
    angle = f"{float(row['launch_angle']):.0f}"
    subtitle = (f"Chance of each result for {ONE_BALL_PLAYER}'s {speed} mph, "
                f"{angle}\N{DEGREE SIGN} fly ball in {ONE_BALL_GAME}")
    surname = ONE_BALL_PLAYER.split()[-1]
    fig = build_one_ball_figure(bars, f"What should {surname}'s fly ball have been?",
                                subtitle, actual=actual_label(all_outcomes[pos][1]))
    path = os.path.join(out_dir, FILE_ONE_BALL)
    finalize(fig, path, dpi=200)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--game-pk", type=int, default=DEFAULT_GAME_PK)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--only", choices=("one_ball", "hr_check"),
                        help="draw one figure; hr_check needs no network")
    args = parser.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    paths = []
    if args.only in (None, "one_ball"):
        paths.append(render_one_ball(args.game_pk, args.out_dir, args.sims, args.seed))
    if args.only in (None, "hr_check"):
        paths.append(render_hr_check(args.out_dir))
    for path in paths:
        print("wrote", path)


if __name__ == "__main__":
    main()
