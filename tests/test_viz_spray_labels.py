"""Spray-chart player labels avoid the fence-distance texts.

On spray charts, ``_place_spray_labels`` scores 8 candidate offsets per player
label against the other player labels and the axis bounds — but not against the
three fence-distance texts (``328'`` / ``410'`` style) that
``draw_baseball_field`` places just outside the fence. A wall-scraper's label
could therefore sit directly on top of a distance text (seen live on gamePk
824473).

The fix: ``draw_baseball_field`` returns the three ``(x, y, ha)`` label anchors
it already computes, and ``_place_spray_labels`` accepts them via a keyword-only
``extra_obstacles`` parameter, seeding its ``placed`` list so the existing
overlap scoring steers labels clear. These tests pin both the new avoidance and
the unchanged default placement for balls nowhere near a fence text.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from Simulator.visualizations import _place_spray_labels  # noqa: E402

# Mirrors of the constants inside _place_spray_labels: approximate label size
# in data coords, and the point-offset-to-data scale of x_extent / 400.
LABEL_HALF_W = 18
LABEL_HALF_H = 8
X_EXTENT = 200
AXIS_LIMIT = 180
SCALE = X_EXTENT / 400.0

BALL = {"x": 0, "y": 100, "xbases": 4.0, "last_name": "Test"}


def _place_one(extra_obstacles):
    """Label BALL on a bare axes; return the chosen (dx, dy) point offset."""
    fig, ax = plt.subplots()
    try:
        _place_spray_labels(ax, [BALL], X_EXTENT, AXIS_LIMIT,
                            min_xbases=1.0, extra_obstacles=extra_obstacles)
        return ax.texts[-1].xyann
    finally:
        plt.close(fig)


def test_default_placement_is_unchanged_without_obstacles():
    """With no obstacles the label keeps its current below-right spot.

    (12, -12) is the closest candidate plus the below bonus. If this moves,
    the fix changed placement for balls nowhere near a fence text.
    """
    assert _place_one(extra_obstacles=None) == (12, -12)


def test_label_moves_off_a_fence_distance_text():
    """An obstacle on the default spot pushes the label a clear step away.

    The obstacle sits exactly where the default label anchor would land
    (adjusted for ha='left' / va='bottom' the way the fix adjusts fence
    texts), so keeping (12, -12) would mean drawing on top of it.
    """
    obstacle = (BALL["x"] + 12 * SCALE, BALL["y"] - 12 * SCALE, "left")
    dx, dy = _place_one(extra_obstacles=[obstacle])

    assert (dx, dy) != (12, -12)

    # The chosen anchor must clear the obstacle's adjusted centre by at
    # least one full label footprint (normalized Chebyshev distance >= 1).
    ob_cx = obstacle[0] + LABEL_HALF_W * 0.5  # ha='left'
    ob_cy = obstacle[1] + LABEL_HALF_H * 0.5  # va='bottom'
    anchor_x = BALL["x"] + dx * SCALE
    anchor_y = BALL["y"] + dy * SCALE
    dist = max(abs(anchor_x - ob_cx) / LABEL_HALF_W,
               abs(anchor_y - ob_cy) / LABEL_HALF_H)
    assert dist >= 1.0, f"label anchor only {dist:.2f} footprints from obstacle"
