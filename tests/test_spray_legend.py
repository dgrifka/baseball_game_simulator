"""The Estimated Bases legend lives below the title strip, not inside it.

``spray_chart`` mounts a horizontal colorbar plus an ``Estimated Bases`` label
and an ``Out/1B/2B/3B/HR`` row between the title strip and the two field axes.
Those three pieces were positioned with literals chosen for the old cream
header (``0.808`` / ``0.852``), so once the strip became a filled band they
straddled its bottom edge: dark label text on the band, the bar half in and
half out.

These tests pin the legend to the strip's own geometry — everything below the
band's bottom edge, above the team headings — and check it with the Agg
renderer on both the default palette and a filled band.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from Simulator import visualizations as viz  # noqa: E402
from Simulator.style import PALETTE  # noqa: E402

# The strip's own numbers, as spray_chart passes them to title_axes().
STRIP_HEIGHT_FRAC = 0.16
STRIP_TOP_PAD = 0.02
STRIP_BOTTOM = 1.0 - STRIP_HEIGHT_FRAC - STRIP_TOP_PAD

OUTCOME_LABELS = ('Out', '1B', '2B', '3B', 'HR')


class _FixedPipeline:
    """Stand-in model: every batted ball is a 3.35-expected-base scorcher.

    Deterministic and instant — these tests are about geometry, and loading
    the real pickle would make them a model test by accident.
    """

    def predict_proba(self, features):
        return np.array([[0.05, 0.05, 0.10, 0.10, 0.70]])


def _ball(name, coord_x, coord_y):
    """One (outcome_dict, _, player_name) tuple in the shape outcomes() emits."""
    data = {
        'launch_speed': 102.0,
        'launch_angle': 26.0,
        'total_distance': 400.0,
        'coord_x': coord_x,
        'coord_y': coord_y,
        'bat_side': 'R',
        'temp_f': 72.0,
        'roof_closed': False,
    }
    return (data, 'home_run', name)


def _measure(fig):
    """Draw the figure and return the artist boxes this module asserts on.

    Everything comes back in figure fractions so the assertions can be
    written against the strip geometry directly.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def frac(artist):
        return artist.get_window_extent(renderer=renderer).transformed(inv)

    label = [t for t in fig.texts if t.get_text() == 'Estimated Bases']
    assert len(label) == 1, f"expected one legend label, found {len(label)}"

    outcome_row = [t for t in fig.texts if t.get_text() in OUTCOME_LABELS]
    assert len(outcome_row) == len(OUTCOME_LABELS), (
        f"expected {len(OUTCOME_LABELS)} outcome labels, found {len(outcome_row)}"
    )

    headings = [a.title for a in fig.axes if a.get_title()]
    assert headings, "expected at least one team heading"

    # The colorbar axes is the only one whose height is a sliver.
    bars = [a for a in fig.axes
            if 0 < a.get_position().height < 0.05 and a.get_position().width > 0.2]
    assert len(bars) == 1, f"expected one colorbar axes, found {len(bars)}"

    return {
        'label': frac(label[0]),
        'bar': bars[0].get_position(),
        'outcomes': [frac(t) for t in outcome_row],
        'headings': [frac(t) for t in headings],
    }


@pytest.fixture
def spray_boxes(monkeypatch, tmp_path):
    """Render a two-ball-per-side spray chart; return the measured boxes.

    ``finalize`` is swapped out so nothing is written to disk and the figure
    is measured while it is still alive (``spray_chart`` closes it on the way
    out).
    """
    def _render():
        captured = {}

        def fake_finalize(fig, filepath, **kwargs):
            captured['boxes'] = _measure(fig)
            return filepath

        monkeypatch.setattr(viz, 'finalize', fake_finalize)

        home = [_ball('Riley Greene', 100.0, 80.0), _ball('Spencer Torkelson', 150.0, 95.0)]
        away = [_ball('Luis Robert', 95.0, 85.0), _ball('Andrew Vaughn', 160.0, 100.0)]
        viz.spray_chart(home, away, 'Tigers', 'White Sox', 3, 1,
                        55.0, 40.0, 5.0, [], '09/19/2026',
                        images_dir=str(tmp_path), pipeline=_FixedPipeline())
        return captured['boxes']

    return _render


@pytest.fixture
def banded_palette():
    """Palette with a filled band, restored afterwards."""
    saved = dict(PALETTE)
    PALETTE.update(band='#123456', title_ink='#FFFFFF')
    yield
    PALETTE.clear()
    PALETTE.update(saved)


def _assert_legend_below_strip(boxes):
    label, bar = boxes['label'], boxes['bar']

    assert label.y1 < STRIP_BOTTOM - 0.01, (
        f"legend label top at {label.y1:.4f} is not clear of the strip bottom "
        f"({STRIP_BOTTOM:.4f})"
    )
    assert bar.y1 < label.y0, (
        f"colorbar top at {bar.y1:.4f} overlaps the label bottom ({label.y0:.4f})"
    )

    lowest_legend = min(b.y0 for b in boxes['outcomes'])
    highest_heading = max(b.y1 for b in boxes['headings'])
    assert lowest_legend > highest_heading, (
        f"outcome row bottom at {lowest_legend:.4f} collides with the tallest "
        f"team heading top ({highest_heading:.4f})"
    )


def test_legend_sits_below_the_strip_on_the_default_palette(spray_boxes):
    _assert_legend_below_strip(spray_boxes())


def test_legend_sits_below_the_strip_with_a_filled_band(spray_boxes, banded_palette):
    _assert_legend_below_strip(spray_boxes())
