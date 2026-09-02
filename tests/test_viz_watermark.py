"""Watermark geometry inside the title strip.

The watermark (logo + ``Data: MLB  |  @mlb_simulator``) used to be pasted onto
the saved PNG by PIL, sized as a percentage of the image's shorter side. That
made the logo 55 px tall on the run-distribution chart and 117 px on the player
chart, and let the handle text land on top of the divider rule on some charts
and under it on others.

It is now drawn in matplotlib inside the title strip: a points-sized logo
right-aligned on the title row, the handle right-aligned on the first subtitle
row, with the rule's band left empty between them. These tests pin that
geometry — the logo's physical size must not depend on figure size, and neither
element may drift into the rule.
"""

import inspect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from Simulator.style import title_axes, draw_title_block  # noqa: E402

# Strip-local y of the divider rule, as drawn by draw_title_block.
RULE_Y = 0.55
LOGO_PT = 30


def _fake_logo():
    """A 2:1 opaque RGBA block standing in for the real logo asset.

    Deliberately synthetic: these tests are about geometry, and a fixed
    aspect ratio makes the expected width exactly computable.
    """
    logo = np.zeros((100, 200, 4), dtype=np.uint8)
    logo[..., :3] = 40
    logo[..., 3] = 217
    return logo


def _build(figsize, handle="Data: MLB  |  @mlb_simulator"):
    """Render one title strip with a watermark; return (fig, strip, logo_axes)."""
    fig = plt.figure(figsize=figsize)
    tax = title_axes(fig, height_frac=0.13, top_pad=0.02)
    draw_title_block(tax, "Distribution of Runs Scored", ["Subtitle line one"],
                     title_size=20, subtitle_size=11,
                     logo=_fake_logo(), handle=handle, logo_pt=LOGO_PT)
    fig.canvas.draw()
    logo_axes = [a for a in fig.axes if a.get_label() == "watermark_logo"]
    return fig, tax, logo_axes


def _handle_text(tax, handle):
    """The Text artist carrying the handle, from the strip's children."""
    matches = [t for t in tax.texts if t.get_text() == handle]
    assert len(matches) == 1, f"expected one handle text, found {len(matches)}"
    return matches[0]


def test_logo_height_is_fixed_in_points_across_figure_sizes():
    """A points-sized logo renders the same physical height on any figure.

    This is the whole point of the change: the PIL paste scaled the logo to
    4.5% of the shorter side, so a tall chart got a much bigger logo than a
    wide one. Height in points must be independent of figsize.
    """
    fig_a, _, axes_a = _build((12, 8.5))
    fig_b, _, axes_b = _build((20, 10))
    try:
        assert len(axes_a) == 1 and len(axes_b) == 1, "expected one watermark_logo axes"
        assert fig_a.dpi == fig_b.dpi, "test assumes both figures share a dpi"

        px_a = axes_a[0].get_position().height * fig_a.get_size_inches()[1] * fig_a.dpi
        px_b = axes_b[0].get_position().height * fig_b.get_size_inches()[1] * fig_b.dpi
        expected = round(LOGO_PT / 72 * fig_a.dpi)

        assert abs(px_a - px_b) <= 1, (
            f"logo height differs between figure sizes: {px_a:.2f} px vs {px_b:.2f} px"
        )
        assert abs(px_a - expected) <= 1, (
            f"logo height {px_a:.2f} px is not {LOGO_PT}pt ({expected} px) at "
            f"dpi {fig_a.dpi}"
        )
    finally:
        plt.close(fig_a)
        plt.close(fig_b)


def test_logo_sits_above_the_rule_and_handle_below_it():
    """Nothing is ever drawn in the rule's band.

    The old paste put the handle on top of the rule on some charts. The logo
    belongs on the title row (above the rule); the handle belongs on the first
    subtitle row (below it).
    """
    handle = "Data: MLB  |  @mlb_simulator"
    fig, tax, logo_axes = _build((12, 8.5), handle=handle)
    try:
        pos = tax.get_position()
        rule_y_fig = pos.y0 + RULE_Y * pos.height

        logo_bottom = logo_axes[0].get_position().y0
        assert logo_bottom > rule_y_fig, (
            f"logo bottom {logo_bottom:.4f} is not above the rule at "
            f"{rule_y_fig:.4f} (figure fraction)"
        )

        extent = _handle_text(tax, handle).get_window_extent(
            renderer=fig.canvas.get_renderer())
        handle_top = extent.y1 / (fig.get_size_inches()[1] * fig.dpi)
        assert handle_top < rule_y_fig, (
            f"handle top {handle_top:.4f} is not below the rule at "
            f"{rule_y_fig:.4f} (figure fraction)"
        )
    finally:
        plt.close(fig)


def test_logo_and_handle_are_flush_with_the_strip_right_edge():
    """Both watermark elements right-align with the rule's right end."""
    handle = "Data: MLB  |  @mlb_simulator"
    fig, tax, logo_axes = _build((12, 8.5), handle=handle)
    try:
        px_per_frac = fig.get_size_inches()[0] * fig.dpi
        target = tax.get_position().x1

        logo_right = logo_axes[0].get_position().x1
        assert abs(logo_right - target) * px_per_frac <= 1, (
            f"logo right edge {logo_right:.5f} is not flush with strip right "
            f"edge {target:.5f}"
        )

        extent = _handle_text(tax, handle).get_window_extent(
            renderer=fig.canvas.get_renderer())
        handle_right = extent.x1 / px_per_frac
        assert abs(handle_right - target) * px_per_frac <= 1, (
            f"handle right edge {handle_right:.5f} is not flush with strip "
            f"right edge {target:.5f}"
        )
    finally:
        plt.close(fig)


def test_only_the_retired_chart_still_pastes_the_watermark():
    """The four social charts draw the watermark; only la_ev_graph pastes it.

    la_ev_graph is retired and keeps the PIL path, so _apply_watermark and
    _watermark_logo must survive — but no social chart may still call them.
    """
    from Simulator import visualizations as viz

    for name in ("run_dist", "spray_chart", "create_estimated_bases_table",
                 "player_contribution_chart"):
        src = inspect.getsource(getattr(viz, name))
        assert "_apply_watermark" not in src, (
            f"{name} still applies the PIL watermark paste"
        )

    assert "_apply_watermark" in inspect.getsource(viz.la_ev_graph), (
        "la_ev_graph should keep the PIL watermark paste"
    )
