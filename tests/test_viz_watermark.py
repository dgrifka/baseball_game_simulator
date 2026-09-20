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
import pytest  # noqa: E402

from Simulator.style import (  # noqa: E402
    PALETTE, title_axes, draw_title_block, fit_band_to_content, BAND_INSET,
)

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
        # The rule's strip-local y is computed per strip (points-based, centred
        # block) and recorded on the strip by draw_title_block.
        rule_y_fig = pos.y0 + tax._dtw_layout['rule_y'] * pos.height

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


def test_site_line_sits_one_subtitle_row_under_the_handle():
    """``site=`` draws a second right-aligned line directly under the handle.

    Same size and colour as the handle, flush with the strip's right edge,
    exactly one subtitle row (the strip's computed row pitch) lower.
    """
    handle, site = "Data: MLB  |  @mlb_simulator", "dtwbaseball.com"
    fig = plt.figure(figsize=(12, 8.5))
    tax = title_axes(fig, height_frac=0.13, top_pad=0.02)
    draw_title_block(tax, "Distribution of Runs Scored", ["Subtitle line one"],
                     title_size=20, subtitle_size=11,
                     logo=_fake_logo(), handle=handle, site=site, logo_pt=LOGO_PT)
    try:
        h = _handle_text(tax, handle)
        s = _handle_text(tax, site)
        pitch = tax._dtw_layout['pitch']
        assert s.get_position()[0] == 1.0
        assert s.get_position()[1] == pytest.approx(h.get_position()[1] - pitch)
        assert s.get_ha() == "right" and s.get_va() == "top"
        assert s.get_fontsize() == h.get_fontsize()
        assert s.get_color() == h.get_color()
    finally:
        plt.close(fig)


def test_subtitle_rows_stay_inside_the_strip():
    """Two subtitle rows plus the title fit inside a 0.13 strip on an 8.5in
    figure — the run-distribution geometry — with band above and below.

    The old fixed fractions (rows at 0.45 and 0.15 of the strip) put the second
    row's descenders on the band's bottom edge.
    """
    fig = plt.figure(figsize=(12, 8.5))
    tax = title_axes(fig, height_frac=0.13, top_pad=0.02)
    draw_title_block(tax, "Distribution of Runs Scored", ["Line one", "Line two"],
                     title_size=20, subtitle_size=11)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        pos = tax.get_position()
        fig_h_px = fig.get_size_inches()[1] * fig.dpi
        strip_bottom = pos.y0 * fig_h_px
        strip_top = (pos.y0 + pos.height) * fig_h_px
        for t in tax.texts:
            ext = t.get_window_extent(renderer)
            assert ext.y0 > strip_bottom + 2, f"{t.get_text()!r} touches the band's bottom edge"
            assert ext.y1 < strip_top - 2, f"{t.get_text()!r} touches the band's top edge"
    finally:
        plt.close(fig)


def test_band_fits_the_plot_beneath_it():
    """After fit_band_to_content the band's x-extent equals the plot's.

    The band starts full-width; the plot here is an axes with tick labels that
    stop well short of the figure's right edge. The strip re-seats inside the
    band with BAND_INSET on each side, and the logo stays flush with the strip.
    """
    saved = dict(PALETTE)
    PALETTE.update(band='#123456', title_ink='#FFFFFF')
    fig = plt.figure(figsize=(12, 8))
    try:
        ax = fig.add_axes([0.15, 0.10, 0.55, 0.65])
        ax.plot([0, 1], [0, 1])
        tax = title_axes(fig, height_frac=0.13, top_pad=0.02)
        draw_title_block(tax, "Title", ["Subtitle"], logo=_fake_logo(),
                         handle="Data: MLB", site="site.com")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        bb = ax.get_tightbbox(renderer)
        want_x0 = inv.transform((bb.x0, 0))[0]
        want_x1 = inv.transform((bb.x1, 0))[0]

        got = fit_band_to_content(fig)
        band = fig._dtw_band
        assert got == pytest.approx((want_x0, want_x1), abs=1e-6)
        assert band.get_x() == pytest.approx(want_x0, abs=1e-6)
        assert band.get_x() + band.get_width() == pytest.approx(want_x1, abs=1e-6)
        sp = tax.get_position()
        assert sp.x0 == pytest.approx(want_x0 + BAND_INSET, abs=1e-6)
        assert sp.x1 == pytest.approx(want_x1 - BAND_INSET, abs=1e-6)
        lax = [a for a in fig.axes if a.get_label() == "watermark_logo"][0]
        assert lax.get_position().x1 == pytest.approx(sp.x1, abs=1e-6)
    finally:
        plt.close(fig)
        PALETTE.clear()
        PALETTE.update(saved)


def test_band_widens_for_a_title_wider_than_the_plot():
    """A narrow plot under a long title: the band grows to cover the title."""
    saved = dict(PALETTE)
    PALETTE.update(band='#123456', title_ink='#FFFFFF')
    fig = plt.figure(figsize=(12, 8))
    try:
        ax = fig.add_axes([0.45, 0.10, 0.10, 0.65])
        ax.plot([0, 1], [0, 1])
        tax = title_axes(fig, height_frac=0.13, top_pad=0.02)
        draw_title_block(tax, "A Very Long Title That Is Wider Than The Plot", ["Subtitle"])
        fig.canvas.draw()
        x0, x1 = fit_band_to_content(fig)
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        for t in tax.texts:
            ext = t.get_window_extent(renderer)
            assert inv.transform((ext.x0, 0))[0] >= x0 - 1e-6
            assert inv.transform((ext.x1, 0))[0] <= x1 + 1e-6
    finally:
        plt.close(fig)
        PALETTE.clear()
        PALETTE.update(saved)
