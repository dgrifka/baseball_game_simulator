"""Optional palette hooks in ``Simulator/style.py`` and the stamp.

``PALETTE`` carries keys a downstream caller may set (``band``, ``title_ink``,
``stamp_ink``, ``stamp_disc``, ``mark_disc``, ``accent``, ``accent_2``). Every
one defaults to ``None``/``False``, and with the defaults a chart must render
exactly as it always has. These tests pin both halves: the default look is
unchanged, and each hook does what it says when set.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from Simulator import style  # noqa: E402
from Simulator import visualizations as viz  # noqa: E402
from Simulator.style import PALETTE, draw_title_block, finalize, place_mark, title_axes  # noqa: E402

HOOK_DEFAULTS = {
    'band': None, 'title_ink': None, 'stamp_ink': None,
    'stamp_disc': False, 'mark_disc': False, 'accent': None, 'accent_2': None,
}


def _clear_logo_caches():
    for fn in (viz._watermark_logo_native_for, viz._watermark_logo_rgba_for,
               viz._watermark_logo_for):
        fn.cache_clear()


@pytest.fixture
def palette():
    """Hand the test PALETTE, restore it (and the logo caches) afterwards."""
    saved = dict(PALETTE)
    yield PALETTE
    PALETTE.clear()
    PALETTE.update(saved)
    _clear_logo_caches()


def _strip(figsize=(4, 3)):
    fig = plt.figure(figsize=figsize)
    tax = title_axes(fig)
    draw_title_block(tax, "Title", ["Subtitle"], handle="Data: MLB", site="site.com")
    return fig, tax


def test_hook_keys_exist_with_behave_as_before_defaults():
    for key, default in HOOK_DEFAULTS.items():
        assert key in PALETTE, f"PALETTE is missing hook key {key!r}"
        assert PALETTE[key] is default, f"PALETTE[{key!r}] default is {PALETTE[key]!r}"


def test_defaults_render_cream_with_no_band(palette, tmp_path):
    fig, _ = _strip()
    assert fig.patches == [], "a default figure must not gain a band patch"
    out = tmp_path / "default.png"
    finalize(fig, str(out), dpi=50)
    px = Image.open(out).convert("RGB").getpixel((5, 5))
    assert px == (0xFC, 0xFA, 0xF6), f"default corner pixel is {px}, not #FCFAF6"


def test_band_adds_one_full_width_patch_and_title_uses_title_ink(palette):
    palette.update(band='#123456', title_ink='#FFFFFF', stamp_ink='#EEEEEE')
    fig, tax = _strip()
    try:
        assert len(fig.patches) == 1, f"expected one band patch, got {len(fig.patches)}"
        band = fig.patches[0]
        x0, y0 = band.get_xy()
        assert (x0, band.get_width()) == (0, 1), "band must span the full figure width"
        assert y0 + band.get_height() == pytest.approx(1.0), "band must reach the top edge"
        assert band.get_zorder() < 0, "band must sit behind everything"

        texts = {t.get_text(): t for t in tax.texts}
        assert matplotlib.colors.to_hex(texts["Title"].get_color()) == '#ffffff'
        assert matplotlib.colors.to_hex(texts["Subtitle"].get_color()) == '#ffffff'
        # Stamp ink is its own key, read when the strip is drawn.
        assert matplotlib.colors.to_hex(texts["Data: MLB"].get_color()) == '#eeeeee'
        assert matplotlib.colors.to_hex(texts["site.com"].get_color()) == '#eeeeee'
    finally:
        plt.close(fig)


def test_stamp_disc_puts_white_behind_the_logo(palette):
    plain = np.asarray(viz._watermark_logo_native())
    ys, xs = np.nonzero(plain[..., 3])
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()

    palette['stamp_disc'] = True
    disc = np.asarray(viz._watermark_logo_native())
    h, w = plain.shape[:2]
    oy, ox = (disc.shape[0] - h) // 2, (disc.shape[1] - w) // 2
    assert disc.shape[0] == disc.shape[1] > max(h, w), "disc image must be square and padded"

    # A 3-px ring just outside the logo's alpha bbox, sampled on the middle
    # third of each side (the bbox corners fall outside a round disc).
    cy, cx = oy + (top + bottom) // 2, ox + (left + right) // 2
    third_h, third_w = (bottom - top) // 6, (right - left) // 6
    ring = np.concatenate([
        disc[oy + top - 3:oy + top, cx - third_w:cx + third_w].reshape(-1, 4),
        disc[oy + bottom + 1:oy + bottom + 4, cx - third_w:cx + third_w].reshape(-1, 4),
        disc[cy - third_h:cy + third_h, ox + left - 3:ox + left].reshape(-1, 4),
        disc[cy - third_h:cy + third_h, ox + right + 1:ox + right + 4].reshape(-1, 4),
    ])
    assert len(ring) > 0
    assert (ring[:, :3] == 255).all(), "ring outside the logo is not white"
    assert (ring[:, 3] > 0).all(), "ring outside the logo is transparent"


def test_place_mark_disc_sits_one_below_the_image(palette):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    try:
        rgba = np.zeros((40, 40, 4), dtype=np.uint8)
        ab = place_mark(ax, rgba, (0.5, 0.5), 30, disc=True, zorder=5)
        assert len(ax.collections) == 1, "disc=True must add exactly one scatter"
        assert ax.collections[0].get_zorder() == 4
        assert ab.get_zorder() == 5
        # zoom = height_px / rgba height * 72 / dpi
        assert ab.offsetbox.get_zoom() == pytest.approx(30 / 40 * 72 / 100)

        ax2 = fig.add_subplot(2, 1, 2)
        place_mark(ax2, rgba, (0.5, 0.5), 30)  # mark_disc defaults False
        assert len(ax2.collections) == 0
    finally:
        plt.close(fig)
