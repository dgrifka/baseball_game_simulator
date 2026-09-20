"""Shared styling helpers for chart functions in this module.

Centralizes the cream / off-white "newsprint" look used by the four social
chart functions (spray, run distribution, estimated bases table, player
contributions). Provides:

- PALETTE: a single source of truth for non-team-color hex values.
- apply_base_style(): idempotent rcParams setup. Loads Inter / Oswald if
  available on the system; falls back silently to DejaVu Sans otherwise.
- get_team_color(): safe accessor with grey fallback (replaces the
  duplicated ``team_colors.get(name, ['#333','#666'])[0]`` call sites).
- stamp_header(): two-line title block at fig coords, used by all four
  charts so vertical spacing matches.
- finalize(): standardized savefig + watermark + close trio.
- lighten(): RGB blend toward white, used for walk-segment dilution.
"""

import os
from matplotlib import rcParams, font_manager
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle


PALETTE = {
    'bg':         '#FCFAF6',   # subtle warm cream — figure & axes facecolor
    'text':       '#1A1A1A',   # primary text
    'text_muted': '#6B6258',   # subtitle / footer / muted labels
    'grid':       '#D8D3C8',   # gridlines (dashed, alpha 0.6)
    'spine':      '#C7C0B4',   # remaining (bottom/left) spines
    # Outcome semantic palette — warmed slightly to read on cream
    'out':        '#9AA0A6',
    'single':     '#F4A340',
    'xbh':        '#E26A2C',
    'hr':         '#C03A2B',
    # Luck / good-bad
    'good':       '#2E7D32',
    'bad':        '#C03A2B',
    # Row stripes (table)
    'row_alt':    '#F4EFE6',
    # Optional hooks. A caller may set these before drawing; None / False
    # means "behave exactly as before", so the defaults render as above.
    'band':       None,    # title-strip fill (a full-width figure patch)
    'title_ink':  None,    # title + subtitle ink inside a filled band
    'stamp_ink':  None,    # watermark text ink
    'stamp_disc': False,   # paste the watermark logo on a white disc
    'mark_disc':  False,   # white disc behind marks drawn by place_mark()
    'accent':     None,    # non-team mark colour
    'accent_2':   None,    # secondary non-team mark colour
}


# Font preference order. Matplotlib walks the list and uses the first one
# present on the system. DejaVu Sans is bundled with matplotlib, so the
# final entry always resolves.
_BODY_FONTS = ['Inter', 'IBM Plex Sans', 'Helvetica Neue', 'DejaVu Sans']
_HEADING_FONTS = ['Oswald', 'Barlow Condensed', 'DejaVu Sans']


_BASE_STYLE_APPLIED = False


def apply_base_style():
    """Set rcParams for cream background + Inter body font.

    Safe to call repeatedly; rcParams overwrites idempotently. The first
    call also probes for Inter/Oswald via font_manager so the matched
    families are visible to ``rcParams['font.family']``.
    """
    global _BASE_STYLE_APPLIED

    if not _BASE_STYLE_APPLIED:
        # Force font_manager to scan installed fonts on first call. If the
        # user has Inter / Oswald installed at the OS level they will be
        # picked up; otherwise DejaVu is used.
        try:
            font_manager.fontManager.findfont(_BODY_FONTS[0], fallback_to_default=True)
        except Exception:
            pass
        _BASE_STYLE_APPLIED = True

    rcParams['figure.facecolor'] = PALETTE['bg']
    rcParams['axes.facecolor']   = PALETTE['bg']
    rcParams['savefig.facecolor'] = PALETTE['bg']

    rcParams['font.family'] = _BODY_FONTS
    rcParams['font.size']   = 11
    rcParams['text.color']  = PALETTE['text']

    rcParams['axes.edgecolor']  = PALETTE['spine']
    rcParams['axes.labelcolor'] = PALETTE['text']
    rcParams['axes.titlecolor'] = PALETTE['text']
    rcParams['axes.titleweight'] = 'bold'
    rcParams['axes.spines.top']    = False
    rcParams['axes.spines.right']  = False
    rcParams['axes.linewidth'] = 0.8

    rcParams['xtick.color'] = PALETTE['text_muted']
    rcParams['ytick.color'] = PALETTE['text_muted']
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10

    rcParams['grid.color']     = PALETTE['grid']
    rcParams['grid.linestyle'] = '--'
    rcParams['grid.linewidth'] = 0.8
    rcParams['grid.alpha']     = 0.6

    rcParams['legend.frameon']    = False
    rcParams['legend.fontsize']   = 10
    rcParams['legend.labelcolor'] = PALETTE['text']


def heading_font():
    """Return the heading font family list for use as ``fontfamily=...``."""
    return _HEADING_FONTS


def get_team_color(team_colors_map, team_name, idx=0, fallback=('#333333', '#666666')):
    """Safe accessor for ``team_colors`` with grey fallback.

    Replaces the four duplicated call sites that did
    ``team_colors.get(name, ['#333333', '#666666'])[idx]``.
    """
    return team_colors_map.get(team_name, list(fallback))[idx]


def lighten(color, amount=0.5):
    """Blend ``color`` toward white by ``amount`` (0..1).

    Used to derive the "walk segment" tint from a team's primary color
    (player contribution chart). Replaces the inline
    ``r + (1 - r) * 0.5`` arithmetic.
    """
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount,
            g + (1 - g) * amount,
            b + (1 - b) * amount)


def darken(color, amount=0.5):
    """Blend ``color`` toward black by ``amount`` (0..1).

    The counterpart to ``lighten``. Used where a color has to be pushed away
    from another one without also pushing it toward the cream background —
    lightening a pale team primary makes it vanish into the page.
    """
    r, g, b = to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


def stamp_header(fig, title, subtitle=None, *, x=0.5, ha='center',
                 y_title=0.97, y_subtitle=0.935,
                 title_size=16, subtitle_size=11):
    """Place a two-line title block in figure coordinates.

    Centralizes the slightly-different inline title stacks used across
    the four chart functions so vertical spacing matches.
    """
    fig.text(x, y_title, title,
             fontsize=title_size, fontweight='bold',
             color=PALETTE['text'], ha=ha, va='top',
             fontfamily=_HEADING_FONTS)
    if subtitle:
        fig.text(x, y_subtitle, subtitle,
                 fontsize=subtitle_size, color=PALETTE['text_muted'],
                 ha=ha, va='top', linespacing=1.5)


# Strip layout, in multiples of the font sizes (so it is the same physical
# spacing on every chart, whatever the strip's height in figure fraction).
# The block — title row, rule, subtitle rows — is centred vertically in the
# strip; a strip that is too short for its block prints a warning.
_TITLE_ROW = 1.20        # x title_size: the title row's height
_RULE_GAP_ABOVE = 0.45   # x subtitle_size: title row -> rule
_RULE_GAP_BELOW = 0.55   # x subtitle_size: rule -> first subtitle row
_SUB_ROW = 1.20          # x subtitle_size: a subtitle row's height
_SUB_GAP = 0.30          # x subtitle_size: between subtitle rows
_LOGO_RULE_GAP = 0.35    # x subtitle_size: rule -> logo bottom
_LOGO_PT_PER_SUBTITLE = 2.6   # default logo height, x subtitle_size
_MIN_PAD = 0.25          # x subtitle_size: least band above/below the block

# Horizontal inset of the strip (title, rule, stamp) from the band's edges
# once fit_band_to_content() has matched the band to the plot, as a fraction
# of the figure width.
BAND_INSET = 0.025


def title_axes(fig, *, height_frac=0.14, top_pad=0.015, right_reserve=0.12):
    """Reserve a dedicated, axis-less strip across the top of the figure
    for a title block.

    Returns a Matplotlib Axes positioned at the top of ``fig`` with no
    spines, ticks, or background fill. Use ``draw_title_block`` to fill
    it. Lets titles breathe in their own coordinate space without
    colliding with the plot region — far cleaner than stamp_header for
    histograms / spray fields where chart elements approach the top.

    ``right_reserve`` controls how much of the figure's right edge is
    left blank for the watermark (logo + handle text). Set to 0 for
    charts where the watermark sits elsewhere.
    """
    band = PALETTE.get('band')
    band_patch = None
    if band:
        # A figure patch, so bbox_inches='tight' keeps it; zorder puts it
        # behind every axes. Full width until fit_band_to_content() (called by
        # finalize) matches it to the plot beneath.
        band_patch = Rectangle((0, 1 - height_frac - top_pad), 1,
                               height_frac + top_pad,
                               transform=fig.transFigure, facecolor=band,
                               edgecolor='none', zorder=-10)
        fig.patches.append(band_patch)
    width = max(0.50, 1.0 - 0.04 - right_reserve)
    ax = fig.add_axes([0.04, 1.0 - height_frac - top_pad, width, height_frac],
                      label='title_strip')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(False)
    ax.set_facecolor('none')
    # Remembered so fit_band_to_content() can find them at save time.
    fig._dtw_band = band_patch
    fig._dtw_strip = ax
    return ax


def _strip_layout(ax, title_size, subtitle_size, n_rows, rule):
    """Vertical positions (strip fractions) for a centred title block.

    Returns ``(title_y, rule_y, line_ys, pitch)``: the title's top, the rule,
    the top of each subtitle row, and the row pitch — all as fractions of the
    strip's height, computed from the strip's physical height in points so the
    spacing is the same on every chart.
    """
    fig = ax.figure
    strip_pt = ax.get_position().height * fig.get_size_inches()[1] * 72.0
    rule_gap = (_RULE_GAP_ABOVE + _RULE_GAP_BELOW) * subtitle_size
    block = (_TITLE_ROW * title_size + rule_gap
             + n_rows * _SUB_ROW * subtitle_size
             + max(0, n_rows - 1) * _SUB_GAP * subtitle_size)
    pad = (strip_pt - block) / 2.0
    if pad < _MIN_PAD * subtitle_size:
        print(f"Warning: title strip is {strip_pt:.0f}pt tall but its text block "
              f"needs {block:.0f}pt; raise height_frac")
        pad = _MIN_PAD * subtitle_size

    def frac(pt):
        return pt / strip_pt

    title_y = 1.0 - frac(pad)
    rule_y = title_y - frac(_TITLE_ROW * title_size + _RULE_GAP_ABOVE * subtitle_size)
    first = rule_y - frac(_RULE_GAP_BELOW * subtitle_size)
    pitch = frac((_SUB_ROW + _SUB_GAP) * subtitle_size)
    line_ys = [first - i * pitch for i in range(n_rows)]
    return title_y, rule_y, line_ys, pitch


def draw_title_block(ax, title, subtitle_lines=None, *,
                     title_size=20, subtitle_size=11,
                     rule=True, logo=None, handle=None, site=None, logo_pt=None):
    """Render a 2-row title block inside the strip from ``title_axes()``.

    Layout: bold title on row 1, optional thin divider rule, then one or
    more subtitle lines (passed as a list so each line can be its own
    height). Subtitle text is muted; rule is a thin grey line. The rows are
    spaced in points (see ``_strip_layout``) and the whole block is centred
    vertically in the strip, so a second subtitle line never lands on the
    band's bottom edge.

    ``logo`` (an RGBA array) and ``handle`` (a string) draw the watermark
    inside the strip instead of pasting it onto the saved PNG afterwards.
    The logo is right-aligned on the title row, the handle right-aligned on
    the first subtitle row, and the rule's band between them is left empty.
    ``site`` (a string) goes right-aligned one subtitle row under the handle.

    ``logo_pt`` is the logo's height in *points*, so it renders at the same
    physical size on every chart at a given dpi; the default is 2.6x the
    subtitle size, so logo and credit text keep one proportion everywhere.
    Sizing it as a fraction of the image — what the PIL paste did — made the
    logo twice as tall on a tall chart as on a wide one.
    """
    if subtitle_lines is None:
        subtitle_lines = []
    elif isinstance(subtitle_lines, str):
        subtitle_lines = [subtitle_lines]
    if logo_pt is None:
        logo_pt = _LOGO_PT_PER_SUBTITLE * subtitle_size

    title_ink = PALETTE.get('title_ink')
    title_color = title_ink or PALETTE['text']
    subtitle_color = title_ink or PALETTE['text_muted']
    stamp_color = PALETTE.get('stamp_ink') or PALETTE['text_muted']

    stamp_rows = (1 if handle is not None else 0) + (1 if site is not None else 0)
    n_rows = max(len(subtitle_lines), stamp_rows)
    title_y, rule_y, line_ys, pitch = _strip_layout(
        ax, title_size, subtitle_size, n_rows, rule)
    ax._dtw_layout = {'title_y': title_y, 'rule_y': rule_y,
                      'line_ys': list(line_ys), 'pitch': pitch}

    if logo is not None:
        fig = ax.figure
        fig_w, fig_h = fig.get_size_inches()
        pos = ax.get_position()
        # Height in figure fraction from a physical point size; width follows
        # from the asset's own aspect ratio, corrected for the figure's.
        h = logo_pt / 72 / fig_h
        w = h * (fig_h / fig_w) * (logo.shape[1] / logo.shape[0])
        strip_pt = pos.height * fig_h * 72.0
        y0 = pos.y0 + (rule_y + _LOGO_RULE_GAP * subtitle_size / strip_pt) * pos.height
        lax = fig.add_axes([pos.x1 - w, y0, w, h], label='watermark_logo')
        lax.imshow(logo, interpolation='antialiased')
        lax.axis('off')
        lax.set_facecolor('none')
        # Below the strip so a long title overprints the logo, not vice versa.
        lax.set_zorder(ax.get_zorder() - 1)

    ax.text(0.0, title_y, title,
            fontsize=title_size, fontweight='bold',
            color=title_color, ha='left', va='top',
            fontfamily=_HEADING_FONTS,
            transform=ax.transAxes)

    if rule:
        ax.plot([0.0, 1.0], [rule_y, rule_y],
                color=title_ink or PALETTE['grid'],
                alpha=0.5 if title_ink else None, linewidth=0.8,
                transform=ax.transAxes, clip_on=False)

    if handle is not None:
        # Same size and colour as a subtitle, on the subtitle's own row.
        ax.text(1.0, line_ys[0], handle,
                fontsize=subtitle_size,
                color=stamp_color,
                ha='right', va='top',
                transform=ax.transAxes)
        if site is not None:
            ax.text(1.0, line_ys[1], site,
                    fontsize=subtitle_size,
                    color=stamp_color,
                    ha='right', va='top',
                    transform=ax.transAxes)

    for line, y in zip(subtitle_lines, line_ys):
        ax.text(0.0, y, line,
                fontsize=subtitle_size,
                color=subtitle_color,
                ha='left', va='top',
                transform=ax.transAxes)


def fit_band_to_content(fig, *, inset=BAND_INSET):
    """Match the title band's width to the plot beneath it.

    The band starts as a full-width figure patch, but ``bbox_inches='tight'``
    crops to the artists, which can reach past the figure on one side (tick
    labels, headshots) and stop short of it on the other — so the saved band
    overhung the plot on the right and missed it on the left. This measures
    the union of everything except the strip itself, sets the band's
    horizontal extent to it, and re-seats the strip (title, rule, stamp)
    inside with an ``inset`` (figure-width fraction) on each side. Widens the
    band if the strip's own text is wider than the plot.

    Called by ``finalize``; a no-op without a band or a strip. Never raises.
    """
    from matplotlib.transforms import Bbox

    band = getattr(fig, '_dtw_band', None)
    strip = getattr(fig, '_dtw_strip', None)
    if band is None or strip is None:
        return None
    try:
        renderer = fig.canvas.get_renderer()
        logo_axes = [a for a in fig.axes if a.get_label() == 'watermark_logo']
        skip = set(logo_axes) | {strip}
        boxes = []
        for a in fig.axes:
            if a in skip or not a.get_visible():
                continue
            bb = a.get_tightbbox(renderer)
            if bb is not None and bb.width > 0:
                boxes.append(bb)
        for t in fig.texts:
            if t.get_visible() and t.get_text():
                boxes.append(t.get_window_extent(renderer))
        for p in fig.patches:
            if p is not band and p.get_visible():
                boxes.append(p.get_window_extent(renderer))
        for lg in fig.legends:
            boxes.append(lg.get_window_extent(renderer))
        for im in fig.images:
            boxes.append(im.get_window_extent(renderer))
        if not boxes:
            return None
        inv = fig.transFigure.inverted()
        u = Bbox.union(boxes)
        x0 = inv.transform((u.x0, 0))[0]
        x1 = inv.transform((u.x1, 0))[0]

        def seat(x0, x1):
            band.set_x(x0)
            band.set_width(x1 - x0)
            sp = strip.get_position()
            strip.set_position([x0 + inset, sp.y0,
                                max(x1 - x0 - 2 * inset, 0.2), sp.height])
            for lax in logo_axes:
                lp = lax.get_position()
                lax.set_position([x1 - inset - lp.width, lp.y0, lp.width, lp.height])

        seat(x0, x1)
        # The strip's text must stay inside the band: widen, never shrink.
        tb = [t.get_window_extent(renderer) for t in strip.texts if t.get_text()]
        if tb:
            tu = Bbox.union(tb)
            tx0 = inv.transform((tu.x0, 0))[0] - inset
            tx1 = inv.transform((tu.x1, 0))[0] + inset
            if tx0 < x0 or tx1 > x1:
                seat(min(x0, tx0), max(x1, tx1))
        return band.get_x(), band.get_x() + band.get_width()
    except Exception as e:
        print(f"Warning: could not fit the title band to the plot: {e}")
        return None


def place_mark(ax, rgba, xy, height_px, *, disc=None, zorder=5, xycoords='data'):
    """Place an RGBA mark (team logo, headshot) ``height_px`` pixels tall.

    ``OffsetImage`` zoom is in points-per-pixel, so a bare ``zoom`` renders a
    different size at every dpi; dividing by ``dpi / 72`` makes ``height_px``
    mean saved pixels. With ``disc`` (default ``PALETTE['mark_disc']``) a
    white circle 1.35x the mark's height is drawn one zorder below it.
    """
    dpi = ax.figure.dpi
    if disc if disc is not None else PALETTE.get('mark_disc'):
        diameter_pt = 1.35 * height_px * 72.0 / dpi
        ax.scatter([xy[0]], [xy[1]], s=diameter_pt ** 2, color='white',
                   edgecolors='none', zorder=zorder - 1, clip_on=False,
                   transform=ax.transAxes if xycoords == 'axes fraction' else ax.transData)
    zoom = height_px / rgba.shape[0] * 72.0 / dpi
    ab = AnnotationBbox(OffsetImage(rgba, zoom=zoom), xy, frameon=False, pad=0,
                        xycoords=xycoords, zorder=zorder)
    ax.add_artist(ab)
    return ab


def finalize(fig, filepath, *, dpi=200, pad_inches=0.1, apply_watermark_fn=None):
    """Standardized save + watermark + close.

    ``apply_watermark_fn`` is injected by the caller (avoids a circular
    import between this module and visualizations.py).

    ``pad_inches`` defaults to 0.1, which is matplotlib's own default for
    ``bbox_inches='tight'`` — so existing callers are byte-unchanged.
    """
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    fit_band_to_content(fig)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches,
                facecolor=PALETTE['bg'], edgecolor='none')
    if apply_watermark_fn is not None:
        apply_watermark_fn(filepath)
    plt.close(fig)
