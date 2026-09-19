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


# Strip-local vertical layout, shared by the rule and the watermark logo so
# the two can never drift into each other.
_RULE_Y = 0.55
_LOGO_BOTTOM = 0.62


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
    if band:
        # A figure patch, so bbox_inches='tight' keeps it; zorder puts it
        # behind every axes.
        fig.patches.append(Rectangle((0, 1 - height_frac - top_pad), 1,
                                     height_frac + top_pad,
                                     transform=fig.transFigure, facecolor=band,
                                     edgecolor='none', zorder=-10))
    width = max(0.50, 1.0 - 0.04 - right_reserve)
    ax = fig.add_axes([0.04, 1.0 - height_frac - top_pad, width, height_frac])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(False)
    ax.set_facecolor('none')
    return ax


def draw_title_block(ax, title, subtitle_lines=None, *,
                     title_size=20, subtitle_size=11,
                     rule=True, logo=None, handle=None, site=None, logo_pt=30):
    """Render a 2-row title block inside the strip from ``title_axes()``.

    Layout: bold title on row 1, optional thin divider rule, then one or
    more subtitle lines (passed as a list so each line can be its own
    height). Subtitle text is muted; rule is a thin grey line.

    ``logo`` (an RGBA array) and ``handle`` (a string) draw the watermark
    inside the strip instead of pasting it onto the saved PNG afterwards.
    The logo is right-aligned on the title row, the handle right-aligned on
    the first subtitle row, and the rule's band between them is left empty.
    ``site`` (a string) goes right-aligned one subtitle row under the handle.

    ``logo_pt`` is the logo's height in *points*, so it renders at the same
    physical size on every chart at a given dpi. Sizing it as a fraction of
    the image — what the PIL paste did — made the logo twice as tall on a
    tall chart as on a wide one.
    """
    if subtitle_lines is None:
        subtitle_lines = []
    elif isinstance(subtitle_lines, str):
        subtitle_lines = [subtitle_lines]

    title_ink = PALETTE.get('title_ink')
    title_color = title_ink or PALETTE['text']
    subtitle_color = title_ink or PALETTE['text_muted']
    stamp_color = PALETTE.get('stamp_ink') or PALETTE['text_muted']

    if logo is not None:
        fig = ax.figure
        fig_w, fig_h = fig.get_size_inches()
        pos = ax.get_position()
        # Height in figure fraction from a physical point size; width follows
        # from the asset's own aspect ratio, corrected for the figure's.
        h = logo_pt / 72 / fig_h
        w = h * (fig_h / fig_w) * (logo.shape[1] / logo.shape[0])
        y0 = pos.y0 + _LOGO_BOTTOM * pos.height
        lax = fig.add_axes([pos.x1 - w, y0, w, h], label='watermark_logo')
        lax.imshow(logo, interpolation='antialiased')
        lax.axis('off')
        lax.set_facecolor('none')
        # Below the strip so a long title overprints the logo, not vice versa.
        lax.set_zorder(ax.get_zorder() - 1)

    ax.text(0.0, 0.92, title,
            fontsize=title_size, fontweight='bold',
            color=title_color, ha='left', va='top',
            fontfamily=_HEADING_FONTS,
            transform=ax.transAxes)

    cursor_y = _RULE_Y
    if rule:
        ax.plot([0.0, 1.0], [cursor_y, cursor_y],
                color=title_ink or PALETTE['grid'],
                alpha=0.5 if title_ink else None, linewidth=0.8,
                transform=ax.transAxes, clip_on=False)
        cursor_y -= 0.10

    if handle is not None:
        # Same size and colour as a subtitle, on the subtitle's own row.
        ax.text(1.0, cursor_y, handle,
                fontsize=subtitle_size,
                color=stamp_color,
                ha='right', va='top',
                transform=ax.transAxes)
        if site is not None:
            ax.text(1.0, cursor_y - 0.30, site,
                    fontsize=subtitle_size,
                    color=stamp_color,
                    ha='right', va='top',
                    transform=ax.transAxes)

    for line in subtitle_lines:
        ax.text(0.0, cursor_y, line,
                fontsize=subtitle_size,
                color=subtitle_color,
                ha='left', va='top',
                transform=ax.transAxes)
        cursor_y -= 0.30


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
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches,
                facecolor=PALETTE['bg'], edgecolor='none')
    if apply_watermark_fn is not None:
        apply_watermark_fn(filepath)
    plt.close(fig)
