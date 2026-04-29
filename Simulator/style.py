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
                     rule=True):
    """Render a 2-row title block inside the strip from ``title_axes()``.

    Layout: bold title on row 1, optional thin divider rule, then one or
    more subtitle lines (passed as a list so each line can be its own
    height). Subtitle text is muted; rule is a thin grey line.
    """
    if subtitle_lines is None:
        subtitle_lines = []
    elif isinstance(subtitle_lines, str):
        subtitle_lines = [subtitle_lines]

    ax.text(0.0, 0.92, title,
            fontsize=title_size, fontweight='bold',
            color=PALETTE['text'], ha='left', va='top',
            fontfamily=_HEADING_FONTS,
            transform=ax.transAxes)

    cursor_y = 0.55
    if rule:
        ax.plot([0.0, 1.0], [cursor_y, cursor_y],
                color=PALETTE['grid'], linewidth=0.8,
                transform=ax.transAxes, clip_on=False)
        cursor_y -= 0.10

    for line in subtitle_lines:
        ax.text(0.0, cursor_y, line,
                fontsize=subtitle_size,
                color=PALETTE['text_muted'],
                ha='left', va='top',
                transform=ax.transAxes)
        cursor_y -= 0.30


def finalize(fig, filepath, *, dpi=200, apply_watermark_fn=None):
    """Standardized save + watermark + close.

    ``apply_watermark_fn`` is injected by the caller (avoids a circular
    import between this module and visualizations.py).
    """
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight',
                facecolor=PALETTE['bg'], edgecolor='none')
    if apply_watermark_fn is not None:
        apply_watermark_fn(filepath)
    plt.close(fig)
