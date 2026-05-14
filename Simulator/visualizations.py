import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors
import pandas as pd
import numpy as np
from scipy import stats
import joblib

from Simulator.constants import (
    team_colors,
    STADIUM_DIMENSIONS, DEFAULT_STADIUM_DIMENSIONS, FEET_TO_PLOT,
    TEAM_LOGO_MAP, TEAM_DISPLAY_MAP
)
from Simulator.game_simulator import prepare_batted_ball_features
from Simulator.style import (
    PALETTE, apply_base_style, get_team_color, lighten,
    stamp_header, title_axes, draw_title_block, finalize, heading_font,
)
from Model.feature_engineering import HOME_PLATE_X, HOME_PLATE_Y, calculate_spray_angle

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import requests
from io import BytesIO
from PIL import Image, ImageEnhance
import os

# Resolve paths relative to the repo root (parent of Simulator/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_REPO_ROOT, 'Model', 'batted_ball_model.pkl')
_CONTOUR_PATH = os.path.join(_REPO_ROOT, 'Data', 'contour_data.csv')
_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'mlb_simulator_logo.png')

# Lazy-loaded caches
_contour_cache = None
_model_cache = None


def _get_contour_data():
    """Load and cache contour data for la_ev_graph."""
    global _contour_cache
    if _contour_cache is None:
        _contour_cache = pd.read_csv(_CONTOUR_PATH).dropna()
    return _contour_cache


def _get_model():
    """Load and cache the batted ball model."""
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(_MODEL_PATH)
    return _model_cache


# ---------------------------------------------------------------------------
# Player headshot helpers
# ---------------------------------------------------------------------------
_HEADSHOT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".headshot_cache"
)
_headshot_mem_cache = {}


def _load_headshot(player_id, size=80):
    """Download, circular-crop, and cache a player headshot. Returns np.ndarray or None."""
    if player_id in _headshot_mem_cache:
        return _headshot_mem_cache[player_id]

    if player_id is None:
        _headshot_mem_cache[player_id] = None
        return None

    try:
        pid = int(player_id)
    except (ValueError, TypeError):
        _headshot_mem_cache[player_id] = None
        return None

    os.makedirs(_HEADSHOT_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_HEADSHOT_CACHE_DIR, f"{pid}.png")

    try:
        if os.path.exists(cache_path):
            img = Image.open(cache_path).convert("RGBA")
        else:
            url = (
                f"https://img.mlbstatic.com/mlb-photos/image/upload/"
                f"d_people:generic:headshot:silo:current.png/"
                f"w_213,q_auto:best/v1/people/{pid}/headshot/silo/current.png"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGBA")
            img.save(cache_path)

        img = img.resize((size, size), Image.Resampling.LANCZOS)
        mask = Image.new("L", (size, size), 0)
        from PIL import ImageDraw as _ImageDraw
        _ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
        r, g, b, a = img.split()
        img = Image.merge("RGBA", (r, g, b, mask))
        data = np.array(img)
        _headshot_mem_cache[player_id] = data
        return data
    except Exception:
        _headshot_mem_cache[player_id] = None
        return None


def _apply_watermark(filepath, position='top-right', y_pct=None):
    """Add logo + watermark text to a saved image file using PIL.

    Positions the logo and text at a fixed percentage offset from the
    specified corner of the saved image.  This is immune to matplotlib
    figure-size, DPI, and bbox_inches='tight' differences.

    Parameters
    ----------
    filepath : str
        Path to the saved image file (modified in place).
    position : str
        'top-right' or 'top-left'.
    y_pct : float, optional
        Vertical position as fraction of image height (0.0 = top, 1.0 = bottom).
        The logo top edge is placed at this position. Default ~1.2% from top.
    """
    from PIL import ImageDraw, ImageFont
    import matplotlib.font_manager as fm

    try:
        img = Image.open(filepath).convert('RGBA')
        w, h = img.size

        logo = Image.open(_LOGO_PATH).convert('RGBA')
        logo_data = np.array(logo)
        r, g, b = logo_data[:, :, 0], logo_data[:, :, 1], logo_data[:, :, 2]
        white_mask = (r > 240) & (g > 240) & (b > 240)
        logo_data[white_mask, 3] = 0
        logo_data[:, :, 3] = (logo_data[:, :, 3].astype(float) * 0.85).astype(np.uint8)
        logo = Image.fromarray(logo_data)

        # Scale logo relative to the smaller dimension (avoids overflow on narrow images)
        ref = min(w, h)
        logo_h = max(20, int(ref * 0.028))
        logo_w = int(logo_h * logo.width / logo.height)
        logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

        # Prepare text — sized so it's slightly wider than the logo
        text = 'Data: MLB  |  @mlb_simulator'
        font_size = max(10, int(ref * 0.0095))
        try:
            font_path = fm.findfont(fm.FontProperties(family='DejaVu Sans'))
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        tmp_draw = ImageDraw.Draw(img)
        text_bbox = tmp_draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Position
        margin = int(w * 0.02)
        gap = int(ref * 0.003)

        block_w = max(logo_w, text_w)

        if position == 'top-left':
            center_x = margin + block_w // 2
        else:
            center_x = w - margin - block_w // 2

        logo_y = int(h * y_pct) if y_pct is not None else int(h * 0.012)
        logo_x = center_x - logo_w // 2
        text_x = center_x - text_w // 2
        text_y = logo_y + logo_h + gap

        img.paste(logo, (logo_x, logo_y), logo)
        draw = ImageDraw.Draw(img)
        draw.text((text_x, text_y), text, fill=(140, 140, 140), font=font)

        img.convert('RGB').save(filepath)
    except Exception as e:
        print(f"Warning: could not apply watermark to {filepath}: {e}")


def get_team_logo(team_name, mlb_team_logos, logo_cache={}):
    """
    Retrieves team logo URL with caching.
    
    Args:
        team_name (str): MLB team name
        mlb_team_logos (list): List of team logo URL dictionaries
        logo_cache (dict): Cache for logo URLs
        
    Returns:
        str: Logo URL or None if not found
    """
    if team_name in logo_cache:
        return logo_cache[team_name]
        
    logo_url = next((team['logo_url'] for team in mlb_team_logos 
                    if team['team'] == team_name), None)
    if logo_url:
        logo_cache[team_name] = logo_url
    else:
        print(f"Logo not found for {team_name}")
    return logo_url

def getImage(path, zoom=0.5, size=(50, 50), alpha=0.6, image_cache={}):
    """
    Processes team logo image with caching of raw image data.
    
    Args:
        path (str): Image URL
        zoom (float): Image zoom level
        size (tuple): Target image size
        alpha (float): Transparency level
        image_cache (dict): Cache for processed image data
        
    Returns:
        OffsetImage: Fresh OffsetImage instance for each call
    """
    try:
        # Cache the processed image data, not the OffsetImage
        if path not in image_cache:
            img = Image.open(BytesIO(requests.get(path).content))
            img = img.resize(size, Image.LANCZOS).convert("RGBA")
            
            # Enhance and crop
            img = ImageEnhance.Sharpness(img).enhance(2.5)
            data = np.array(img)
            mask = data[:, :, 3] > 0
            ymin, ymax = np.where(np.any(mask, axis=1))[0][[0, -1]]
            xmin, xmax = np.where(np.any(mask, axis=0))[0][[0, -1]]
            
            # Create transparent background
            new_img = Image.new('RGBA', size, (255, 255, 255, 0))
            cropped = img.crop((xmin, ymin, xmax+1, ymax+1))
            paste_pos = tuple((s - c) // 2 for s, c in zip(size, cropped.size))
            new_img.paste(cropped, paste_pos, cropped)
            
            # Apply alpha and store the processed data
            data = np.array(new_img)
            data[:, :, 3] = (data[:, :, 3] * alpha).astype(np.uint8)
            image_cache[path] = data
        
        # Create a fresh OffsetImage instance for each call
        return OffsetImage(Image.fromarray(image_cache[path].copy()), zoom=zoom)
    
    except Exception as e:
        print(f"Error loading image from {path}: {str(e)}")
        return None
        
def la_ev_graph(home_outcomes, away_outcomes, away_estimated_total_bases, home_estimated_total_bases, 
                home_team, away_team, home_score, away_score, home_win_percentage, away_win_percentage, 
                tie_percentage, mlb_team_logos, formatted_date, images_dir="images"):
    """
    Creates enhanced launch angle vs exit velocity visualization with team performance overlay.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.close('all')
    
    percentages = {
        'away': f"{away_win_percentage:.0f}",
        'home': f"{home_win_percentage:.0f}",
        'tie': f"{tie_percentage:.0f}"
    }
    
    # Extract data
    outcomes = {
        'home': {'ev': [], 'la': [], 'walks': home_outcomes.count('walk'), 
                'stolen_base': home_outcomes.count('stolen_base')},
        'away': {'ev': [], 'la': [], 'walks': away_outcomes.count('walk'),
                'stolen_base': away_outcomes.count('stolen_base')}
    }
    
    # At the beginning of your function, filter out data points below your x-axis limit:
    for team, team_outcomes in [('home', home_outcomes), ('away', away_outcomes)]:
        temp_ev = [o[0] for o in team_outcomes if isinstance(o, list)]
        temp_la = [o[1] for o in team_outcomes if isinstance(o, list)]
        # Filter points within axis limits
        filtered_indices = [i for i, ev in enumerate(temp_ev) if 55 <= ev <= 120]
        outcomes[team]['ev'] = [temp_ev[i] for i in filtered_indices]
        outcomes[team]['la'] = [temp_la[i] for i in filtered_indices]
        
    # Create figure with higher DPI for sharper rendering. Add constrained_layout instead of tight bbox_inches
    fig = plt.figure(figsize=(12, 8), dpi=150, constrained_layout=True)
    
    # Load and process contour data with improved interpolation
    contour_data = _get_contour_data()
    x, y, z = contour_data['x'].values, contour_data['y'].values, contour_data['z'].values
    
    xi = np.linspace(x.min(), x.max(), 200)  # Increased resolution
    yi = np.linspace(y.min(), y.max(), 200)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='cubic', fill_value=0)
    
    # Grayscale colormap
    colors_list = ["white", "#E6E6E6", "#CCCCCC", "#B3B3B3", "#999999", "#808080", "#666666", "#4A4A4A"]
    levels = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    cmap = colors.LinearSegmentedColormap.from_list("custom", colors_list, N=256)
    
    # Plot contours with improved aesthetics
    Z = np.clip(Z, 0.5, 4)
    contour = plt.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.8)
    plt.contour(X, Y, Z, levels=levels, colors='gray', linewidths=0.3, alpha=0.3)
    
    # Enhanced colorbar
    cbar = plt.colorbar(contour, label='Average Total Bases')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Average Total Bases', size=14, labelpad=15)
    
    # Plot team data with improved visibility
    for team, data in outcomes.items():
        team_name = home_team if team == 'home' else away_team
        logo_url = get_team_logo(team_name, mlb_team_logos)
        
        if logo_url:
            for x, y in zip(data['ev'], data['la']):
                img = getImage(logo_url, zoom=0.65 if team == 'home' else 0.635,
                             alpha=0.8 if team == 'home' else 0.785)
                if img:
                    ab = AnnotationBbox(img, (x, y), frameon=False)
                    plt.gca().add_artist(ab)
    
    # Enhanced formatting
    plt.axhline(y=0, color='black', alpha=0.6, linewidth=1.0, linestyle='--')
    
    # Walks/HBP/SB section with improved typography
    plt.text(0.05, 0.95, 'Walks/HBP/SB', transform=plt.gca().transAxes, 
             fontsize=16, fontweight='bold', verticalalignment='top')
    plt.text(0.05, 0.90, f'{away_team}: {outcomes["away"]["walks"] + outcomes["away"]["stolen_base"]}', 
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    plt.text(0.05, 0.85, f'{home_team}: {outcomes["home"]["walks"] + outcomes["home"]["stolen_base"]}', 
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    
    # Watermark (applied after save via _apply_watermark below)
    
    # Improved labels and title
    plt.xlabel('Exit Velocity (mph)', fontsize=16, labelpad=12)
    plt.ylabel('Launch Angle', fontsize=16, labelpad=12)
    
    title = f'Batted Ball Exit Velo / Launch Angle by Team\n' \
            f'Actual Score:     {away_team} {str(away_score)} - {home_team} {str(home_score)}  ({formatted_date})\n' \
            f'Deserve-to-Win: {away_team} {percentages["away"]}%, {home_team} ' \
            f'{percentages["home"]}%, Tie {percentages["tie"]}%'
    
    plt.title(title, fontsize=16, loc='left', pad=15, fontweight='bold')
    
    # Enhanced tick formatting
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%d°'))
    
    # Clean up spines
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)
    
    # Set axis limits with padding
    plt.xlim(55, 120)
    plt.ylim(-70, 70)
    
    # Save with high quality
    os.makedirs(images_dir, exist_ok=True)
    filename = f'{away_team}_{home_team}_{str(away_score)}-{str(home_score)}--{percentages["away"]}-{percentages["home"]}_bb.png'
    filepath = os.path.join(images_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=200)
    plt.close(fig)
    _apply_watermark(filepath)


def run_dist(num_simulations, home_runs_scored, away_runs_scored, home_team, away_team,
             home_score, away_score, home_win_percentage, away_win_percentage, tie_percentage,
             formatted_date, images_dir="images"):
    """Histogram of simulated run distributions for both teams.

    Solid filled bars at moderate alpha so individual run-totals stay
    legible (no hatches). If the two teams' primary colors are too
    similar, the away team falls back to its secondary color so the
    histograms remain distinguishable. Score badges (dashed vline +
    rounded label) call out the actual outcome, and a bottom-right
    inline note states the leader's win share.
    """
    apply_base_style()

    percentages = {
        'away': f"{away_win_percentage:.0f}",
        'home': f"{home_win_percentage:.0f}",
        'tie':  f"{tie_percentage:.0f}",
    }

    home_runs_scored = np.asarray(home_runs_scored)
    away_runs_scored = np.asarray(away_runs_scored)
    max_runs = int(max(home_runs_scored.max(), away_runs_scored.max()))
    bins = np.arange(max_runs + 2) - 0.5  # bars centered on integer runs

    home_mode = int(stats.mode(home_runs_scored, keepdims=True).mode.flatten()[0])
    away_mode = int(stats.mode(away_runs_scored, keepdims=True).mode.flatten()[0])

    home_color = get_team_color(team_colors, home_team, idx=0)
    away_color = get_team_color(team_colors, away_team, idx=0)

    # If the two team primaries are too close, lighten the away team's
    # primary to keep the bars distinguishable. (Some teams' secondary
    # entries in team_colors aren't valid hex strings, so we don't rely
    # on them here.)
    def _color_distance(c1, c2):
        r1, g1, b1 = to_rgb(c1)
        r2, g2, b2 = to_rgb(c2)
        return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5

    if _color_distance(home_color, away_color) < 0.20:
        r, g, b = lighten(away_color, 0.45)
        away_color = colors.to_hex((r, g, b))

    # Reserve space at the top for a dedicated title strip.
    fig = plt.figure(figsize=(12, 8.5), dpi=150)
    fig.patch.set_facecolor(PALETTE['bg'])
    ax = fig.add_axes([0.08, 0.10, 0.88, 0.70])
    ax.set_facecolor(PALETTE['bg'])

    # Histograms — solid fills, thin matching edges (no hatches).
    home_counts, _, home_patches = ax.hist(
        home_runs_scored, bins=bins, color=home_color, alpha=0.78,
        edgecolor=home_color, linewidth=0.8, label=home_team, zorder=3,
    )
    away_counts, _, away_patches = ax.hist(
        away_runs_scored, bins=bins, color=away_color, alpha=0.55,
        edgecolor=away_color, linewidth=0.8, label=away_team, zorder=2,
    )

    # Actual-score markers + rounded-bbox labels anchored above bar tops.
    # Bumped headroom so badges sit clearly above the tallest bar.
    # Close-score stagger spans more vertically so the two boxes don't
    # graze each other or the bar tops.
    y_top = max(home_counts.max(), away_counts.max()) * 1.45
    scores_close = abs(home_score - away_score) <= 1
    if scores_close:
        # Lower-scoring team's badge sits higher to clear the overlap.
        away_badge_y = y_top * 0.97 if away_score > home_score else y_top * 0.78
        home_badge_y = y_top * 0.97 if home_score > away_score else y_top * 0.78
        if away_score == home_score:
            away_badge_y, home_badge_y = y_top * 0.97, y_top * 0.78
    else:
        away_badge_y = home_badge_y = y_top * 0.97

    for score, color, team, badge_y in [
        (away_score, away_color, away_team, away_badge_y),
        (home_score, home_color, home_team, home_badge_y),
    ]:
        ax.axvline(x=score, color=color, linestyle='--',
                   linewidth=2.0, alpha=0.85, zorder=4)
        ax.text(score, badge_y,
                f'{team} {score}\n(Actual)',
                ha='center', va='top', fontsize=9, fontweight='bold',
                color=PALETTE['text'], zorder=6,
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor=PALETTE['bg'], edgecolor=color,
                          linewidth=1.4))

    # Win-prob inline annotation — concise phrasing.
    leader = home_team if home_win_percentage >= away_win_percentage else away_team
    leader_pct = max(home_win_percentage, away_win_percentage)
    leader_color = home_color if leader == home_team else away_color
    ax.text(0.99, 0.04,
            f'{leader} win {leader_pct:.0f}% of simulations',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, fontstyle='italic', color=leader_color,
            fontweight='bold')

    # Axis formatting
    ax.set_xlim(-0.5, max_runs + 1.5)
    ax.set_ylim(0, y_top * 1.05)
    ax.set_xticks(range(max_runs + 1))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel('Runs Scored', fontsize=12, labelpad=8, color=PALETTE['text'])
    ax.set_ylabel('Frequency', fontsize=12, labelpad=8, color=PALETTE['text'])
    ax.grid(True, axis='y', linestyle='--', alpha=0.45, color=PALETTE['grid'])
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']:
        ax.spines[s].set_color(PALETTE['spine'])

    ax.legend(loc='upper right', frameon=False, fontsize=11)

    # Dedicated title strip at the top of the figure (separate axes
    # from the plot, so it can't collide with bars / legend / score
    # badges). Title big and bold; subtitle in two muted lines below a
    # thin divider rule.
    tax = title_axes(fig, height_frac=0.13, top_pad=0.02)
    subtitle_lines = [
        f'Actual: {away_team} {away_score} - {home_team} {home_score}   '
        f'({formatted_date})    DTW: {away_team} {percentages["away"]}% • '
        f'{home_team} {percentages["home"]}% • Tie {percentages["tie"]}%',
        f'Most Likely: {away_team} {away_mode} - {home_team} {home_mode}',
    ]
    draw_title_block(tax,
                     f'Distribution of Runs Scored  —  {num_simulations:,} Simulations',
                     subtitle_lines,
                     title_size=20, subtitle_size=11)

    os.makedirs(images_dir, exist_ok=True)
    filename = (f'{away_team}_{home_team}_{away_score}-{home_score}'
                f'--{percentages["away"]}-{percentages["home"]}_rd.png')
    filepath = os.path.join(images_dir, filename)
    finalize(fig, filepath, dpi=200, apply_watermark_fn=_apply_watermark)

def prepare_table_data(df):
    """Prepare and format data for table visualization."""
    df = df.copy()
    
    # Add rank column
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    # Calculate spray direction if coordinates are available
    if 'coord_x' in df.columns and 'coord_y' in df.columns and 'bat_side' in df.columns:
        df['Spray'] = df.apply(
            lambda row: get_spray_direction(row['coord_x'], row['coord_y'], row['bat_side']), 
            axis=1
        )
    else:
        df['Spray'] = '-'
    
    # Format Result column
    df['Result'] = df['Result'].str.replace('_', ' ').str.title()
    
    # Create xBA (expected batting average) from hit probabilities
    df['xBA'] = ((df['Single Prob'].str.rstrip('%').astype(float) + 
                  df['Double Prob'].str.rstrip('%').astype(float) + 
                  df['Triple Prob'].str.rstrip('%').astype(float) + 
                  df['Hr Prob'].str.rstrip('%').astype(float)) / 100).round(3)
    df['xBA'] = df['xBA'].apply(lambda x: f'.{str(int(x*1000)).zfill(3)}')
    
    # Simplify probability columns - keep only HR probability as it's most impactful
    df['HR%'] = df['Hr Prob']
    
    # Select and reorder columns for clarity (added Spray column)
    columns_to_keep = ['Rank', 'Team', 'Player', 'Launch Speed', 'Launch Angle', 
                       'Spray', 'Result', 'Estimated Bases', 'xBA', 'HR%']
    df = df[columns_to_keep]
    
    # Format launch angle to include degree symbol
    df['Launch Angle'] = df['Launch Angle'].astype(str) + '°'
    
    # Format launch speed
    df['Launch Speed'] = df['Launch Speed'].astype(str) + ' mph'
    
    # Round estimated bases to 2 decimals
    df['Estimated Bases'] = df['Estimated Bases'].round(2)
    
    return df

def create_estimated_bases_table(df, away_team, home_team, away_score, home_score,
                               away_win_percentage, home_win_percentage, formatted_date,
                               mlb_team_logos, images_dir):
    """Creates enhanced table visualization of estimated bases statistics with team logos."""

    apply_base_style()

    df = df.copy().head(15)

    if 'team_color' not in df.columns:
        df['team_color'] = df['Team'].apply(
            lambda x: get_team_color(team_colors, x)
        )

    team_color_map = dict(zip(df['Team'], df['team_color']))

    # Store team names before preparing data (for logo lookup)
    team_names = df['Team'].tolist()

    # Prepare data - this now keeps full player names
    df = prepare_table_data(df)

    fig = plt.figure(figsize=(20, 10), dpi=150)
    fig.patch.set_facecolor(PALETTE['bg'])

    ax = fig.add_subplot(111)
    ax.set_position([0.05, 0.05, 0.9, 0.65])  # [left, bottom, width, height]
    ax.axis('off')
    ax.set_facecolor(PALETTE['bg'])

    # Column widths: Rank, Team, Player, Launch Speed, Launch Angle, Spray, Result, Est Bases, xBA, HR%
    col_widths = [0.05, 0.06, 0.18, 0.10, 0.10, 0.07, 0.08, 0.12, 0.07, 0.07]

    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    loc='center',
                    cellLoc='center',
                    colWidths=col_widths)

    create_enhanced_cell_styles_with_logos(table, df, team_color_map)

    table.auto_set_font_size(False)
    table.scale(1.1, 1.5)

    add_team_logos_to_table(ax, table, team_names, mlb_team_logos, df)

    title = "Top 15 Batted Balls by Estimated Total Bases"
    subtitle = (
        f"{away_team} {away_score} - {home_team} {home_score}   •   {formatted_date}\n"
        f"Win Probability: {away_team} {away_win_percentage:.0f}% • "
        f"{home_team} {home_win_percentage:.0f}%"
    )
    stamp_header(fig, title, subtitle,
                 y_title=0.93, y_subtitle=0.88,
                 title_size=23, subtitle_size=15)

    os.makedirs(images_dir, exist_ok=True)
    filename = f'{away_team}_{home_team}_{away_score}-{home_score}--{away_win_percentage:.0f}-{home_win_percentage:.0f}_estimated_bases.png'
    filepath = os.path.join(images_dir, filename)
    finalize(fig, filepath, dpi=200, apply_watermark_fn=_apply_watermark)


def create_enhanced_cell_styles_with_logos(table, df, team_color_map):
    """Apply enhanced styling to table cells, hiding team text for logo placement."""

    rank_col = 0
    team_col = 1
    player_col = 2
    bases_col = df.columns.get_loc('Estimated Bases')
    result_col = df.columns.get_loc('Result')
    xba_col = df.columns.get_loc('xBA')

    # Cream-friendly stripe colors
    row_even = PALETTE['row_alt']   # warmer cream
    row_odd  = PALETTE['bg']        # base cream
    cell_edge = '#E6DFD2'

    # Single styling pass — base + alternating fill for every (row, col).
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_height(0.06)
            cell.set_text_props(weight='bold',
                                fontsize=14 if col == team_col else 15)
            cell.set_facecolor(PALETTE['text'])
            cell.get_text().set_color(PALETTE['bg'])
            cell.set_edgecolor(PALETTE['text'])
            cell.set_linewidth(1.5)
        else:
            cell.set_height(0.055)
            cell.set_text_props(fontsize=14)
            cell.set_edgecolor(cell_edge)
            cell.set_linewidth(0.5)
            cell.set_facecolor(row_even if row % 2 == 0 else row_odd)

    # Per-row highlights — applied AFTER the base pass so they win.
    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(0, 4)
    # Compute the luminance threshold below which white text reads better
    # than dark text against the YlOrRd cmap. Drops the magic 2.5 constant.
    def _needs_white_text(rgba):
        r, g, b = rgba[0], rgba[1], rgba[2]
        return (r * 0.299 + g * 0.587 + b * 0.114) < 0.55

    for row in range(1, len(df) + 1):
        # Rank — bold
        rank_cell = table[(row, rank_col)]
        rank_cell.get_text().set_weight('bold')
        rank_cell.get_text().set_fontsize(14)

        # Team — text hidden so the logo sits in the empty cell.
        # Background was already set in the base pass.
        team_cell = table[(row, team_col)]
        team_cell.get_text().set_alpha(0)

        # Player — slightly smaller font so full names fit.
        player_cell = table[(row, player_col)]
        player_cell.get_text().set_fontsize(13)
        player_cell.get_text().set_color(PALETTE['text'])

        # Estimated bases — YlOrRd gradient with luminance-driven text color.
        bases_value = df.iloc[row - 1]['Estimated Bases']
        bases_cell = table[(row, bases_col)]
        rgba = cmap(norm(bases_value))
        bases_cell.set_facecolor(rgba)
        bases_cell.get_text().set_weight('bold')
        bases_cell.get_text().set_fontsize(14)
        bases_cell.get_text().set_color('white' if _needs_white_text(rgba) else PALETTE['text'])

        # Result — semantic colors warmed for cream background.
        result = df.iloc[row - 1]['Result']
        result_cell = table[(row, result_col)]
        if result == 'Out':
            result_cell.set_facecolor('#F7DEDA')
            result_cell.get_text().set_color('#A6362B')
        elif result == 'Home Run':
            result_cell.set_facecolor('#DDECDC')
            result_cell.get_text().set_color('#2E7D32')
        elif result in ['Single', 'Double', 'Triple']:
            result_cell.set_facecolor('#DDE7F4')
            result_cell.get_text().set_color('#1A4F8B')

        # xBA — highlight high values
        xba_value = float(df.iloc[row - 1]['xBA'])
        xba_cell = table[(row, xba_col)]
        if xba_value >= 0.500:
            xba_cell.get_text().set_weight('bold')
            xba_cell.get_text().set_color('#2E7D32')
        elif xba_value >= 0.300:
            xba_cell.get_text().set_color('#1A4F8B')


def player_contribution_chart(home_outcomes, away_outcomes, home_team, away_team,
                             home_score, away_score, home_win_percentage, away_win_percentage,
                             tie_percentage, mlb_team_logos, formatted_date, images_dir="images",
                             player_id_map=None, precomputed_home=None, precomputed_away=None):
    """
    Creates a two-panel horizontal bar chart:
      Top: hitter contributions (batted balls + walks) with PA counts.
      Bottom: pitcher bases allowed with BF counts, on its own x-axis scale.
    Uses player headshots when player_id_map is provided.
    """
    apply_base_style()

    percentages = {
        'away': f"{away_win_percentage:.0f}",
        'home': f"{home_win_percentage:.0f}",
        'tie': f"{tie_percentage:.0f}"
    }

    # ------- Aggregate outcomes -------
    player_contributions = {}   # (name, team) -> {batted_balls, walks, pa, total}
    pitcher_contributions = {}  # (name, team) -> {batted_balls, walks, bf, total}

    for team, team_outcomes, team_name in [(home_team, home_outcomes, home_team),
                                           (away_team, away_outcomes, away_team)]:
        opposing_team = away_team if team_name == home_team else home_team
        precomputed = precomputed_home if team_name == home_team else precomputed_away

        for outcome in team_outcomes:
            if not (isinstance(outcome, tuple) and len(outcome) >= 3):
                continue
            outcome_data, event_type, player_name = outcome[0], outcome[1], outcome[2]
            pitcher_name = outcome[3] if len(outcome) >= 4 else None

            # --- Hitter ---
            player_key = (player_name, team_name)
            if player_key not in player_contributions:
                player_contributions[player_key] = {'batted_balls': 0, 'walks': 0, 'pa': 0, 'total': 0}

            if outcome_data == 'walk':
                player_contributions[player_key]['walks'] += 1
                player_contributions[player_key]['total'] += 1
                player_contributions[player_key]['pa'] += 1
                if pitcher_name:
                    p_key = (pitcher_name, opposing_team)
                    if p_key not in pitcher_contributions:
                        pitcher_contributions[p_key] = {'batted_balls': 0, 'walks': 0, 'bf': 0, 'total': 0}
                    pitcher_contributions[p_key]['walks'] += 1
                    pitcher_contributions[p_key]['total'] += 1
                    pitcher_contributions[p_key]['bf'] += 1
            elif outcome_data == 'strikeout':
                player_contributions[player_key]['pa'] += 1
                if pitcher_name:
                    p_key = (pitcher_name, opposing_team)
                    if p_key not in pitcher_contributions:
                        pitcher_contributions[p_key] = {'batted_balls': 0, 'walks': 0, 'bf': 0, 'total': 0}
                    pitcher_contributions[p_key]['bf'] += 1
            elif isinstance(outcome_data, dict) and 'launch_speed' in outcome_data:
                player_contributions[player_key]['pa'] += 1
                try:
                    play_id = outcome_data.get('play_id')
                    if precomputed is not None and play_id is not None and play_id in precomputed:
                        estimated_bases = precomputed[play_id]
                    else:
                        pipeline = _get_model()
                        features = prepare_batted_ball_features(
                            launch_speed=outcome_data['launch_speed'],
                            launch_angle=outcome_data['launch_angle'],
                            venue_name=outcome_data.get('venue_name', 'Unknown'),
                            coord_x=outcome_data.get('coord_x'),
                            coord_y=outcome_data.get('coord_y'),
                            bat_side=outcome_data.get('bat_side')
                        )
                        probs = pipeline.predict_proba(features)[0]
                        estimated_bases = probs[1]*1 + probs[2]*2 + probs[3]*3 + probs[4]*4

                    player_contributions[player_key]['batted_balls'] += estimated_bases
                    player_contributions[player_key]['total'] += estimated_bases

                    if pitcher_name:
                        p_key = (pitcher_name, opposing_team)
                        if p_key not in pitcher_contributions:
                            pitcher_contributions[p_key] = {'batted_balls': 0, 'walks': 0, 'bf': 0, 'total': 0}
                        pitcher_contributions[p_key]['batted_balls'] += estimated_bases
                        pitcher_contributions[p_key]['total'] += estimated_bases
                        pitcher_contributions[p_key]['bf'] += 1
                except Exception as e:
                    print(f"Error calculating estimated bases for {player_name}: {e}")

    sorted_hitters = sorted(player_contributions.items(), key=lambda x: x[1]['total'], reverse=True)[:12]
    sorted_pitchers = sorted(pitcher_contributions.items(), key=lambda x: x[1]['total'], reverse=True)[:10]

    if not sorted_hitters:
        print("No player contributions found")
        return

    n_hitters = len(sorted_hitters)
    n_pitchers = len(sorted_pitchers)
    has_pitchers = n_pitchers > 0

    # Pre-load headshots
    if player_id_map:
        all_names = set()
        for (name, _), _ in sorted_hitters:
            all_names.add(name)
        for (name, _), _ in sorted_pitchers:
            all_names.add(name)
        for name in all_names:
            pid = player_id_map.get(name)
            if pid is not None:
                _load_headshot(pid)

    # ------- Figure with two independent axes -------
    # Add ~1.5" of vertical headroom for the dedicated title strip.
    if has_pitchers:
        fig, (ax_hit, ax_pitch) = plt.subplots(
            2, 1, figsize=(14, max(11.5, (n_hitters + n_pitchers) * 0.52 + 4.5)),
            dpi=150,
            gridspec_kw={'height_ratios': [n_hitters, n_pitchers], 'hspace': 0.30},
        )
    else:
        fig, ax_hit = plt.subplots(figsize=(14, max(9.5, n_hitters * 0.52 + 4.5)), dpi=150)
        ax_pitch = None

    fig.patch.set_facecolor(PALETTE['bg'])
    # Reserve top of figure for title strip; subplots pulled down accordingly.
    fig.subplots_adjust(top=0.85)

    # ------- Helper: draw one section -------
    def _draw_section(ax, entries, stat_key):
        """stat_key: 'pa' for hitters, 'bf' for pitchers."""
        from matplotlib.transforms import blended_transform_factory

        labels = []
        bb_vals = []
        walk_vals = []
        tc_list = []
        wc_list = []
        stat_vals = []

        for (pname, tname), contribs in entries:
            name_parts = pname.split()
            if len(name_parts) >= 2:
                fmt = f"{name_parts[0][0]}. {' '.join(name_parts[1:])}"
            else:
                fmt = pname
            labels.append(fmt)
            bb_vals.append(contribs['batted_balls'])
            walk_vals.append(contribs['walks'])
            stat_vals.append(contribs[stat_key])

            team_color = get_team_color(team_colors, tname)
            tc_list.append(team_color)
            wc_list.append(lighten(team_color, 0.5))

        y_positions = np.arange(len(entries))
        ax.set_facecolor(PALETTE['bg'])
        ax.barh(y_positions, bb_vals, color=tc_list,
                edgecolor=PALETTE['bg'], linewidth=0.6)
        ax.barh(y_positions, walk_vals, left=bb_vals, color=wc_list,
                edgecolor=PALETTE['bg'], linewidth=0.6)

        # Use blended transform: x in axes fraction (0-1), y in data coords
        # This keeps name/headshot spacing identical regardless of x-axis scale
        trans = blended_transform_factory(ax.transAxes, ax.transData)

        # Fixed x positions (axes fraction) — consistent across both panels
        headshot_x = -0.015   # just left of y-axis
        name_x = -0.055      # left of headshot

        stat_label = 'PA' if stat_key == 'pa' else 'BF'
        for idx, ((pname, tname), _) in enumerate(entries):
            # Headshot / logo
            placed = False
            if player_id_map:
                pid = player_id_map.get(pname)
                if pid is not None:
                    headshot_data = _load_headshot(pid)
                    if headshot_data is not None:
                        headshot_img = OffsetImage(headshot_data, zoom=0.38)
                        ab = AnnotationBbox(headshot_img, (headshot_x, idx),
                                           frameon=False, xycoords=trans,
                                           box_alignment=(1, 0.5))
                        ax.add_artist(ab)
                        placed = True
            if not placed:
                logo_url = get_team_logo(tname, mlb_team_logos)
                if logo_url:
                    try:
                        img_obj = getImage(logo_url, zoom=0.4, size=(30, 30), alpha=1.0)
                        if img_obj:
                            ab = AnnotationBbox(img_obj, (headshot_x, idx),
                                              frameon=False, xycoords=trans,
                                              box_alignment=(1, 0.5))
                            ax.add_artist(ab)
                    except Exception:
                        pass

            # Name label with PA/BF count
            display_name = f"{labels[idx]}  ({stat_vals[idx]} {stat_label})"
            ax.text(name_x, idx, display_name, ha='right', va='center',
                    fontsize=11, color=PALETTE['text'], transform=trans)

        # Value labels on bars
        for idx, (bb, w) in enumerate(zip(bb_vals, walk_vals)):
            if bb > 0.5:
                ax.text(bb/2, idx, f'{bb:.1f}', ha='center', va='center',
                       fontsize=10, color='white', fontweight='bold')
            if w > 0.5:
                ax.text(bb + w/2, idx, f'{w:.0f}', ha='center', va='center',
                       fontsize=10, color='white', fontweight='bold')
            total = bb + w
            ax.text(total + 0.08, idx, f'{total:.1f}', ha='left', va='center',
                   fontsize=11, color=PALETTE['text'], fontweight='bold')

        # Styling
        ax.set_yticks(y_positions)
        ax.set_yticklabels([''] * len(entries))
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(PALETTE['spine'])
        ax.spines['bottom'].set_color(PALETTE['spine'])
        ax.grid(axis='x', alpha=0.5, linestyle='--', color=PALETTE['grid'])
        ax.set_axisbelow(True)

        max_val = max(bb + w for bb, w in zip(bb_vals, walk_vals)) if bb_vals else 1
        ax.set_xlim(0, max_val * 1.18)

        # Track per-section colors for the legend (representative pair)
        return tc_list[0] if tc_list else PALETTE['text'], wc_list[0] if wc_list else PALETTE['text_muted']

    # ------- Draw hitter section -------
    rep_bb_color, rep_walk_color = _draw_section(ax_hit, sorted_hitters, 'pa')
    ax_hit.set_xlabel('Estimated Total Bases', fontsize=12, labelpad=8,
                      color=PALETTE['text'])

    # Dedicated title strip — matches Run Distribution and Spray Chart.
    # Reserves its own axes at the top of the figure with title, divider
    # rule, and subtitle row below.
    fig_h = fig.get_size_inches()[1]
    strip_height_frac = max(0.10, min(0.16, 1.5 / fig_h))
    tax = title_axes(fig, height_frac=strip_height_frac, top_pad=0.015)
    subtitle = (
        f'Actual: {away_team} {away_score} - {home_team} {home_score}   '
        f'({formatted_date})    DTW: {away_team} {percentages["away"]}% • '
        f'{home_team} {percentages["home"]}% • Tie {percentages["tie"]}%'
    )
    draw_title_block(tax, 'Player Contributions by Estimated Total Bases',
                     [subtitle], title_size=22, subtitle_size=12)

    ax_hit.set_title('Hitting  —  Estimated Bases', fontsize=13, fontweight='bold',
                    loc='left', color=PALETTE['text_muted'], pad=8,
                    fontfamily=heading_font())

    # ------- Draw pitcher section -------
    if has_pitchers and ax_pitch is not None:
        _draw_section(ax_pitch, sorted_pitchers, 'bf')
        ax_pitch.set_xlabel('Estimated Total Bases Allowed', fontsize=12, labelpad=8,
                            color=PALETTE['text'])
        ax_pitch.set_title('Pitching  —  Bases Allowed  (different scale)',
                          fontsize=13, fontweight='bold', loc='left',
                          color=PALETTE['text_muted'], pad=8,
                          fontfamily=heading_font())

    # Legend swatches use the first hitter's actual team-color pair so the
    # legend reflects the bars rather than generic greys.
    legend_patches = [
        patches.Patch(facecolor=rep_bb_color, edgecolor='none',
                      label='Batted Balls (estimated bases)'),
        patches.Patch(facecolor=rep_walk_color, edgecolor='none',
                      label='Walks (1 base each)'),
    ]
    ax_hit.legend(handles=legend_patches, loc='lower right', fontsize=10,
                 frameon=False)

    os.makedirs(images_dir, exist_ok=True)
    filename = f'{away_team}_{home_team}_{away_score}-{home_score}--{percentages["away"]}-{percentages["home"]}_player_contributions.png'
    filepath = os.path.join(images_dir, filename)
    finalize(fig, filepath, dpi=200, apply_watermark_fn=_apply_watermark)


# =============================================================================
# SPRAY CHART VISUALIZATION
# =============================================================================

def get_logo_team_name(short_name):
    """Convert short team name to logo lookup name."""
    return TEAM_LOGO_MAP.get(short_name, short_name)


def get_display_team_name(short_name):
    """Convert short team name to full display name."""
    return TEAM_DISPLAY_MAP.get(short_name, short_name)


def get_spray_direction(coord_x, coord_y, bat_side):
    """
    Calculate spray direction category for display in tables.

    Returns:
        str: 'Pull', 'Center', 'Oppo', or '-' if data unavailable
    """
    if coord_x is None or coord_y is None or bat_side is None:
        return '-'
    if pd.isna(coord_x) or pd.isna(coord_y) or pd.isna(bat_side):
        return '-'

    spray_angle = calculate_spray_angle(coord_x, coord_y)
    if bat_side == 'L':
        spray_angle = -spray_angle

    if spray_angle < -15:
        return 'Pull'
    elif spray_angle > 15:
        return 'Oppo'
    else:
        return 'Center'

def calculate_landing_distance(launch_speed, launch_angle):
    """
    Estimate where a batted ball lands using projectile physics.
    
    Uses projectile motion with empirically-tuned drag factors.
    For ground balls (negative launch angle), estimates roll distance.
    
    Args:
        launch_speed: Exit velocity in mph
        launch_angle: Launch angle in degrees
    
    Returns:
        float: Estimated landing distance in feet
    """
    v_fps = launch_speed * 1.467  # mph to ft/s
    theta_rad = np.radians(launch_angle)
    g = 32.174  # ft/s²
    
    # Ground balls (land immediately, then roll)
    if launch_angle <= 0:
        roll_distance = 30 + (launch_speed - 50) * 1.2
        return np.clip(roll_distance, 20, 150)
    
    # Projectile motion base distance
    base_distance = (v_fps ** 2 * np.sin(2 * theta_rad)) / g
    
    # Drag factors tuned to match expected distances
    if launch_angle <= 10:
        drag_factor = 0.78  # Low liners
    elif launch_angle <= 20:
        drag_factor = 0.72  # Line drives
    elif launch_angle <= 30:
        drag_factor = 0.64  # Optimal fly balls
    elif launch_angle <= 40:
        drag_factor = 0.54  # Fly balls
    elif launch_angle <= 50:
        drag_factor = 0.45  # High fly balls
    else:
        drag_factor = 0.33  # Pop-ups
    
    return np.clip(base_distance * drag_factor, 20, 500)


def calculate_expected_bases_for_spray(outcome_dict, pipeline):
    """
    Calculate expected bases for a batted ball in spray chart context.

    Returns:
        float: Expected bases (0-4 scale)
    """
    try:
        features = prepare_batted_ball_features(
            launch_speed=outcome_dict['launch_speed'],
            launch_angle=outcome_dict['launch_angle'],
            venue_name=outcome_dict['venue_name'],
            coord_x=outcome_dict.get('coord_x'),
            coord_y=outcome_dict.get('coord_y'),
            bat_side=outcome_dict.get('bat_side')
        )
        probs = pipeline.predict_proba(features)[0]
        return probs[1]*1 + probs[2]*2 + probs[3]*3 + probs[4]*4
    except Exception:
        return 0.5


# Continuous Estimated Bases colormap, anchored at PALETTE values so the
# spray chart's ring colors stay consistent with the rest of the chart family.
# Anchors: out (xbases=0) -> single (1) -> xbh (2) -> hr (4).
ESTIMATED_BASES_CMAP = LinearSegmentedColormap.from_list(
    'estimated_bases',
    [
        (0.00, PALETTE['out']),     # xbases=0 (out)
        (0.25, PALETTE['single']),  # xbases=1 (1B)
        (0.50, PALETTE['xbh']),     # xbases=2 (2B anchor)
        (1.00, PALETTE['hr']),      # xbases=4 (HR)
    ],
    N=256,
)
ESTIMATED_BASES_NORM = Normalize(vmin=0.0, vmax=4.0)


def get_expected_bases_color(xbases):
    """Return RGBA from the continuous Estimated Bases colormap.

    Anchored at PALETTE: out (xbases=0) -> single (1) -> xbh (2) -> hr (4).
    Inputs outside [0, 4] are clipped by Normalize.
    """
    return ESTIMATED_BASES_CMAP(ESTIMATED_BASES_NORM(xbases))


def get_stadium_fence_curve(venue_name):
    """
    Generate fence curve points for a specific stadium.

    Supports two data formats in STADIUM_DIMENSIONS:
      1. List of (angle, distance) tuples — used directly (new format)
      2. Dict with named keys (LF, LCF, CF, etc.) — converted via key_to_angle mapping (legacy)

    Waypoints are converted to Cartesian directly (no polar interpolation).
    matplotlib connects them with straight lines, so straight walls like
    Fenway's Green Monster render as straight lines instead of circular arcs.

    Returns:
        tuple: (x_coords, y_coords, dims_raw, angles_deg, distances_plot)
    """
    dims_raw = STADIUM_DIMENSIONS.get(venue_name, DEFAULT_STADIUM_DIMENSIONS)

    if isinstance(dims_raw, list):
        angle_dist_pairs = list(dims_raw)
    else:
        # Legacy dict format — convert using key-to-angle mapping
        key_to_angle = {
            'LF': -45,
            'LCF': -22.5,
            'DLCF': -15,
            'CF': 0,
            'RCF': 22.5,
            'DRCF': 30,
            'RF': 45,
        }
        angle_dist_pairs = [(angle, dims_raw[key])
                            for key, angle in key_to_angle.items() if key in dims_raw]

    angle_dist_pairs.sort(key=lambda x: x[0])

    angles_deg = np.array([p[0] for p in angle_dist_pairs])
    distances_ft = np.array([p[1] for p in angle_dist_pairs])
    distances_plot = distances_ft * FEET_TO_PLOT

    # Convert to Cartesian directly — straight lines between waypoints
    angles_rad = np.radians(90 - angles_deg)
    x_coords = distances_plot * np.cos(angles_rad)
    y_coords = distances_plot * np.sin(angles_rad)

    return x_coords, y_coords, dims_raw, angles_deg, distances_plot


def draw_baseball_field(ax, venue_name='default'):
    """
    Draw baseball field with stadium-specific fence dimensions.
    Grass fills only to the fence line.
    """
    fence_x, fence_y, dims, smooth_angles, smooth_distances = get_stadium_fence_curve(venue_name)

    # Vivid field colors — the cream figure background is subtle enough
    # that the field can keep its natural saturation and still read well.
    grass_color = '#9FCB8A'
    dirt_color  = '#D2B48C'
    line_color  = '#FFFFFF'
    grass_edge  = '#3F8741'

    # Create outfield grass polygon (follows fence curve)
    grass_vertices = [(0, 0)]
    for x, y in zip(fence_x, fence_y):
        grass_vertices.append((x, y))
    grass_vertices.append((0, 0))

    grass_polygon = Polygon(grass_vertices, facecolor=grass_color,
                           edgecolor=grass_edge, linewidth=2, alpha=0.6, zorder=1)
    ax.add_patch(grass_polygon)
    
    # Infield dirt — real grass-dirt arc is ~95ft from home plate
    infield_radius = 95 * FEET_TO_PLOT  # ~23.75 plot units
    infield = patches.Wedge(
        center=(0, 0), r=infield_radius,
        theta1=45, theta2=135,
        facecolor=dirt_color, edgecolor='none', alpha=0.5, zorder=2
    )
    ax.add_patch(infield)
    
    # Extract max distance and LF/CF/RF for labels
    if isinstance(dims, list):
        max_fence_ft = max(dist for _, dist in dims)
        lf_dist = dims[0][1]   # First point (most negative angle)
        rf_dist = dims[-1][1]  # Last point (most positive angle)
        # CF: find point closest to 0°
        cf_dist = min(dims, key=lambda p: abs(p[0]))[1]
    else:
        max_fence_ft = max(dims.values())
        lf_dist = dims['LF']
        cf_dist = dims['CF']
        rf_dist = dims['RF']

    # Foul lines (extend past fence for HR visibility)
    max_fence = max_fence_ft * FEET_TO_PLOT
    foul_extension = max_fence + 12

    for angle in [-45, 45]:
        angle_rad = np.radians(90 - angle)
        x_end = foul_extension * np.cos(angle_rad)
        y_end = foul_extension * np.sin(angle_rad)
        ax.plot([0, x_end], [0, y_end], color=line_color, linewidth=2, alpha=0.8, zorder=3)

    # Fence line
    ax.plot(fence_x, fence_y, color='#3F3A33', linewidth=2.5, zorder=4)

    # Distance labels just outside the fence
    label_positions = [
        (-45, lf_dist, 'right'),
        (0, cf_dist, 'center'),
        (45, rf_dist, 'left')
    ]
    
    for angle, dist_ft, ha in label_positions:
        angle_rad = np.radians(90 - angle)
        dist_plot = dist_ft * FEET_TO_PLOT
        label_dist = dist_plot + 6
        label_x = label_dist * np.cos(angle_rad)
        label_y = label_dist * np.sin(angle_rad)
        ax.text(label_x, label_y, f"{dist_ft}'", ha=ha, va='bottom',
                fontsize=11, color='#3F3A33', fontweight='bold', zorder=5)
    
    # Home plate — real proportions: 17" front, 8.5" sides, 12" back edges
    # Scaled so half-width = 2.5 plot units; back depth = sqrt(12²-8.5²)/8.5 * 2.5
    hp_hw = 2.5   # half-width
    hp_sd = 2.5   # side depth (8.5" / 8.5" * 2.5)
    hp_bd = 2.49  # back-edge depth (sqrt(71.75) / 8.5 * 2.5)
    home_plate = Polygon([
        (-hp_hw,  2.0),   # front left  (faces pitcher)
        ( hp_hw,  2.0),   # front right
        ( hp_hw, -0.5),   # right corner
        ( 0,     -3.0),   # point       (faces catcher)
        (-hp_hw, -0.5),   # left corner
    ], closed=True, facecolor='white', edgecolor='black', linewidth=1, zorder=6)
    ax.add_patch(home_plate)


def _place_spray_labels(ax, team_bbs, x_extent, axis_limit,
                        min_xbases=None, max_labels=10, fallback_top=3):
    """Place batted-ball labels with smart collision-avoiding placement.

    Scores 8 candidate offsets per label to avoid overlaps with other labels,
    stay inside axis bounds, and prefer positions below the marker. Draws a
    thin gray connecting line from the label to the batted ball marker.

    Selection rule:
      - If ``min_xbases`` is set, label every ball with ``xbases >= min_xbases``
        (capped at ``max_labels`` highest-xbases balls to prevent label hell on
        slug-fests). If nothing clears the threshold, fall back to the top
        ``fallback_top`` by xbases so quiet games still get callouts.
      - If ``min_xbases`` is None, label the top 5 by xbases (legacy behavior).

    Parameters
    ----------
    ax : matplotlib Axes
    team_bbs : list[dict]
        Batted ball dicts with keys 'x', 'y', 'xbases', 'last_name'.
    x_extent : float
        Half-width of the x-axis (symmetric about 0).
    axis_limit : float
        Upper y-axis limit.
    min_xbases : float or None
        Threshold for inclusion. None preserves legacy top-5 behavior.
    max_labels : int
        Cap on labels when ``min_xbases`` is used.
    fallback_top : int
        Number of top-xbases balls to label if nothing clears ``min_xbases``.
    """
    sorted_bbs = sorted(team_bbs, key=lambda b: b['xbases'], reverse=True)

    if min_xbases is not None:
        threshold_hits = [bb for bb in sorted_bbs if bb['xbases'] >= min_xbases]
        if threshold_hits:
            top_bbs = threshold_hits[:max_labels]
        else:
            top_bbs = sorted_bbs[:fallback_top]
    else:
        top_bbs = sorted_bbs[:5]

    top_bbs = [bb for bb in top_bbs if bb['last_name']]

    if not top_bbs:
        return

    # Candidate offsets (dx, dy in points) and alignment hints
    # (offset_pts_x, offset_pts_y, ha, va)
    _CANDIDATES = [
        ( 12, -12, 'left',   'top'),      # below-right
        (-12, -12, 'right',  'top'),      # below-left
        ( 12,  12, 'left',   'bottom'),   # above-right
        (-12,  12, 'right',  'bottom'),   # above-left
        ( 16,   0, 'left',   'center'),   # right
        (-16,   0, 'right',  'center'),   # left
        ( 10,  22, 'left',   'bottom'),   # far-above-right
        (-10,  22, 'right',  'bottom'),   # far-above-left
    ]

    # Approximate label size in data coordinates for overlap checks.
    # A rough heuristic: ~18 data-units wide, ~8 tall (depends on zoom/DPI,
    # but close enough for penalty scoring).
    label_half_w = 18
    label_half_h = 8

    placed = []  # list of (cx, cy) in data coords for placed labels

    for bb in top_bbs:
        bx, by = bb['x'], bb['y']
        best_score = -1e9
        best_pos = _CANDIDATES[0]

        for dx_pt, dy_pt, ha, va in _CANDIDATES:
            # Convert point offset to approximate data-coord offset.
            # The axes span ~2*x_extent horizontally over ~800 display points
            # (16-inch figure at 100 DPI / 2 subplots), so 1 point ≈ x_extent/400.
            scale = x_extent / 400.0
            cx = bx + dx_pt * scale
            cy = by + dy_pt * scale

            score = 0.0

            # --- Overlap penalty (moderate) ---
            # Loosened from the original 200/40 weights so denser games
            # (10+ qualifying labels) place callouts even when no
            # collision-free position exists. Slight overlap is preferable
            # to silently dropping labels.
            for px, py in placed:
                dist = max(abs(cx - px) / label_half_w,
                           abs(cy - py) / label_half_h)
                if dist < 1.0:
                    score -= 90 * (1.0 - dist)
                elif dist < 1.6:
                    score -= 18 * (1.6 - dist)

            # --- Boundary penalty ---
            margin_x = x_extent * 0.05
            margin_y_top = axis_limit * 0.12  # extra heavy near subtitle
            margin_y_bot = 3  # bottom axis starts at -3

            if cx - label_half_w * 0.5 < -x_extent + margin_x:
                score -= 100
            if cx + label_half_w * 0.5 > x_extent - margin_x:
                score -= 100
            if cy + label_half_h > axis_limit - margin_y_top:
                score -= 150  # heavier near top (subtitle)
            if cy - label_half_h < -margin_y_bot:
                score -= 80

            # --- Distance penalty (mild, prefer closer) ---
            data_dist = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5
            score -= data_dist * 0.3

            # --- Below bonus (natural reading) ---
            if dy_pt < 0:
                score += 10

            if score > best_score:
                best_score = score
                best_pos = (dx_pt, dy_pt, ha, va)

        dx_pt, dy_pt, ha, va = best_pos
        scale = x_extent / 400.0
        placed.append((bx + dx_pt * scale, by + dy_pt * scale))

        ax.annotate(
            bb['last_name'], (bx, by),
            textcoords='offset points', xytext=(dx_pt, dy_pt),
            fontsize=9, fontweight='bold', ha=ha, va=va,
            color='#1A1A1A', zorder=11,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#999999', alpha=0.92, linewidth=0.6),
            arrowprops=dict(arrowstyle='-', color='#888888',
                            linewidth=0.8, shrinkA=0, shrinkB=6),
        )


def spray_chart(home_outcomes, away_outcomes,
                home_team, away_team,
                home_score, away_score,
                home_win_percentage, away_win_percentage, tie_percentage,
                mlb_team_logos, formatted_date,
                venue_name='default',
                images_dir="images",
                pipeline=None):
    """
    Creates side-by-side spray chart visualizations with stadium-specific dimensions.

    Shows batted ball locations with team logos, colored rings indicating expected
    outcome based on the batted ball model (Out, Single, XBH, HR).

    Args:
        home_outcomes: List of home team outcome tuples
        away_outcomes: List of away team outcome tuples
        home_team: Home team short name
        away_team: Away team short name
        home_score: Home team actual score
        away_score: Away team actual score
        home_win_percentage: Home deserve-to-win %
        away_win_percentage: Away deserve-to-win %
        tie_percentage: Tie percentage
        mlb_team_logos: List of team logo dictionaries
        formatted_date: Date string for display
        venue_name: Stadium name for fence dimensions
        images_dir: Directory to save output
        pipeline: Pre-loaded model pipeline (loaded from disk if None)

    Returns:
        str: Path to saved image file
    """
    apply_base_style()

    if pipeline is None:
        pipeline = _get_model()

    percentages = {
        'away': f"{away_win_percentage:.0f}",
        'home': f"{home_win_percentage:.0f}",
        'tie': f"{tie_percentage:.0f}"
    }

    home_logo_name = get_logo_team_name(home_team)
    away_logo_name = get_logo_team_name(away_team)
    home_display_name = get_display_team_name(home_team)
    away_display_name = get_display_team_name(away_team)

    fig, (ax_away, ax_home) = plt.subplots(1, 2, figsize=(16, 9.5), dpi=150)
    fig.patch.set_facecolor(PALETTE['bg'])
    for ax in (ax_away, ax_home):
        ax.set_facecolor(PALETTE['bg'])
    plt.subplots_adjust(left=0.015, right=0.985, top=0.78, bottom=0.05, wspace=0.0)
    
    # Process outcomes
    batted_balls = {'home': [], 'away': []}
    walk_counts = {'home': 0, 'away': 0}
    
    for team_key, team_outcomes in [('home', home_outcomes), ('away', away_outcomes)]:
        for outcome in team_outcomes:
            outcome_data = outcome[0] if isinstance(outcome, tuple) else outcome
            player_name = outcome[2] if isinstance(outcome, tuple) and len(outcome) > 2 else ''

            if outcome_data == 'walk':
                walk_counts[team_key] += 1
                continue

            if not isinstance(outcome_data, dict):
                continue

            coord_x = outcome_data.get('coord_x')
            coord_y = outcome_data.get('coord_y')

            if coord_x is None or coord_y is None:
                continue

            # Spray angle from coordinates (direction is accurate)
            spray_angle = calculate_spray_angle(coord_x, coord_y)

            # Distance: use Statcast totalDistance (feet) when available; fall back to physics estimate
            launch_speed = outcome_data.get('launch_speed')
            launch_angle = outcome_data.get('launch_angle')
            total_distance = outcome_data.get('total_distance')
            if pd.notna(total_distance) and float(total_distance) > 0:
                distance_ft = float(total_distance)
            else:
                distance_ft = calculate_landing_distance(launch_speed, launch_angle)
            distance = distance_ft * FEET_TO_PLOT  # Convert to plot units

            outcome_data['venue_name'] = outcome_data.get('venue_name', venue_name)
            xbases = calculate_expected_bases_for_spray(outcome_data, pipeline)

            angle_rad = np.radians(90 - spray_angle)
            plot_x = distance * np.cos(angle_rad)
            plot_y = distance * np.sin(angle_rad)

            # Extract last name for labeling top batted balls
            last_name = player_name.split()[-1] if player_name else ''

            batted_balls[team_key].append({
                'x': plot_x,
                'y': plot_y,
                'xbases': xbases,
                'last_name': last_name,
            })
    
    # Axis limits based on stadium dimensions
    dims = STADIUM_DIMENSIONS.get(venue_name, DEFAULT_STADIUM_DIMENSIONS)
    if isinstance(dims, list):
        max_fence = max(dist for _, dist in dims) * FEET_TO_PLOT
    else:
        max_fence = max(dims.values()) * FEET_TO_PLOT
    axis_limit = max_fence + 12
    x_extent = axis_limit * 0.80

    # Plot each team
    for ax, team_key, display_name, logo_name in [
        (ax_away, 'away', away_display_name, away_logo_name),
        (ax_home, 'home', home_display_name, home_logo_name)
    ]:
        draw_baseball_field(ax, venue_name)
        ax.grid(False)

        logo_url = get_team_logo(logo_name, mlb_team_logos)

        if not logo_url:
            print(f"Warning: No logo found for {display_name} (tried: {logo_name})")

        for bb in batted_balls[team_key]:
            color = get_expected_bases_color(bb['xbases'])

            if logo_url:
                img = getImage(logo_url, zoom=0.45, size=(40, 40), alpha=0.85)
            else:
                img = None

            if img:
                ab = AnnotationBbox(img, (bb['x'], bb['y']), frameon=False, zorder=10)
                ax.add_artist(ab)

                # Subtle continuous-color halo around the logo. Lighter than
                # the previous 2.5/0.9 ring so the color reads as a tint
                # rather than a bold outline; the saturated end of the cmap
                # still carries weight for HRs.
                ring = plt.Circle(
                    (bb['x'], bb['y']), radius=5.5,
                    fill=False, edgecolor=color,
                    linewidth=1.6, alpha=0.75, zorder=9
                )
                ax.add_patch(ring)
            else:
                # Fallback: filled circle with team color + outcome ring.
                # 1.2px white stroke separates overlapping dots cleanly.
                fill_color = get_team_color(team_colors, display_name)
                dot = plt.Circle(
                    (bb['x'], bb['y']), radius=4.0,
                    facecolor=fill_color, edgecolor='white',
                    linewidth=1.2, alpha=0.9, zorder=10
                )
                ax.add_patch(dot)
                ring = plt.Circle(
                    (bb['x'], bb['y']), radius=5.5,
                    fill=False, edgecolor=color,
                    linewidth=1.6, alpha=0.75, zorder=9
                )
                ax.add_patch(ring)

        # Label every hit-quality ball (xbases >= 1.0 — well-struck singles
        # and better), capped at 10 to prevent overlap on slug-fests. Quiet
        # games fall back to top 3 by xbases so something still gets named.
        _place_spray_labels(ax, batted_balls[team_key], x_extent, axis_limit,
                            min_xbases=1.0, max_labels=10, fallback_top=3)

        ax.set_xlim(-x_extent, x_extent)
        ax.set_ylim(-3, axis_limit)
        ax.set_aspect('equal')
        ax.axis('off')

        bip_count = len(batted_balls[team_key])
        walk_count = walk_counts[team_key]
        ax.set_title(f"{display_name}\n{bip_count} BIP  •  {walk_count} BB/HBP",
                     fontsize=16, fontweight='bold', pad=8,
                     color=PALETTE['text'], fontfamily=heading_font())

    # Dedicated title strip — title, divider rule, then subtitle row.
    # Outcome legend is mounted to the right edge of the same strip so
    # everything reads as one cohesive header (no more colliding with
    # the watermark or the field).
    tax = title_axes(fig, height_frac=0.16, top_pad=0.02)
    subtitle = (
        f"Actual: {away_display_name} {away_score} - {home_display_name} {home_score}   "
        f"({formatted_date})    DTW: {away_display_name} {percentages['away']}% • "
        f"{home_display_name} {percentages['home']}% • Tie {percentages['tie']}%"
    )
    draw_title_block(tax, "Batted Ball Spray Chart", [subtitle],
                     title_size=22, subtitle_size=12)

    # Continuous Estimated Bases legend — horizontal colorbar inset.
    # Centered at fig x=0.5 in the same vertical band the old discrete
    # legend lived in (below the title strip, above the subplots).
    cbar_ax = fig.add_axes([0.35, 0.808, 0.30, 0.018])
    sm = ScalarMappable(norm=ESTIMATED_BASES_NORM, cmap=ESTIMATED_BASES_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(['0', '1', '2', '3', '4'])
    cbar.ax.tick_params(labelsize=10, length=0, pad=3, colors=PALETTE['text'])
    cbar.outline.set_visible(False)

    fig.text(0.5, 0.852, 'Estimated Bases',
             fontsize=11, fontweight='bold', ha='center', va='bottom',
             color=PALETTE['text'], fontfamily=heading_font())

    # Outcome subtitle row beneath the numeric ticks (Out / 1B / 2B / 3B / HR).
    # Five evenly-spaced x positions matching cbar tick centers.
    for x_frac, outcome_label in zip(
        [0.35, 0.425, 0.50, 0.575, 0.65],
        ['Out', '1B', '2B', '3B', 'HR']
    ):
        fig.text(x_frac, 0.778, outcome_label,
                 fontsize=8.5, ha='center', va='top',
                 color=PALETTE['text_muted'], fontstyle='italic')

    fig.text(0.5, 0.015, 'Stadium dimensions are estimated for this visual',
             fontsize=9, fontstyle='italic', color=PALETTE['text_muted'],
             ha='center', va='bottom')

    os.makedirs(images_dir, exist_ok=True)
    filename = (f"{away_display_name}_{home_display_name}_{away_score}-{home_score}--"
                f"{percentages['away']}-{percentages['home']}_spray.png")
    filepath = os.path.join(images_dir, filename)

    fig.savefig(filepath, dpi=200, facecolor=PALETTE['bg'], edgecolor='none',
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    _apply_watermark(filepath)

    print(f"Saved spray chart: {filepath}")
    return filepath
                    
def add_team_logos_to_table(ax, table, team_names, mlb_team_logos, df):
    """Add team logos to the table at the appropriate cell positions."""
    
    team_col = 1  # Team column index
    
    # Force a draw to ensure table is fully rendered
    ax.figure.canvas.draw()
    
    # Get renderer
    renderer = ax.figure.canvas.get_renderer()
    
    # Calculate positions for each team logo
    for row_idx, team_name in enumerate(team_names, start=1):
        try:
            # Get the cell for this team
            cell = table[(row_idx, team_col)]
            
            # Get cell position in display coordinates
            cell_bbox = cell.get_window_extent(renderer)
            
            # Calculate center of cell in display coordinates
            cell_center_display = [
                (cell_bbox.x0 + cell_bbox.x1) / 2,
                (cell_bbox.y0 + cell_bbox.y1) / 2
            ]
            
            # Transform to axes coordinates (0-1 range)
            inv_transform = ax.transAxes.inverted()
            cell_center_axes = inv_transform.transform([cell_center_display])[0]
            
            # Get team logo URL
            logo_url = get_team_logo(team_name, mlb_team_logos)
            
            if logo_url:
                # Get the logo image with smaller size for better fit
                logo_size = (28, 28)  # Further reduced for better cell fit
                img = getImage(logo_url, zoom=0.75, size=logo_size, alpha=1.0)  # Reduced zoom for smaller appearance
                
                if img:
                    # Create annotation box for the logo using axes coordinates
                    ab = AnnotationBbox(img, cell_center_axes,
                                      frameon=False,
                                      xycoords='axes fraction',
                                      box_alignment=(0.5, 0.5))
                    ax.add_artist(ab)
        except Exception as e:
            print(f"Error adding logo for {team_name} at row {row_idx}: {e}")
