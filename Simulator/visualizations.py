import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap, to_rgb
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
        logo_h = max(25, int(ref * 0.04))
        logo_w = int(logo_h * logo.width / logo.height)
        logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

        # Prepare text — sized so it's slightly wider than the logo
        text = 'Data: MLB  |  @mlb_simulator'
        font_size = max(12, int(ref * 0.013))
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

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.close('all')
    
    percentages = {
        'away': f"{away_win_percentage:.0f}",
        'home': f"{home_win_percentage:.0f}",
        'tie': f"{tie_percentage:.0f}"
    }
    
    # Calculate modes
    modes = {
        'home': stats.mode(home_runs_scored, keepdims=True),
        'away': stats.mode(away_runs_scored, keepdims=True)
    }
    
    mode_strs = {team: ','.join(map(str, 
                mode.mode.flatten() if hasattr(mode, 'mode') 
                else mode.flatten() if isinstance(mode, np.ndarray)
                else [mode])) 
                for team, mode in modes.items()}
    
    # Create figure with improved resolution
    plt.figure(figsize=(12, 8), dpi=150)
    
    # Set up bins and colors
    max_runs = max(max(home_runs_scored), max(away_runs_scored))
    bins = range(max_runs + 2)
    
    # Pick team colors — away team gets lower alpha for separation
    home_color = team_colors.get(home_team, ['#333333', '#666666'])[0]
    away_color = team_colors.get(away_team, ['#333333', '#666666'])[0]

    # Create histograms with enhanced styling
    for runs, team, color, alpha, pattern in [
        (home_runs_scored, home_team, home_color, 0.85, '//'),
        (away_runs_scored, away_team, away_color, 0.55, '\\')
    ]:
        plt.hist(runs, bins=bins, alpha=alpha, label=team,
                color=color, edgecolor='black',
                linewidth=1.2, hatch=pattern)

    # Enhanced axis formatting
    ax = plt.gca()
    ax.set_xticks(np.arange(max_runs + 1) + 0.5)
    ax.set_xticklabels(range(max_runs + 1), fontsize=12)
    
    # Set integer y-axis ticks
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.yticks(fontsize=12)
    
    # Improved labels
    plt.xlabel('Runs Scored', fontsize=14, labelpad=10)
    plt.ylabel('Frequency', fontsize=14, labelpad=10)
    
    # Title: bold main line + plain subtitle (using ax.text for precise control)
    scores_close = abs(home_score - away_score) <= 2
    title_y = 1.29 if scores_close else 1.22
    subtitle_y = title_y - 0.04

    ax.text(0.0, title_y,
            f'Distribution of Runs Scored ({num_simulations:,} Simulations)',
            transform=ax.transAxes, fontsize=16, fontweight='bold', ha='left', va='top')

    subtitle = (
        f'Actual Score: {away_team} {away_score} - {home_team} {home_score}  ({formatted_date})\n'
        f'Deserve-to-Win: {away_team} {percentages["away"]}% - {home_team} '
        f'{percentages["home"]}%, Tie {percentages["tie"]}%\n'
        f'Most Likely Outcome: {away_team} {mode_strs["away"]} - {home_team} '
        f'{mode_strs["home"]}'
    )
    ax.text(0.0, subtitle_y, subtitle, transform=ax.transAxes,
            fontsize=12, color='#333333', ha='left', va='top', linespacing=1.5)

    # Actual score vertical lines with labels just above the plot area
    # Away gets lower alpha to match its histogram
    score_labels = [
        (away_score, away_color, away_team, 0.55),
        (home_score, home_color, home_team, 0.85)
    ]
    for score, color, team, alpha in score_labels:
        ax.axvline(x=score + 0.5, color=color, linestyle='--', linewidth=2.5, alpha=alpha, zorder=5)

    # Place labels above the plot in axes coordinates
    xlim = ax.get_xlim()
    x_range = xlim[1] - xlim[0]

    base_y = 1.005  # Just above the plot edge
    label_offsets = [base_y, base_y]  # [away, home]

    # If scores are close (within 2 runs), offset one label higher to avoid overlap
    if scores_close:
        if away_score <= home_score:
            label_offsets[0] = base_y + 0.055
        else:
            label_offsets[1] = base_y + 0.055

    for idx, (score, color, team, alpha) in enumerate(score_labels):
        x_frac = (score + 0.5 - xlim[0]) / x_range
        ax.text(x_frac, label_offsets[idx], f'{team}: {score}\n(Actual)',
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                color=color, alpha=alpha, va='bottom', ha='center', zorder=6)

    # Enhanced legend in top right
    plt.legend(fontsize=12, frameon=True, framealpha=0.9,
              edgecolor='black', fancybox=True, loc='upper right')

    # Clean up spines
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)

    # Save with high quality
    os.makedirs(images_dir, exist_ok=True)
    filename = f'{away_team}_{home_team}_{str(away_score)}-{str(home_score)}--{percentages["away"]}-{percentages["home"]}_rd.png'
    filepath = os.path.join(images_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=200)
    plt.close()
    _apply_watermark(filepath)

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
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.close('all')
    
    # Prepare data - take top 15 instead of top 10
    df = df.copy().head(15)
    
    # Add team colors if not present
    if 'team_color' not in df.columns:
        df['team_color'] = df['Team'].apply(
            lambda x: team_colors.get(x, {}).get('primary', '#333333') 
            if isinstance(team_colors.get(x), dict) 
            else team_colors.get(x, ['#333333'])[0]
        )
    
    team_color_map = dict(zip(df['Team'], df['team_color']))
    
    # Store team names before preparing data (for logo lookup)
    team_names = df['Team'].tolist()
    
    # Prepare data - this now keeps full player names
    df = prepare_table_data(df)
    
    # Create figure with wider, less tall proportions
    fig = plt.figure(figsize=(20, 10), dpi=150)
    
    # Add subplot with more space at top for titles
    ax = fig.add_subplot(111)
    ax.set_position([0.05, 0.05, 0.9, 0.65])  # [left, bottom, width, height]
    ax.axis('off')
    
    # Column widths: Rank, Team, Player, Launch Speed, Launch Angle, Spray, Result, Est Bases, xBA, HR%
    col_widths = [0.05, 0.06, 0.18, 0.10, 0.10, 0.07, 0.08, 0.12, 0.07, 0.07]
    
    # Create table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    loc='center',
                    cellLoc='center',
                    colWidths=col_widths)
    
    # Apply enhanced styling (modified to hide team text)
    create_enhanced_cell_styles_with_logos(table, df, team_color_map)
    
    # Scale table with adjusted scaling
    table.auto_set_font_size(False)
    table.scale(1.1, 1.5)  # Reduced height scaling from 2.0 to 1.5
    
    # Add team logos after table is created and scaled
    add_team_logos_to_table(ax, table, team_names, mlb_team_logos, df)
    
    # Enhanced title with better spacing - moved all text elements up
    title_lines = [
        f"Top 15 Batted Balls by Estimated Total Bases",
        f"{away_team} {away_score} - {home_team} {home_score}  •  {formatted_date}",
        f"Win Probability: {away_team} {away_win_percentage:.0f}% - {home_team} {home_win_percentage:.0f}%"
    ]
    
    # Main title - moved higher to 0.94
    plt.text(0.5, 0.93, title_lines[0], transform=fig.transFigure,
             fontsize=23, fontweight='bold', ha='center', va='top')
    
    # Subtitle lines - moved higher
    plt.text(0.5, 0.88, title_lines[1], transform=fig.transFigure,
             fontsize=16, ha='center', va='top', color='#333333')
    
    plt.text(0.5, 0.84, title_lines[2], transform=fig.transFigure,
             fontsize=14, ha='center', va='top', color='#666666')
    
    # Save with high quality
    os.makedirs(images_dir, exist_ok=True)
    filename = f'{away_team}_{home_team}_{away_score}-{home_score}--{away_win_percentage:.0f}-{home_win_percentage:.0f}_estimated_bases.png'
    filepath = os.path.join(images_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=200,
                facecolor='white', edgecolor='none')
    plt.close()
    _apply_watermark(filepath)


def create_enhanced_cell_styles_with_logos(table, df, team_color_map):
    """Apply enhanced styling to table cells, hiding team text for logo placement."""
    
    # Column indices
    rank_col = 0
    team_col = 1
    player_col = 2
    bases_col = df.columns.get_loc('Estimated Bases')
    result_col = df.columns.get_loc('Result')
    xba_col = df.columns.get_loc('xBA')
    hr_col = df.columns.get_loc('HR%')
    
    # Helper function for color brightness
    def is_dark_color(color):
        from matplotlib.colors import to_rgb
        r, g, b = to_rgb(color)
        return (r * 0.299 + g * 0.587 + b * 0.114) < 0.5
    
    # Style all cells
    for (row, col), cell in table.get_celld().items():
        
        # Base cell styling
        if row == 0:  # Header row
            cell.set_height(0.06)
            # Adjust font size for Team header since column is narrower
            if col == team_col:
                cell.set_text_props(weight='bold', fontsize=14)
            else:
                cell.set_text_props(weight='bold', fontsize=15)
            cell.set_facecolor('#2C3E50')
            cell.get_text().set_color('white')
            cell.set_edgecolor('#1A252F')
            cell.set_linewidth(2)
        else:  # Data rows
            cell.set_height(0.055)
            cell.set_text_props(fontsize=14)
            cell.set_edgecolor('#E0E0E0')
            cell.set_linewidth(0.5)
            
            # Alternating row colors for better readability
            if row % 2 == 0:
                cell.set_facecolor('#F8F9FA')
            else:
                cell.set_facecolor('#FFFFFF')
    
    # Apply special formatting
    for row in range(1, len(df) + 1):
        
        # Rank column - bold and centered
        rank_cell = table[(row, rank_col)]
        rank_cell.get_text().set_weight('bold')
        rank_cell.get_text().set_fontsize(14)
        
        # Team column - hide text and use alternating row background
        team = df.iloc[row-1]['Team']
        team_cell = table[(row, team_col)]
        # Use alternating row colors to match other columns
        if row % 2 == 0:
            team_cell.set_facecolor('#F8F9FA')  # Light gray for even rows
        else:
            team_cell.set_facecolor('#FFFFFF')  # White for odd rows
        # Make text transparent/invisible since we'll add logo
        team_cell.get_text().set_alpha(0)
        
        # Player names - keep readable size for full names
        player_cell = table[(row, player_col)]
        player_cell.get_text().set_fontsize(13)  # Slightly smaller to fit full names
        
        # Estimated bases - gradient coloring
        bases_value = df.iloc[row-1]['Estimated Bases']
        bases_cell = table[(row, bases_col)]
        
        # Create gradient from light yellow to dark orange/red
        norm = plt.Normalize(0, 4)
        cmap = plt.cm.YlOrRd
        color = cmap(norm(bases_value))
        bases_cell.set_facecolor(color)
        bases_cell.get_text().set_weight('bold')
        bases_cell.get_text().set_fontsize(14)
        if bases_value > 2.5:
            bases_cell.get_text().set_color('white')
        
        # Result column - color coding
        result = df.iloc[row-1]['Result']
        result_cell = table[(row, result_col)]
        if result == 'Out':
            result_cell.set_facecolor('#FFE5E5')
            result_cell.get_text().set_color('#D32F2F')
        elif result == 'Home Run':
            result_cell.set_facecolor('#E8F5E9')
            result_cell.get_text().set_color('#2E7D32')
        elif result in ['Single', 'Double', 'Triple']:
            result_cell.set_facecolor('#E3F2FD')
            result_cell.get_text().set_color('#1565C0')
        
        # xBA column - highlight high values
        xba_value = float(df.iloc[row-1]['xBA'])
        xba_cell = table[(row, xba_col)]
        if xba_value >= 0.500:
            xba_cell.get_text().set_weight('bold')
            xba_cell.get_text().set_color('#2E7D32')
        elif xba_value >= 0.300:
            xba_cell.get_text().set_color('#1565C0')


def player_contribution_chart(home_outcomes, away_outcomes, home_team, away_team,
                             home_score, away_score, home_win_percentage, away_win_percentage,
                             tie_percentage, mlb_team_logos, formatted_date, images_dir="images",
                             player_id_map=None):
    """
    Creates a two-panel horizontal bar chart:
      Top: hitter contributions (batted balls + walks) with PA counts.
      Bottom: pitcher bases allowed with BF counts, on its own x-axis scale.
    Uses player headshots when player_id_map is provided.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.close('all')

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
    if has_pitchers:
        fig, (ax_hit, ax_pitch) = plt.subplots(
            2, 1, figsize=(14, max(10, (n_hitters + n_pitchers) * 0.52 + 3)),
            dpi=150,
            gridspec_kw={'height_ratios': [n_hitters, n_pitchers], 'hspace': 0.30},
        )
    else:
        fig, ax_hit = plt.subplots(figsize=(14, max(8, n_hitters * 0.52 + 3)), dpi=150)
        ax_pitch = None

    # ------- Helper: draw one section -------
    def _draw_section(ax, entries, color_style, stat_key):
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

            team_color = team_colors.get(tname, ['#666666'])[0]
            tc_list.append(team_color)
            r, g, b = to_rgb(team_color)
            wc_list.append((r + (1 - r) * 0.5, g + (1 - g) * 0.5, b + (1 - b) * 0.5))

        y_positions = np.arange(len(entries))

        ax.barh(y_positions, bb_vals, color=tc_list, edgecolor='black', linewidth=0.5)
        ax.barh(y_positions, walk_vals, left=bb_vals, color=wc_list, edgecolor='black', linewidth=0.5)

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
                    fontsize=11, color='black', transform=trans)

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
                   fontsize=11, color='black', fontweight='bold')

        # Styling
        ax.set_yticks(y_positions)
        ax.set_yticklabels([''] * len(entries))
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        max_val = max(bb + w for bb, w in zip(bb_vals, walk_vals)) if bb_vals else 1
        ax.set_xlim(0, max_val * 1.18)

    # ------- Draw hitter section -------
    _draw_section(ax_hit, sorted_hitters, 'hitter', 'pa')
    ax_hit.set_xlabel('Estimated Total Bases', fontsize=12, labelpad=8)

    # ------- Titles anchored to axes (no floating fig.text gap) -------
    subtitle = (
        f'Actual Score: {away_team} {away_score} - {home_team} {home_score}  ({formatted_date})    '
        f'Deserve-to-Win: {away_team} {percentages["away"]}% - {home_team} '
        f'{percentages["home"]}%, Tie {percentages["tie"]}%'
    )
    fig.suptitle('Player Contributions by Estimated Total Bases',
                 fontsize=16, fontweight='bold', x=0.01, ha='left', y=0.98)
    fig.text(0.01, 0.955, subtitle, fontsize=11, color='#555555', va='top')

    ax_hit.set_title('Hitting — Estimated Bases', fontsize=13, fontweight='bold',
                    loc='left', color='#555555', pad=8)

    # ------- Draw pitcher section -------
    if has_pitchers and ax_pitch is not None:
        _draw_section(ax_pitch, sorted_pitchers, 'pitcher', 'bf')
        ax_pitch.set_xlabel('Estimated Total Bases Allowed', fontsize=12, labelpad=8)
        ax_pitch.set_title('Pitching — Bases Allowed  (different scale)',
                          fontsize=13, fontweight='bold', loc='left', color='#555555', pad=8)

    # Legend on hitter axis
    legend_patches = [
        patches.Patch(facecolor='#444444', edgecolor='black', linewidth=0.5, label='Batted Balls'),
        patches.Patch(facecolor='#AAAAAA', edgecolor='black', linewidth=0.5, label='Walks'),
    ]
    ax_hit.legend(handles=legend_patches, loc='lower right', fontsize=10,
                 frameon=True, framealpha=0.9, edgecolor='black')

    # Save
    os.makedirs(images_dir, exist_ok=True)
    filename = f'{away_team}_{home_team}_{away_score}-{home_score}--{percentages["away"]}-{percentages["home"]}_player_contributions.png'
    filepath = os.path.join(images_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=200,
                facecolor='white', edgecolor='none')
    plt.close()
    _apply_watermark(filepath)


# =============================================================================
# SPRAY CHART VISUALIZATION
# =============================================================================

def get_logo_team_name(short_name):
    """Convert short team name to logo lookup name."""
    return TEAM_LOGO_MAP.get(short_name, short_name)


def get_display_team_name(short_name):
    """Convert short team name to full display name."""
    return TEAM_DISPLAY_MAP.get(short_name, short_name)


# Statcast coordinate system constants
HOME_PLATE_X = 125.42
HOME_PLATE_Y = 199.02


def get_spray_direction(coord_x, coord_y, bat_side):
    """
    Calculate spray direction category for display in tables.
    
    Returns:
        str: 'Pull', 'Center', 'Oppo', or '-' if data unavailable
    """
    # Check for missing data
    if coord_x is None or coord_y is None or bat_side is None:
        return '-'
    if pd.isna(coord_x) or pd.isna(coord_y) or pd.isna(bat_side):
        return '-'
    
    # Calculate raw spray angle
    delta_x = coord_x - HOME_PLATE_X
    delta_y = HOME_PLATE_Y - coord_y
    angle_rad = np.arctan2(delta_x, delta_y)
    spray_angle = np.degrees(angle_rad)
    
    # Adjust for handedness (flip for lefties so pull is always negative)
    if bat_side == 'L':
        spray_angle = -spray_angle
    
    # Categorize
    if spray_angle < -15:
        return 'Pull'
    elif spray_angle > 15:
        return 'Oppo'
    else:
        return 'Center'


def calculate_spray_angle(coord_x, coord_y):
    """
    Calculate spray angle from Statcast coordinates.
    
    Returns:
        float: Angle in degrees where 0° = CF, negative = LF, positive = RF
    """
    delta_x = coord_x - HOME_PLATE_X
    delta_y = HOME_PLATE_Y - coord_y
    angle_rad = np.arctan2(delta_x, delta_y)
    return np.degrees(angle_rad)

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


def get_expected_bases_color(xbases):
    """
    Get color for expected bases value.
    
    4-category color scheme:
    - Out: < 0.5 expected bases (gray)
    - Single: 0.5 - 1.5 expected bases (orange)
    - XBH: 1.5 - 3.0 expected bases (tomato)
    - HR: > 3.0 expected bases (crimson)
    """
    if xbases < 0.5:
        return '#808080'  # Gray - Out
    elif xbases < 1.5:
        return '#FFA500'  # Orange - Single
    elif xbases < 3.0:
        return '#FF6347'  # Tomato - XBH
    else:
        return '#DC143C'  # Crimson - HR


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
    
    grass_color = '#90EE90'
    dirt_color = '#D2B48C'
    line_color = '#FFFFFF'
    
    # Create outfield grass polygon (follows fence curve)
    grass_vertices = [(0, 0)]
    for x, y in zip(fence_x, fence_y):
        grass_vertices.append((x, y))
    grass_vertices.append((0, 0))
    
    grass_polygon = Polygon(grass_vertices, facecolor=grass_color, 
                           edgecolor='#228B22', linewidth=2, alpha=0.6, zorder=1)
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
    ax.plot(fence_x, fence_y, color='#333333', linewidth=3, zorder=4)

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
                fontsize=9, color='#444444', fontweight='bold', zorder=5)
    
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


def _place_spray_labels(ax, team_bbs, x_extent, axis_limit):
    """Place labels for the top 3 batted balls with smart positioning.

    Scores 8 candidate offsets per label to avoid overlaps with other labels,
    stay inside axis bounds, and prefer positions below the marker.  Draws a
    thin gray connecting line from the label to the batted ball marker.

    Parameters
    ----------
    ax : matplotlib Axes
    team_bbs : list[dict]
        Batted ball dicts with keys 'x', 'y', 'xbases', 'last_name'.
    x_extent : float
        Half-width of the x-axis (symmetric about 0).
    axis_limit : float
        Upper y-axis limit.
    """
    top_bbs = sorted(team_bbs, key=lambda b: b['xbases'], reverse=True)[:3]
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

            # --- Overlap penalty (steep) ---
            for px, py in placed:
                dist = max(abs(cx - px) / label_half_w,
                           abs(cy - py) / label_half_h)
                if dist < 1.0:
                    score -= 200 * (1.0 - dist)
                elif dist < 1.8:
                    score -= 40 * (1.8 - dist)

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
            fontsize=8, fontweight='bold', ha=ha, va=va,
            color='#222222', zorder=11,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='#aaaaaa', alpha=0.85, linewidth=0.5),
            arrowprops=dict(arrowstyle='-', color='#999999',
                            linewidth=0.7, shrinkA=0, shrinkB=6),
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
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.close('all')

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
    
    fig, (ax_away, ax_home) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=150)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.03, wspace=0.02)
    
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

            # Distance from physics (landing distance based on EV/LA)
            launch_speed = outcome_data.get('launch_speed')
            launch_angle = outcome_data.get('launch_angle')
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

                ring = plt.Circle(
                    (bb['x'], bb['y']), radius=5.5,
                    fill=False, edgecolor=color,
                    linewidth=2.5, alpha=0.9, zorder=9
                )
                ax.add_patch(ring)
            else:
                # Fallback: filled circle with team color + outcome ring
                fill_color = team_colors.get(display_name, ('#666666', '#666666'))[0]
                dot = plt.Circle(
                    (bb['x'], bb['y']), radius=4.0,
                    facecolor=fill_color, edgecolor='white',
                    linewidth=1.0, alpha=0.85, zorder=10
                )
                ax.add_patch(dot)
                ring = plt.Circle(
                    (bb['x'], bb['y']), radius=5.5,
                    fill=False, edgecolor=color,
                    linewidth=2.5, alpha=0.9, zorder=9
                )
                ax.add_patch(ring)

        # Label top 3 batted balls by expected bases with smart placement
        _place_spray_labels(ax, batted_balls[team_key], x_extent, axis_limit)

        ax.set_xlim(-x_extent, x_extent)
        ax.set_ylim(-3, axis_limit)
        ax.set_aspect('equal')
        ax.axis('off')

        bip_count = len(batted_balls[team_key])
        walk_count = walk_counts[team_key]
        ax.set_title(f"{display_name}\n{bip_count} BIP  •  {walk_count} BB/HBP",
                     fontsize=14, fontweight='bold', pad=4)

    title_line1 = "Batted Ball Spray Chart"
    title_line2 = (
        f"Actual: {away_display_name} {away_score} - {home_display_name} {home_score}  ({formatted_date})    "
        f"DTW: {away_display_name} {percentages['away']}% - "
        f"{home_display_name} {percentages['home']}%, Tie {percentages['tie']}%"
    )
    fig.text(0.5, 0.97, title_line1,
             fontsize=15, fontweight='bold', ha='center', va='top')
    fig.text(0.5, 0.935, title_line2,
             fontsize=11, color='#333333', ha='center', va='top')

    from matplotlib.lines import Line2D
    legend_items = [
        ('Out', '#808080'),
        ('Single', '#FFA500'),
        ('XBH', '#FF6347'),
        ('HR', '#DC143C')
    ]
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=color, markeredgewidth=2.5, markersize=11, label=label)
        for label, color in legend_items
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4,
               fontsize=11, frameon=False, title='Expected Outcome',
               title_fontproperties={'weight': 'bold', 'size': 11},
               bbox_to_anchor=(0.5, 0.895))

    fig.text(0.5, 0.02, 'Stadium dimensions are estimated for this visual',
             fontsize=9, fontstyle='italic', color='gray', ha='center', va='bottom')

    os.makedirs(images_dir, exist_ok=True)
    filename = (f"{away_display_name}_{home_display_name}_{away_score}-{home_score}--"
                f"{percentages['away']}-{percentages['home']}_spray.png")
    filepath = os.path.join(images_dir, filename)

    plt.savefig(filepath, dpi=200, facecolor='white', edgecolor='none',
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
