import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from scipy.interpolate import interp1d
import seaborn as sns
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import colorsys
import matplotlib.colors as colors
import pandas as pd
import numpy as np
from scipy import stats

from Simulator.constants import (
    team_colors, 
    STADIUM_DIMENSIONS, DEFAULT_STADIUM_DIMENSIONS, FEET_TO_PLOT,
    TEAM_LOGO_MAP, TEAM_DISPLAY_MAP
)

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import requests
from io import BytesIO
from PIL import Image, ImageEnhance
import os
import ast

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
    
    # for team, team_outcomes in [('home', home_outcomes), ('away', away_outcomes)]:
    #     outcomes[team]['ev'] = [o[0] for o in team_outcomes if isinstance(o, list)]
    #     outcomes[team]['la'] = [o[1] for o in team_outcomes if isinstance(o, list)]

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
    contour_data = pd.read_csv('Data/contour_data.csv').dropna()
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
    
    # Enhanced metadata with larger font
    plt.text(0.02, 0.02, 'Data: MLB\nBy: @mlb_simulator', transform=plt.gca().transAxes, 
             fontsize=12, color='gray', ha='left', va='bottom')
    
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
    # plt.savefig(os.path.join(images_dir, filename), bbox_inches='tight', dpi=300)
    # plt.savefig(os.path.join(images_dir, filename), dpi=300)
    plt.savefig(os.path.join(images_dir, filename), bbox_inches='tight', dpi=350)
    plt.close(fig)


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
    
    # Custom colors with better contrast
    home_color = team_colors[home_team][0]
    away_color = team_colors[away_team][0]
    
    # Create histograms with enhanced styling
    for runs, team, color, pattern in [
        (home_runs_scored, home_team, home_color, '//'),
        (away_runs_scored, away_team, away_color, '\\')
    ]:
        plt.hist(runs, bins=bins, alpha=0.75, label=team,
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
    
    # Enhanced title with better spacing
    title = (f'Distribution of Runs Scored ({num_simulations:,} Simulations)\n'
            f'Actual Score: {away_team} {away_score} - {home_team} {home_score}  ({formatted_date})\n'
            f'Deserve-to-Win: {away_team} {percentages["away"]}% - {home_team} '
            f'{percentages["home"]}%, Tie {percentages["tie"]}%\n'
            f'Most Likely Outcome: {away_team} {mode_strs["away"]} - {home_team} '
            f'{mode_strs["home"]}')
    
    plt.title(title, fontsize=16, loc='left', pad=15, fontweight='bold')
    
    # Larger watermark
    plt.text(0.01, -0.09, 'Data: MLB', transform=plt.gca().transAxes,
             fontsize=12, color='gray', ha='left', va='bottom')
    plt.text(0.01, -0.12, 'By: @mlb_simulator', transform=plt.gca().transAxes,
             fontsize=12, color='gray', ha='left', va='bottom')
    
    # Enhanced legend in top right
    plt.legend(fontsize=12, frameon=True, framealpha=0.9,
              edgecolor='black', fancybox=True, loc='upper right')
    
    # Clean up spines
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)
    
    # Save with high quality
    os.makedirs(images_dir, exist_ok=True)
    filename = f'{away_team}_{home_team}_{str(away_score)}-{str(home_score)}--{percentages["away"]}-{percentages["home"]}_rd.png'
    plt.savefig(os.path.join(images_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

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
    
    # Attribution - back to top left corner
    plt.text(0.1, 0.915, 'Data: MLB', 
             transform=fig.transFigure, fontsize=18, 
             ha='left', va='top', color='#999999')
    
    plt.text(0.1, 0.89, 'By: @mlb_simulator', 
             transform=fig.transFigure, fontsize=18, 
             ha='left', va='top', color='#999999')
    
    # Save with high quality
    os.makedirs(images_dir, exist_ok=True)
    filename = f'{away_team}_{home_team}_{away_score}-{home_score}--{away_win_percentage:.0f}-{home_win_percentage:.0f}_estimated_bases.png'
    plt.savefig(os.path.join(images_dir, filename), 
                bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()


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
                             tie_percentage, mlb_team_logos, formatted_date, images_dir="images"):
    """
    Creates a horizontal stacked bar chart showing individual player contributions to the game.
    Each bar shows total estimated bases broken down by type (batted balls, walks).
    
    Args:
        home_outcomes: List of home team outcomes
        away_outcomes: List of away team outcomes  
        home_team: Home team name
        away_team: Away team name
        home_score: Home team actual score
        away_score: Away team actual score
        home_win_percentage: Home team deserve-to-win percentage
        away_win_percentage: Away team deserve-to-win percentage
        tie_percentage: Tie percentage
        mlb_team_logos: List of team logo URLs
        formatted_date: Formatted date string
        images_dir: Directory to save images
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.close('all')
    
    percentages = {
        'away': f"{away_win_percentage:.0f}",
        'home': f"{home_win_percentage:.0f}",
        'tie': f"{tie_percentage:.0f}"
    }
    
    # Process outcomes to aggregate by player
    player_contributions = {}
    
    # Process each team's outcomes
    for team, team_outcomes, team_name in [(home_team, home_outcomes, home_team), 
                                       (away_team, away_outcomes, away_team)]:
        for outcome in team_outcomes:
            # Handle tuple format (outcome_data, event_type, player_name)
            if isinstance(outcome, tuple) and len(outcome) >= 3:
                outcome_data, event_type, player_name = outcome[0], outcome[1], outcome[2]
                
                # Create unique player key with team
                player_key = (player_name, team_name)
                
                # Initialize player data if not exists
                if player_key not in player_contributions:
                    player_contributions[player_key] = {
                        'batted_balls': 0,
                        'walks': 0,
                        'total': 0
                    }
                
                # Categorize and add contribution
                if outcome_data == 'walk':
                    player_contributions[player_key]['walks'] += 1
                    player_contributions[player_key]['total'] += 1
                elif isinstance(outcome_data, dict) and 'launch_speed' in outcome_data:
                    # NEW: Handle dict format from updated outcomes
                    import joblib
                    from Simulator.game_simulator import prepare_batted_ball_features
                    
                    try:
                        pipeline = joblib.load('Model/batted_ball_model.pkl')
                        
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
                    except Exception as e:
                        print(f"Error calculating estimated bases for {player_name}: {e}")
                elif isinstance(outcome_data, list) and len(outcome_data) >= 3:
                    # LEGACY: Handle old list format [launch_speed, launch_angle, stadium]
                    import pickle
                    from Simulator.game_simulator import create_features_for_prediction
                    
                    try:
                        with open('Model/gb_classifier_pipeline_improved.pkl', 'rb') as file:
                            pipeline = pickle.load(file)
                        
                        launch_speed, launch_angle, stadium = outcome_data
                        features = create_features_for_prediction(launch_speed, launch_angle, stadium)
                        probs = pipeline.predict_proba(features)[0]
                        estimated_bases = (probs[1] * 1 + probs[2] * 2 + probs[3] * 3 + probs[4] * 4)
                        
                        player_contributions[player_key]['batted_balls'] += estimated_bases
                        player_contributions[player_key]['total'] += estimated_bases
                    except Exception as e:
                        print(f"Error calculating estimated bases: {e}")
    
    # Sort players by total contribution
    sorted_players = sorted(player_contributions.items(), key=lambda x: x[1]['total'], reverse=True)
    
    # Take top 20 players (or all if less than 20)
    top_players = sorted_players[:20]
    
    if not top_players:
        print("No player contributions found")
        return
    
    # Create figure with high DPI for sharp rendering
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150, constrained_layout=True)
    
    # Prepare data for plotting
    player_labels = []
    batted_balls_values = []
    walks_values = []
    team_colors_list = []
    
    for (player_name, team_name), contributions in top_players:
        # Format player label (shortened first name + last name)
        name_parts = player_name.split()
        if len(name_parts) >= 2:
            formatted_name = f"{name_parts[0][0]}. {' '.join(name_parts[1:])}"
        else:
            formatted_name = player_name
        player_labels.append(formatted_name)
        
        batted_balls_values.append(contributions['batted_balls'])
        walks_values.append(contributions['walks'])
        
        # Get team color
        team_color = team_colors.get(team_name, ['#666666'])[0]
        team_colors_list.append(team_color)
    
    # Create horizontal bars
    y_positions = np.arange(len(player_labels))
    
    # Plot stacked bars with distinct colors
    bars1 = ax.barh(y_positions, batted_balls_values, label='Batted Balls', 
                    color='#2E7D32', edgecolor='black', linewidth=0.5)
    bars2 = ax.barh(y_positions, walks_values, left=batted_balls_values,
                    label='Walks', color='#1976D2', edgecolor='black', linewidth=0.5)
    
    # Add team logos right at the axis edge and player names to the left
    for idx, ((player_name, team_name), _) in enumerate(top_players):
        # Add player name to the left of where logo will be (right-aligned)
        formatted_name = player_labels[idx]
        ax.text(-0.27, idx, formatted_name, ha='right', va='center', 
                fontsize=12, color='black')
        
        # Add team logo closer to the axis
        logo_url = get_team_logo(team_name, mlb_team_logos)
        if logo_url:
            try:
                img = getImage(logo_url, zoom=0.4, size=(30, 30), alpha=1.0)
                if img:
                    # Position logo right at the axis edge
                    ab = AnnotationBbox(img, (-0.075, idx), frameon=False, 
                                      xycoords=('data', 'data'), box_alignment=(1, 0.5))
                    ax.add_artist(ab)
            except Exception as e:
                print(f"Error adding logo for {team_name}: {e}")
    
    # Add value labels on bars (only if value > 0.5)
    for idx, (bb, w) in enumerate(zip(batted_balls_values, walks_values)):
        # Batted balls label
        if bb > 0.5:
            ax.text(bb/2, idx, f'{bb:.1f}', ha='center', va='center', 
                   fontsize=10, color='white', fontweight='bold')
        
        # Walks label
        if w > 0.5:
            ax.text(bb + w/2, idx, f'{w:.0f}', ha='center', va='center',
                   fontsize=10, color='white', fontweight='bold')
        
        # Total at end of bar
        total = bb + w
        ax.text(total + 0.125, idx, f'{total:.1f}', ha='left', va='center',
               fontsize=12, color='black', fontweight='bold')
    
    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])  # Remove y-axis labels since we're adding them manually
    ax.invert_yaxis()  # Highest values at top
    
    # Set labels - REMOVED the ylabel for "Player"
    ax.set_xlabel('Estimated Total Bases', fontsize=14, labelpad=10)
    # ax.set_ylabel('Player', fontsize=14, labelpad=40)  # REMOVED THIS LINE
    
    # Title with game information
    title = f'Player Contributions by Estimated Total Bases\n' \
            f'Actual Score: {away_team} {away_score} - {home_team} {home_score}  ({formatted_date})\n' \
            f'Deserve-to-Win: {away_team} {percentages["away"]}% - {home_team} ' \
            f'{percentages["home"]}%, Tie {percentages["tie"]}%'
    
    ax.set_title(title, fontsize=16, loc='left', pad=15, fontweight='bold')
    
    # Add legend
    ax.legend(loc='lower right', fontsize=11, frameon=True, 
             framealpha=0.9, edgecolor='black')
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # MOVED watermark to top right
    ax.text(0.91, 1.07, 'Data: MLB', transform=ax.transAxes,
           fontsize=13, color='gray', ha='right', va='top')
    ax.text(0.91, 1.05, 'By: @mlb_simulator', transform=ax.transAxes,
           fontsize=13, color='gray', ha='right', va='top')
    
    # Set x-axis limit with some padding
    max_value = max([sum(x) for x in zip(batted_balls_values, walks_values)])
    ax.set_xlim(0, max_value * 1.15)
    
    # Save figure
    os.makedirs(images_dir, exist_ok=True)
    filename = f'{away_team}_{home_team}_{away_score}-{home_score}--{percentages["away"]}-{percentages["home"]}_player_contributions.png'
    plt.savefig(os.path.join(images_dir, filename), bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()


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


def calculate_hit_distance(coord_x, coord_y):
    """
    Calculate actual hit distance from Statcast coordinates.
    
    Returns:
        float: Distance scaled for plotting
    """
    delta_x = coord_x - HOME_PLATE_X
    delta_y = HOME_PLATE_Y - coord_y
    raw_distance = np.sqrt(delta_x**2 + delta_y**2)
    return raw_distance * 0.6


def calculate_expected_bases_for_spray(outcome_dict, pipeline):
    """
    Calculate expected bases for a batted ball in spray chart context.
    
    Returns:
        float: Expected bases (0-4 scale)
    """
    from Simulator.game_simulator import prepare_batted_ball_features
    
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
    except:
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


def get_stadium_fence_curve(venue_name, num_points=100):
    """
    Generate fence curve points for a specific stadium.
    
    Returns:
        tuple: (x_coords, y_coords, dims, smooth_angles, smooth_distances)
    """
    dims = STADIUM_DIMENSIONS.get(venue_name, DEFAULT_STADIUM_DIMENSIONS)
    
    angles_deg = np.array([-45, -22.5, 0, 22.5, 45])
    distances_ft = np.array([dims['LF'], dims['LCF'], dims['CF'], dims['RCF'], dims['RF']])
    distances_plot = distances_ft * FEET_TO_PLOT
    
    interp_func = interp1d(angles_deg, distances_plot, kind='quadratic')
    
    smooth_angles = np.linspace(-45, 45, num_points)
    smooth_distances = interp_func(smooth_angles)
    
    angles_rad = np.radians(90 - smooth_angles)
    x_coords = smooth_distances * np.cos(angles_rad)
    y_coords = smooth_distances * np.sin(angles_rad)
    
    return x_coords, y_coords, dims, smooth_angles, smooth_distances


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
    
    # Infield dirt
    infield_radius = 35
    infield = patches.Wedge(
        center=(0, 0), r=infield_radius,
        theta1=45, theta2=135,
        facecolor=dirt_color, edgecolor='none', alpha=0.5, zorder=2
    )
    ax.add_patch(infield)
    
    # Foul lines (extend past fence for HR visibility)
    max_fence = max(dims.values()) * FEET_TO_PLOT
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
        (-45, dims['LF'], 'right'),
        (0, dims['CF'], 'center'),
        (45, dims['RF'], 'left')
    ]
    
    for angle, dist_ft, ha in label_positions:
        angle_rad = np.radians(90 - angle)
        dist_plot = dist_ft * FEET_TO_PLOT
        label_dist = dist_plot + 6
        label_x = label_dist * np.cos(angle_rad)
        label_y = label_dist * np.sin(angle_rad)
        ax.text(label_x, label_y, f"{dist_ft}'", ha=ha, va='bottom',
                fontsize=9, color='#444444', fontweight='bold', zorder=5)
    
    # Home plate
    home_plate = patches.RegularPolygon(
        (0, 0), numVertices=5, radius=2.5,
        orientation=np.pi, facecolor='white',
        edgecolor='black', linewidth=1, zorder=6
    )
    ax.add_patch(home_plate)


def spray_chart(home_outcomes, away_outcomes, 
                home_team, away_team, 
                home_score, away_score,
                home_win_percentage, away_win_percentage, tie_percentage,
                mlb_team_logos, formatted_date, 
                venue_name='default',
                images_dir="images"):
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
        
    Returns:
        str: Path to saved image file
    """
    import joblib
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.close('all')
    
    pipeline = joblib.load('Model/batted_ball_model.pkl')
    
    percentages = {
        'away': f"{away_win_percentage:.0f}",
        'home': f"{home_win_percentage:.0f}",
        'tie': f"{tie_percentage:.0f}"
    }
    
    home_logo_name = get_logo_team_name(home_team)
    away_logo_name = get_logo_team_name(away_team)
    home_display_name = get_display_team_name(home_team)
    away_display_name = get_display_team_name(away_team)
    
    # Figure size optimized for two side-by-side square-ish fields
    fig, (ax_away, ax_home) = plt.subplots(1, 2, figsize=(15, 7), dpi=150)
    
    # Tight margins
    plt.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.08, wspace=0.02)
    
    # Process outcomes
    batted_balls = {'home': [], 'away': []}
    walk_counts = {'home': 0, 'away': 0}
    
    for team_key, team_outcomes in [('home', home_outcomes), ('away', away_outcomes)]:
        for outcome in team_outcomes:
            outcome_data = outcome[0] if isinstance(outcome, tuple) else outcome
            
            if outcome_data == 'walk':
                walk_counts[team_key] += 1
                continue
            
            if not isinstance(outcome_data, dict):
                continue
            
            coord_x = outcome_data.get('coord_x')
            coord_y = outcome_data.get('coord_y')
            
            if coord_x is None or coord_y is None:
                continue
            
            spray_angle = calculate_spray_angle(coord_x, coord_y)
            distance = calculate_hit_distance(coord_x, coord_y)
            
            outcome_data['venue_name'] = outcome_data.get('venue_name', venue_name)
            xbases = calculate_expected_bases_for_spray(outcome_data, pipeline)
            
            angle_rad = np.radians(90 - spray_angle)
            plot_x = distance * np.cos(angle_rad)
            plot_y = distance * np.sin(angle_rad)
            
            batted_balls[team_key].append({
                'x': plot_x,
                'y': plot_y,
                'xbases': xbases,
            })
    
    # Axis limits based on stadium dimensions
    dims = STADIUM_DIMENSIONS.get(venue_name, DEFAULT_STADIUM_DIMENSIONS)
    max_fence = max(dims.values()) * FEET_TO_PLOT
    axis_limit = max_fence + 12
    
    # Plot each team
    for ax, team_key, display_name, logo_name in [
        (ax_away, 'away', away_display_name, away_logo_name),
        (ax_home, 'home', home_display_name, home_logo_name)
    ]:
        draw_baseball_field(ax, venue_name)
        
        logo_url = get_team_logo(logo_name, mlb_team_logos)
        
        if not logo_url:
            print(f"Warning: No logo found for {display_name} (tried: {logo_name})")
            continue
        
        for bb in batted_balls[team_key]:
            color = get_expected_bases_color(bb['xbases'])
            img = getImage(logo_url, zoom=0.45, size=(40, 40), alpha=0.85)
            
            if img:
                ab = AnnotationBbox(img, (bb['x'], bb['y']), frameon=False, zorder=10)
                ax.add_artist(ab)
                
                ring = plt.Circle(
                    (bb['x'], bb['y']), radius=5.5,
                    fill=False, edgecolor=color,
                    linewidth=2.5, alpha=0.9, zorder=9
                )
                ax.add_patch(ring)
        
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-5, axis_limit)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Team title with stats
        bip_count = len(batted_balls[team_key])
        walk_count = walk_counts[team_key]
        ax.set_title(f"{display_name}\n{bip_count} BIP  •  {walk_count} BB/HBP", 
                     fontsize=14, fontweight='bold', pad=12)
    
    # Main title
    title_text = (
        f"Batted Ball Spray Chart  •  {venue_name}\n"
        f"Actual: {away_display_name} {away_score} - {home_display_name} {home_score}  ({formatted_date})    "
        f"Deserve-to-Win: {away_display_name} {percentages['away']}% - "
        f"{home_display_name} {percentages['home']}%, Tie {percentages['tie']}%"
    )
    fig.suptitle(title_text, fontsize=13, fontweight='bold', y=0.94)
    
    # Attribution (centered under title)
    fig.text(0.5, 0.86, 'Data: MLB  |  @mlb_simulator', 
             fontsize=10, color='gray', ha='center', va='top')
    
    # Disclaimer about estimated dimensions (just below attribution)
    fig.text(0.5, 0.83, '* Ballpark dimensions estimated using LF/CF/RF fence distances', 
             fontsize=8, color='gray', ha='center', va='top', style='italic')
    
    # Legend (bottom center)
    legend_items = [
        ('Out', '#808080'),
        ('Single', '#FFA500'),
        ('XBH', '#FF6347'),
        ('HR', '#DC143C')
    ]
    
    fig.text(0.35, 0.025, "Expected Outcome:", ha='right', fontsize=10, fontweight='bold', va='center')
    
    for i, (label, color) in enumerate(legend_items):
        x_pos = 0.37 + (i * 0.085)
        circle = plt.Circle((x_pos, 0.025), 0.01, 
                            transform=fig.transFigure, 
                            color=color, zorder=100)
        fig.patches.append(circle)
        fig.text(x_pos + 0.015, 0.025, label, ha='left', fontsize=10, va='center')
    
    # Save
    os.makedirs(images_dir, exist_ok=True)
    filename = (f"{away_display_name}_{home_display_name}_{away_score}-{home_score}--"
                f"{percentages['away']}-{percentages['home']}_spray.png")
    filepath = os.path.join(images_dir, filename)
    
    plt.savefig(filepath, dpi=250, facecolor='white', edgecolor='none',
                bbox_inches='tight', pad_inches=0.05)
    
    print(f"Saved spray chart: {filepath}")
    plt.close(fig)
    
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
