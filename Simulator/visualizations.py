import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import colorsys
import matplotlib.colors as colors
import pandas as pd
import numpy as np
from scipy import stats

from Simulator.constants import team_colors
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

def getImage(path, zoom=0.45, size=(50, 50), alpha=0.625, image_cache={}):
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
                img = getImage(logo_url, zoom=0.5 if team == 'home' else 0.465,
                             alpha=0.825 if team == 'home' else 0.795)
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
    plt.savefig(os.path.join(images_dir, filename), bbox_inches='tight', dpi=300)
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
    """Prepare and format data for the table display."""
    df = df.copy()
    
    # Add rank column
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    # Format player names - keep on single line for cleaner look
    df['Player'] = df['Player'].apply(lambda x: x.split()[0][0] + '. ' + ' '.join(x.split()[1:]))
    
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
    
    # Select and reorder columns for clarity
    columns_to_keep = ['Rank', 'Team', 'Player', 'Launch Speed', 'Launch Angle', 
                       'Result', 'Estimated Bases', 'xBA', 'HR%']
    df = df[columns_to_keep]
    
    # Format launch angle to include degree symbol
    df['Launch Angle'] = df['Launch Angle'].astype(str) + '°'
    
    # Format launch speed
    df['Launch Speed'] = df['Launch Speed'].astype(str) + ' mph'
    
    # Round estimated bases to 2 decimals
    df['Estimated Bases'] = df['Estimated Bases'].round(2)
    
    return df

def create_enhanced_cell_styles(table, df, team_color_map):
    """Apply enhanced styling to table cells."""
    
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
            cell.set_height(0.06)  # Reduced from 0.08
            cell.set_text_props(weight='bold', fontsize=14)
            cell.set_facecolor('#2C3E50')
            cell.get_text().set_color('white')
            cell.set_edgecolor('#1A252F')
            cell.set_linewidth(2)
        else:  # Data rows
            cell.set_height(0.055)  # Reduced from 0.075
            cell.set_text_props(fontsize=13)
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
        
        # Team colors
        team = df.iloc[row-1]['Team']
        team_cell = table[(row, team_col)]
        color = team_color_map[team]
        team_cell.set_facecolor(color)
        if is_dark_color(color):
            team_cell.get_text().set_color('white')
        team_cell.get_text().set_weight('bold')
        
        # Player names - slightly larger
        player_cell = table[(row, player_col)]
        player_cell.get_text().set_fontsize(13.5)
        
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


def create_estimated_bases_table(df, away_team, home_team, away_score, home_score,
                               away_win_percentage, home_win_percentage, formatted_date, images_dir):
    """Creates enhanced table visualization of estimated bases statistics."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.close('all')
    
    # Prepare data - take top 15 instead of top 10
    df = df.copy().head(15)
    team_color_map = dict(zip(df['Team'], df['team_color']))
    df = prepare_table_data(df)
    
    # Create figure with wider, less tall proportions
    fig = plt.figure(figsize=(20, 10), dpi=150)
    
    # Add subplot with more space at top for titles
    # Reduced height from 0.72 to 0.65 and moved down from 0.08 to 0.05
    ax = fig.add_subplot(111)
    ax.set_position([0.05, 0.05, 0.9, 0.65])  # [left, bottom, width, height]
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.06, 0.08, 0.15, 0.12, 0.12, 0.10, 0.14, 0.08, 0.08])
    
    # Apply enhanced styling
    create_enhanced_cell_styles(table, df, team_color_map)
    
    # Scale table with adjusted scaling
    table.auto_set_font_size(False)
    table.scale(1.1, 1.5)  # Reduced height scaling from 2.0 to 1.5
    
    # Enhanced title with better spacing - moved all text elements up
    title_lines = [
        f"Top 15 Batted Balls by Estimated Total Bases",
        f"{away_team} {away_score} - {home_team} {home_score}  •  {formatted_date}",
        f"Win Probability: {away_team} {away_win_percentage:.0f}% - {home_team} {home_win_percentage:.0f}%"
    ]
    
    # Main title - moved higher to 0.94
    plt.text(0.5, 0.94, title_lines[0], transform=fig.transFigure,
             fontsize=22, fontweight='bold', ha='center', va='top')
    
    # Subtitle lines - moved higher
    plt.text(0.5, 0.88, title_lines[1], transform=fig.transFigure,
             fontsize=16, ha='center', va='top', color='#333333')
    
    plt.text(0.5, 0.84, title_lines[2], transform=fig.transFigure,
             fontsize=14, ha='center', va='top', color='#666666')
    
    # Attribution - back to top left corner
    plt.text(0.02, 0.98, 'Data: MLB', 
             transform=fig.transFigure, fontsize=12, 
             ha='left', va='top', color='#999999')
    
    plt.text(0.075, 0.92, 'By: @mlb_simulator', 
             transform=fig.transFigure, fontsize=12, 
             ha='left', va='top', color='#999999')
    
    # Save with high quality
    os.makedirs(images_dir, exist_ok=True)
    filename = f'{away_team}_{home_team}_{away_score}-{home_score}--{away_win_percentage:.0f}-{home_win_percentage:.0f}_estimated_bases.png'
    plt.savefig(os.path.join(images_dir, filename), 
                bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
