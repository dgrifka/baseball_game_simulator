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

def getImage(path, zoom=0.39, size=(50, 50), alpha=0.65, image_cache={}):
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
    Creates launch angle vs exit velocity visualization with team performance overlay.
    
    Args:
        home_outcomes/away_outcomes (list): Team batting outcomes
        away/home_estimated_total_bases (list): Estimated bases for each team
        home/away_team (str): Team names
        home/away_score (int): Actual game scores
        home/away/tie_win_percentage (float): Win percentages from simulation
        mlb_team_logos (list): Team logo URLs
        formatted_date (str): Formatted date string to display
        images_dir (str): Output directory for saved visualization
    """
    # Clear any existing figures
    plt.close('all')
    
    percentages = {
        'away': f"{away_win_percentage:.0f}",
        'home': f"{home_win_percentage:.0f}",
        'tie': f"{tie_percentage:.0f}"
    }
    
    # Extract launch angles and exit velocities
    outcomes = {
        'home': {'ev': [], 'la': [], 'walks': home_outcomes.count('walk')},
        'away': {'ev': [], 'la': [], 'walks': away_outcomes.count('walk')}
    }
    
    for team, team_outcomes in [('home', home_outcomes), ('away', away_outcomes)]:
        outcomes[team]['ev'] = [o[0] for o in team_outcomes if isinstance(o, list)]
        outcomes[team]['la'] = [o[1] for o in team_outcomes if isinstance(o, list)]

    # Create a new figure
    fig = plt.figure(figsize=(10, 6))

    # Load and process contour data
    contour_data = pd.read_csv('Data/contour_data.csv').dropna()
    x, y, z = contour_data['x'].values, contour_data['y'].values, contour_data['z'].values
    
    # Create interpolation grid
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='linear', fill_value=0)
    
    # Define custom colormap
    colors_list = ["white", "#E6E6E6", "#CCCCCC", "#B3B3B3", "#999999", "#808080", "#666666", "#4A4A4A"]
    levels = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    cmap = colors.LinearSegmentedColormap.from_list("custom", colors_list, N=len(levels)-1)
    
    # Plot contours
    Z = np.clip(Z, 0.5, 4)
    contour = plt.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.7, extend='both')
    plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.25, alpha=0.1)
    
    # Add colorbar
    cbar = plt.colorbar(contour, label='Average Total Bases', ticks=levels)
    cbar.set_ticklabels([f'{level:.1f}' for level in levels])

    # Plot team data
    for team, data in outcomes.items():
        team_name = home_team if team == 'home' else away_team
        logo_url = get_team_logo(team_name, mlb_team_logos)
        
        if logo_url:
            for x, y in zip(data['ev'], data['la']):
                # Create a new image instance for each point
                img = getImage(logo_url, alpha=0.765 if team == 'home' else 0.75)
                if img:
                    # Create a new AnnotationBbox instance for each point
                    ab = AnnotationBbox(img, (x, y), frameon=False)
                    plt.gca().add_artist(ab)
        else:
            plt.scatter(data['ev'], data['la'], s=175, alpha=0.85, 
                       label=team_name, color='blue' if team == 'home' else 'red',
                       marker='o' if team == 'home' else '^')

    # Rest of the formatting code
    plt.axhline(y=0, color='black', alpha=0.8, linewidth=0.8)
    
    plt.text(0.05, 0.95, 'Walks/HBP', transform=plt.gca().transAxes, 
             fontsize=16, verticalalignment='top')
    plt.text(0.05, 0.947, '___________', transform=plt.gca().transAxes, 
             fontsize=14, verticalalignment='top')
    plt.text(0.05, 0.90, f'{away_team}: {outcomes["away"]["walks"]}', 
             transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')
    plt.text(0.05, 0.85, f'{home_team}: {outcomes["home"]["walks"]}', 
             transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')

    plt.text(-.06, -.1, 'Data: MLB', transform=plt.gca().transAxes, 
             fontsize=8, color='black', ha='left', va='bottom')
    plt.text(-.06, -.122, 'By: @mlb_simulator', transform=plt.gca().transAxes, 
             fontsize=8, color='black', ha='left', va='bottom')

    plt.xlabel('Exit Velocity (mph)', fontsize=18)
    plt.ylabel('Launch Angle', fontsize=18)
    plt.title(f'Batted Ball Exit Velo / Launch Angle by Team\n'
              f'Actual Score:     {away_team} {str(away_score)} - {home_team} {str(home_score)}  ({formatted_date})\n'
              f'Deserve-to-Win: {away_team} {percentages["away"]}%, {home_team} '
              f'{percentages["home"]}%, Tie {percentages["tie"]}%', 
              fontsize=16, loc='left', pad=12)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%d°'))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Save and close the figure
    os.makedirs(images_dir, exist_ok=True)
    filename = f'{away_team}_{home_team}_{str(away_score)}-{str(home_score)}--{percentages["away"]}-{percentages["home"]}_bb.png'
    plt.savefig(os.path.join(images_dir, filename), bbox_inches='tight')
    plt.close(fig)  # Close the specific figure

def run_dist(num_simulations, home_runs_scored, away_runs_scored, home_team, away_team,
             home_score, away_score, home_win_percentage, away_win_percentage, tie_percentage, 
             formatted_date, images_dir="images"):
    """
    Creates visualization of run distribution from simulations.
    
    Args:
        num_simulations (int): Number of simulations run
        home/away_runs_scored (array): Simulated runs for each team
        home/away_team (str): Team names
        home/away_score (int): Actual game scores
        home/away/tie_win_percentage (float): Win percentages from simulation
        formatted_date (str): Formatted date string to display
        images_dir (str): Output directory for saved visualization
    """
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
    
    def format_mode(mode_result):
        if hasattr(mode_result, 'mode'):
            values = mode_result.mode.flatten()
        elif isinstance(mode_result, np.ndarray):
            values = mode_result.flatten()
        else:
            values = [mode_result]
        return ','.join(map(str, values))
    
    mode_strs = {team: format_mode(mode) for team, mode in modes.items()}
    
    plt.figure(figsize=(10, 6))
    
    # Create histograms
    max_runs = max(max(home_runs_scored), max(away_runs_scored))
    bins = range(max_runs + 2)
    
    for runs, team, pattern in [(home_runs_scored, home_team, '/'), 
                               (away_runs_scored, away_team, '\\')]:
        plt.hist(runs, bins=bins, alpha=0.6, label=team,
                color=team_colors[team][0], edgecolor='black',
                linewidth=1, hatch=pattern)
    
    # Add labels and formatting
    plt.xlabel('Runs Scored', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    plt.title(f'Distribution of Runs Scored ({num_simulations} Simulations)\n'
              f'Actual Score: {away_team} {away_score} - {home_team} {home_score}\n'
              f'Deserve-to-Win: {away_team} {percentages["away"]}% - {home_team} '
              f'{percentages["home"]}%, Tie {percentages["tie"]}%\n'
              f'Most Likely Outcome: {away_team} {mode_strs["away"]} - {home_team} '
              f'{mode_strs["home"]}',
              fontsize=16, loc='left', pad=20)
    
    # Add metadata
    plt.text(-.05, -.09, 'Data: MLB', transform=plt.gca().transAxes,
             fontsize=8, color='black', ha='left', va='bottom')
    plt.text(-.05, -.11, 'By: @mlb_simulator', transform=plt.gca().transAxes,
             fontsize=8, color='black', ha='left', va='bottom')
    
    # Format ticks and legend
    plt.xticks(range(max_runs + 2), fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Save visualization
    os.makedirs(images_dir, exist_ok=True)
    filename = f'{away_team}_{home_team}_{str(away_score)}-{str(home_score)}--{percentages["away"]}-{percentages["home"]}_rd.png'
    plt.savefig(os.path.join(images_dir, filename), bbox_inches='tight')
    plt.close()


def create_estimated_bases_table(df, away_team, home_team, away_score, home_score,
                               away_win_percentage, home_win_percentage, formatted_date, images_dir):
    """
    Creates formatted table visualization of estimated bases statistics.
    
    Args:
        df (pd.DataFrame): Estimated bases data
        away/home_team (str): Team names
        away/home_score (int): Actual game scores
        away/home_win_percentage (float): Win percentages from simulation
        formatted_date (str): Formatted date string to display
        images_dir (str): Output directory for saved visualization
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.axis('off')
    
    # Preprocess DataFrame
    df = df.copy()
    df['Player'] = df['Player'].str.replace(' ', '\n', n=1)
    df['Result'] = df['Result'].str.replace('_', ' ')
    
    # Create color mappings
    team_color_map = dict(zip(df['Team'], df['team_color']))
    df = df.drop('team_color', axis=1)
    df.columns = df.columns.str.replace(' ', '\n')
    
    # Create table
    col_width = 1.01 / len(df.columns)
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    loc='center',
                    cellLoc='center',
                    colWidths=[col_width] * len(df.columns))
    
    # Format cells
    for (row, col), cell in table.get_celld().items():
        cell.set_height(0.09)
        if row == 0:
            cell.set_text_props(weight='bold', fontsize=35)
        else:
            cell.set_text_props(fontsize=30)
    
    def is_dark_color(color):
        r, g, b = to_rgb(color)
        return (r * 0.299 + g * 0.587 + b * 0.114) < 0.5
    
    # Apply color formatting
    team_col = df.columns.get_loc('Team')
    bases_col = df.columns.get_loc('Estimated\nBases')
    result_col = df.columns.get_loc('Result')
    
    # Team colors
    for row in range(1, len(df) + 1):
        team = df.iloc[row-1]['Team']
        cell = table[(row, team_col)]
        color = team_color_map[team]
        cell.set_facecolor(color)
        if is_dark_color(color):
            cell.get_text().set_color('white')
    
    # Estimated bases heat map
    values = df['Estimated\nBases'].values
    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(min(values), max(values))
    colors = [cmap(norm(v)) for v in values]
    
    for row in range(1, len(df) + 1):
        cell = table[(row, bases_col)]
        cell.set_facecolor(colors[row - 1])
    
    # Result highlighting
    for row in range(1, len(df) + 1):
        if df.iloc[row-1]['Result'] == 'Out':
            cell = table[(row, result_col)]
            cell.set_facecolor('red')
            cell.set_alpha(0.25)
    
    # Add title and metadata
    fig.text(0.5, .9725, 'Data: MLB    By: @mlb_simulator', 
             fontsize=14, color='black', ha='center', va='center')
    
    plt.title(f'Top 10 Estimated Bases\n'
              f'Actual Score: {away_team} {away_score} - {home_team} {home_score}  ({formatted_date})\n'
              f'Deserve-to-Win %: {away_team} {away_win_percentage:.0f}% - '
              f'{home_team} {home_win_percentage:.0f}%',
              fontsize=17.5, loc='left', y=1.03)
    
    # Save visualization
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    
    os.makedirs(images_dir, exist_ok=True)
    filename = f'{away_team}_{home_team}_{away_score}-{home_score}--{away_win_percentage:.0f}-{home_win_percentage:.0f}_estimated_bases.png'
    plt.savefig(os.path.join(images_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
