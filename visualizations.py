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

from constants import team_colors

import os


def launch_angle_range(exit_velocity):
    if exit_velocity < 98:
        return None
    elif exit_velocity == 98:
        return 26, 30
    else:
        min_launch_angle = 26 - (exit_velocity - 98) * (2 if exit_velocity <= 116 else 3)
        max_launch_angle = 30 + (exit_velocity - 98) * (2 if exit_velocity <= 116 else 3)
        return min_launch_angle, max_launch_angle
    
def la_ev_graph(home_outcomes, away_outcomes, away_estimated_total_bases, home_estimated_total_bases, home_team, away_team, home_score, away_score, home_win_percentage, away_win_percentage, tie_percentage, images_dir = "images"):
    away_win_percentage_str = f"{away_win_percentage:.0f}"
    home_win_percentage_str = f"{home_win_percentage:.0f}"
    tie_percentage_str = f"{tie_percentage:.0f}"

    home_ev = [outcome[0] for outcome in home_outcomes if isinstance(outcome, list)]
    home_la = [outcome[1] for outcome in home_outcomes if isinstance(outcome, list)]
    away_ev = [outcome[0] for outcome in away_outcomes if isinstance(outcome, list)]
    away_la = [outcome[1] for outcome in away_outcomes if isinstance(outcome, list)]

    home_walks = home_outcomes.count('walk')
    away_walks = away_outcomes.count('walk')

    plt.figure(figsize=(10, 6))

    # Load and plot the contour data
    contour_data = pd.read_csv('contour_data.csv')
    contour_data = contour_data.dropna()
    
    # Extract x, y, z values
    x = contour_data['x'].values
    y = contour_data['y'].values
    z = contour_data['z'].values
    
    # Create a grid for interpolation
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)
    
    # Interpolate the data
    Z = griddata((x, y), z, (X, Y), method='linear', fill_value=0)
    
    # Create a custom colormap with evenly spaced colors
    colors_list = ["white", "#E6E6E6", "#CCCCCC", "#B3B3B3", "#999999", "#808080", "#666666", "#4A4A4A"]
    levels = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    n_bins = len(levels) - 1  # number of color bins
    cmap = colors.LinearSegmentedColormap.from_list("custom", colors_list, N=n_bins)

    # Clip the data to the desired range
    Z = np.clip(Z, 0.5, 4)
    
    # Plot filled contours
    contour = plt.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.7, extend='both')

    # Add contour lines (optional)
    line_contour = plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.25, alpha=0.1)
    
    # Add colorbar with specific ticks
    cbar = plt.colorbar(contour, label='Average Total Bases', ticks=levels)
    cbar.set_ticklabels([f'{level:.1f}' for level in levels])  # Format tick labels to one decimal place
    
    plt.scatter(home_ev, home_la, s=175, alpha=0.65, label=f'{home_team}', color=team_colors[home_team][0], marker='o')
    plt.scatter(away_ev, away_la, s=175, alpha=0.65, label=f'{away_team}', color=team_colors[away_team][0], marker="^")
    plt.axhline(y=0, color='black', alpha = 0.8, linewidth=0.8)

    plt.text(0.05, 0.95, 'Walks/HBP', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
    plt.text(0.05, 0.947, '___________', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    plt.text(0.05, 0.90, f'{away_team}: {away_walks}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')
    plt.text(0.05, 0.85, f'{home_team}: {home_walks}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')

    ## Add watermark
    plt.text(-.06, -.1, 'Data: MLB', transform=plt.gca().transAxes, fontsize=7, color='darkgray', ha='left', va='bottom')
    plt.text(-.06, -.122, 'By: @mlb_simulator', transform=plt.gca().transAxes, fontsize=7, color='darkgray', ha='left', va='bottom')
    # plt.text(0.05, 0.75, 'Estimated Total Bases', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    # plt.text(0.05, 0.745, '_____________________', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    # plt.text(0.05, 0.70, f'{away_team}: {away_estimated_total_bases}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    # plt.text(0.05, 0.65, f'{home_team}: {home_estimated_total_bases}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.xlabel('Exit Velocity (mph)', fontsize=18)
    plt.ylabel('Launch Angle', fontsize=18)
    plt.title(f'Batted Ball Exit Velo / Launch Angle by Team\nActual Score:     {away_team} {str(away_score)} - {home_team} {str(home_score)}\nDeserve-to-Win: {away_team} {str(away_win_percentage_str)}%, {home_team} {str(home_win_percentage_str)}%, Tie {tie_percentage_str}%', fontsize=16, loc = 'left', pad=12)

    plt.legend(loc='upper right')  # Add a legend in the upper right corner

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # Format y-tick labels as degrees
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%d°'))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Save the plot to the "images" folder in the repository
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    plt.savefig(os.path.join(images_dir, f'{away_team}_{home_team}_{str(away_score)}-{str(home_score)}--{str(away_win_percentage_str)}-{str(home_win_percentage_str)}_bb.png'), bbox_inches='tight')
    plt.close()


def run_dist(num_simulations, home_runs_scored, away_runs_scored, home_team, away_team, home_score, away_score, home_win_percentage, away_win_percentage, tie_percentage, images_dir = "images"):
    away_win_percentage_str = f"{away_win_percentage:.0f}"
    home_win_percentage_str = f"{home_win_percentage:.0f}"
    tie_percentage_str = f"{tie_percentage:.0f}"
    
    # Calculate modes
    home_mode = stats.mode(home_runs_scored, keepdims=True)
    away_mode = stats.mode(away_runs_scored, keepdims=True)
    
    # Convert to string, handling multiple modes and single values
    def mode_to_str(mode_result):
        if hasattr(mode_result, 'mode'):
            mode_values = mode_result.mode.flatten()
        elif isinstance(mode_result, np.ndarray):
            mode_values = mode_result.flatten()
        else:
            mode_values = [mode_result]
        
        return ','.join(map(str, mode_values))

    home_mode_str = mode_to_str(home_mode)
    away_mode_str = mode_to_str(away_mode)
    
    # Graph the distributions of runs scored
    plt.figure(figsize=(10, 6))
    max_runs = max(max(home_runs_scored), max(away_runs_scored))
    bins = range(0, max_runs + 2)  # Start from 0 and include the maximum runs scored
    
    # Home team histogram with hatching
    plt.hist(home_runs_scored, bins=bins, alpha=0.6, label=f'{home_team}', 
             color=team_colors[home_team][0], edgecolor='black', linewidth=1, hatch='/')
    
    # Away team histogram with different hatching
    plt.hist(away_runs_scored, bins=bins, alpha=0.6, label=f'{away_team}', 
             color=team_colors[away_team][0], edgecolor='black', linewidth=1, hatch='\\')
    
    plt.xlabel('Runs Scored', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    # Multi-line title with Most Likely Outcomes
    title = (f'Distribution of Runs Scored ({num_simulations} Simulations)\n'
             f'Actual Score: {away_team} {str(away_score)} - {home_team} {str(home_score)}\n'
             f'Deserve-to-Win: {away_team} {str(away_win_percentage_str)}% - {home_team} {str(home_win_percentage_str)}%, Tie {tie_percentage_str}%\n'
             f'Most Likely Outcome: {away_team} {away_mode_str} - {home_team} {home_mode_str}')
    
    plt.title(title, fontsize=16, loc='left', pad=20)  # Increased pad for more space

    ## Add watermark
    plt.text(-.05, -.09, 'Data: MLB', transform=plt.gca().transAxes, fontsize=7, color='darkgray', ha='left', va='bottom')
    plt.text(-.05, -.11, 'By: @mlb_simulator', transform=plt.gca().transAxes, fontsize=7, color='darkgray', ha='left', va='bottom')
    
    # Set integer x-axis ticks
    x_ticks = range(0, max_runs + 2)
    plt.xticks(x_ticks, fontsize=12)
    
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Save the plot to the "images" folder in the repository
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    plt.savefig(os.path.join(images_dir, f'{away_team}_{home_team}_{str(away_score)}-{str(home_score)}--{str(away_win_percentage_str)}-{str(home_win_percentage_str)}_rd.png'), bbox_inches='tight')
    plt.close()
    
def create_estimated_bases_table(df, away_team, home_team, away_score, home_score, away_win_percentage, home_win_percentage, images_dir):
    # Create a new figure and axis, ensuring it's clear of any previous content
    fig, ax = plt.subplots(figsize=(14, 6))  # Reduced figure height
    
    # Clear the axis completely
    ax.clear()
    
    # Hide axis
    ax.axis('off')
    
    # Replace first space with newline in Player names and "_" with " " in Result
    df['Player'] = df['Player'].str.replace(' ', '\n', n=1)
    df['Result'] = df['Result'].str.replace('_', ' ')
    
    # Create color mapping for teams
    team_colors = dict(zip(df['Team'], df['team_color']))
    
    # Drop the team_color column
    df = df.drop('team_color', axis=1)
    
    # Replace spaces with newlines in column names
    df.columns = df.columns.str.replace(' ', '\n')
    
    # Create the table with equal column widths
    n_cols = len(df.columns)
    col_width = 0.925 / n_cols
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center',
                     cellLoc='center',
                     colWidths=[col_width] * n_cols)  # Set uniform column widths

    # Set font size and style for column labels and cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', fontsize=50)  # Increased font size for headers
        else:
            cell.set_text_props(fontsize=47)  # Increased font size for cell content
        cell.set_height(0.08)
    
    # Function to determine if color is dark
    def is_dark(color):
        r, g, b = to_rgb(color)
        return (r * 0.299 + g * 0.587 + b * 0.114) < 0.5

    # Function to apply continuous color gradient for Estimated Bases
    def color_scale(values, alpha=0.50):
        cmap = plt.cm.get_cmap('YlOrRd')
        norm = plt.Normalize(min(values), max(values))
        colors = [cmap(norm(value)) for value in values]
        return [(r, g, b, alpha) for r, g, b, _ in colors]
    
    # Apply formatting to Team column
    team_col_index = df.columns.get_loc('Team')
    for row in range(1, len(df) + 1):
        team = df.iloc[row-1]['Team']
        cell = table[(row, team_col_index)]
        cell.set_facecolor(team_colors[team])
        if is_dark(team_colors[team]):
            cell.get_text().set_color('white')
    
    # Apply formatting to Estimated Bases column
    col_index = df.columns.get_loc('Estimated\nBases')
    column_values = df['Estimated\nBases'].values
    colors = color_scale(column_values)
    for row in range(1, len(df) + 1):
        cell = table[(row, col_index)]
        cell.set_facecolor(colors[row - 1])
    
    # Apply formatting to Result column
    result_col_index = df.columns.get_loc('Result')
    for row in range(1, len(df) + 1):
        result = df.iloc[row-1]['Result']
        cell = table[(row, result_col_index)]
        if result == 'Out':
            cell.set_facecolor('red')
            cell.set_alpha(0.25)
    
    # Remove any existing texts or other elements
    for text in ax.texts:
        text.remove()
    
    # Clear collections and patches
    while ax.collections:
        ax.collections[0].remove()
    while ax.patches:
        ax.patches[0].remove()
    
    # Add watermark above the table
    fig.text(0.5, 0.95, 'Data: MLB    By: @mlb_simulator', fontsize=16, color='darkgray', ha='center', va='center')
    
    # Set combined title above the watermark, aligned to the left
    plt.title(f'Top 15 Estimated Bases\n'
              f'Actual Score: {away_team} {away_score} - {home_team} {home_score}\n'
              f'Deserve-to-Win %: {away_team} {away_win_percentage:.0f}% - {home_team} {home_win_percentage:.0f}%', 
              fontsize=18, loc='left', y=1.01)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    plt.savefig(os.path.join(images_dir, f'{away_team}_{home_team}_{away_score}-{home_score}--{away_win_percentage:.0f}-{home_win_percentage:.0f}_estimated_bases.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def tb_barplot(home_estimated_total_bases, away_estimated_total_bases, home_win_percentage, away_win_percentage, tie_percentage, home_team, away_team, home_score, away_score, images_dir = "images"):

    # Create a bar plot for estimated total bases
    labels = [f'{away_team}', f'{home_team}']
    total_bases = [away_estimated_total_bases, home_estimated_total_bases]
    colors = [team_colors[away_team][0], team_colors[home_team][0]]  # Use hex codes for bar colors
    away_win_percentage_str = f"{away_win_percentage:.0f}"
    home_win_percentage_str = f"{home_win_percentage:.0f}"
    tie_percentage_str = f"{tie_percentage:.0f}"

    plt.figure(figsize=(8, 6))
    plt.bar(labels, total_bases, color=colors)
    plt.xlabel('Team', fontsize=14)
    plt.ylabel('Estimated Total Bases', fontsize=14)
    plt.title(f'Estimated Total Bases using Hit Probabilities\nActual Score:     {away_team} {str(away_score)} - {home_team} {str(home_score)}\nDeserve-to-Win: {away_team} {str(away_win_percentage_str)}%, {home_team} {str(home_win_percentage_str)}%, Tie {tie_percentage_str}%', fontsize=16, loc = 'left', pad=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().set_yticklabels([f'{x:.1f}' for x in plt.gca().get_yticks()])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    ## Add watermark
    plt.text(0.5, 0.01, '@mlb_simulator', transform=plt.gca().transAxes, fontsize=8, color='darkgray', ha='center', va='bottom')
    
    # Save the plot to the "images" folder in the repository
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    plt.savefig(os.path.join(images_dir, f'{away_team}_{home_team}_{str(away_score)}-{str(home_score)}--{str(away_win_percentage_str)}-{str(home_win_percentage_str)}_tb.png'), bbox_inches='tight')
    plt.close()
