import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.interpolate import griddata
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
    home_mode = stats.mode(home_runs_scored)
    away_mode = stats.mode(away_runs_scored)
    
    # Convert to string, handling multiple modes
    def mode_to_str(mode_result):
        if hasattr(mode_result, 'mode'):
            return ', '.join(map(str, mode_result.mode))
        else:
            return ', '.join(map(str, mode_result[0]))

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
             f'Actual Score:     {away_team} {str(away_score)} - {home_team} {str(home_score)}\n'
             f'Deserve-to-Win: {away_team} {str(away_win_percentage_str)}%, {home_team} {str(home_win_percentage_str)}%, Tie {tie_percentage_str}%\n'
             f'Most Likely Outcomes: {away_team} {away_mode_str}, {home_team} {home_mode_str}')
    
    plt.title(title, fontsize=16, loc='left', pad=20)  # Increased pad for more space
    
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


def create_estimated_bases_graph(df, title, away_team, home_team, away_score, home_score, away_win_percentage, home_win_percentage, images_dir):
    # Create a figure with two subplots (one for table, one for graph)
    fig, (ax_table, ax_graph) = plt.subplots(2, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [1, 2]})

    away_win_percentage_str = f"{away_win_percentage:.0f}"
    home_win_percentage_str = f"{home_win_percentage:.0f}"

    # Create the table
    table_data = df.drop(columns=['team_color']).copy()
    table_data['Estimated Bases'] = table_data['Estimated Bases'].round(2)  # Round to 2 decimal places
    
    # Rename columns to include line breaks
    column_names = [
        'Team', 'Player', 'Launch\nSpeed', 'Launch\nAngle', 'Result', 'Estimated\nBases',
        'Out\nProb', 'Single\nProb', 'Double\nProb', 'Triple\nProb', 'Hr\nProb'
    ]
    
    # Set column widths
    col_widths = [0.1, 0.15, 0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.08, 0.08, 0.08]
    
    table = ax_table.table(cellText=table_data.values, colLabels=column_names, cellLoc='center', loc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Increased font size
    table.scale(1, 1.5)  # Adjust table size
    
    ax_table.axis('off')  # Hide axis for the table subplot

    # Create a dictionary to map teams to hatch patterns
    hatch_patterns = ['/', '\\', 'x', '+', '.', 'o', '*', '-']
    team_hatches = {team: hatch_patterns[i % len(hatch_patterns)] for i, team in enumerate(df['Team'].unique())}
    
    # Create the horizontal bar plot
    bars = ax_graph.barh(range(len(df)), df['Estimated Bases'], color=df['team_color'], alpha=0.7, edgecolor='black', linewidth=1)
    
    # Apply hatching to each bar based on team
    for bar, team in zip(bars, df['Team']):
        bar.set_hatch(team_hatches[team])
        bar.set_edgecolor('black')  # Ensure the hatch is visible

    # Customize the plot
    ax_graph.set_xlabel('Estimated Bases', fontsize=19)
    fig.suptitle(f'{title}', fontsize=22, y=0.91)  # Move title up slightly
    
    # Create labels for y-axis
    y_labels = [f"{player}\n({result})" for player, result in zip(df['Player'], df['Result'])]
    
    # Set y-axis ticks and labels
    ax_graph.set_yticks(range(len(df)))
    ax_graph.set_yticklabels(y_labels, fontsize=13)

    # Add value labels at the end of each bar
    for i, v in enumerate(df['Estimated Bases']):
        ax_graph.text(v, i, f' {v:.2f}', va='center', fontsize=18)

    # Invert y-axis to show highest value at the top
    ax_graph.invert_yaxis()

    # Remove x-axis tick marks and labels
    ax_graph.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Remove top and right spines
    ax_graph.spines['top'].set_visible(False)
    ax_graph.spines['right'].set_visible(False)

    # Add a legend with hatching
    teams = df['Team'].unique()
    handles = [plt.Rectangle((0,0),1,1, facecolor=team_colors[team][0], alpha=0.8, edgecolor='black', linewidth=1, hatch=team_hatches[team]) for team in teams]
    ax_graph.legend(handles, teams, fontsize=18)

    # Adjust layout
    plt.tight_layout()
    
    # Adjust the spacing between subplots and title
    plt.subplots_adjust(top=0.95, hspace=-0.075)

    if title == "Lucky Hits (Using Estimated Bases)":
      title_save = "lh"
    elif title == "Unlucky Outs (Using Estimated Bases)":
      title_save = "uo"
    # Save the figure
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    plt.savefig(os.path.join(images_dir, f'{away_team}_{home_team}_{str(away_score)}-{str(home_score)}--{str(away_win_percentage_str)}-{str(home_win_percentage_str)}_{title_save}.png'), bbox_inches='tight')
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

    # Save the plot to the "images" folder in the repository
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    plt.savefig(os.path.join(images_dir, f'{away_team}_{home_team}_{str(away_score)}-{str(home_score)}--{str(away_win_percentage_str)}-{str(home_win_percentage_str)}_tb.png'), bbox_inches='tight')
    plt.close()
