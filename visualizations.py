import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle

import numpy as np

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

    plt.scatter(home_ev, home_la, s=150, alpha=0.6, label=f'{home_team}', color=team_colors[home_team][0], marker='o')
    plt.scatter(away_ev, away_la, s=150, alpha=0.6, label=f'{away_team}', color=team_colors[away_team][0], marker="^")
    plt.axhline(y=0, color='black', alpha = 0.8, linewidth=0.8)

    plt.text(0.05, 0.95, 'Walks/HBP', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
    plt.text(0.05, 0.947, '___________', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    plt.text(0.05, 0.90, f'{away_team}: {away_walks}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')
    plt.text(0.05, 0.85, f'{home_team}: {home_walks}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')

    x = np.linspace(98, 123, 100)  # Exit velocity range
    y1 = np.zeros_like(x)
    y2 = np.zeros_like(x)

    for i, ev in enumerate(x):
        angle_range = launch_angle_range(ev)
        if angle_range:
            y1[i], y2[i] = angle_range

    # Shade the area between the launch angle ranges
    plt.fill_between(x, y1, y2, where=(y1 > 0) & (y2 > 0), color='green', alpha=0.15)

    # Calculate the position of the label relative to the right middle of the green fill between
    label_ev = 114  # Specify the exit velocity at which to position the label
    label_la = (launch_angle_range(label_ev)[0] + launch_angle_range(label_ev)[1]) / 2  # Calculate the middle launch angle

    barrel_zone_label = 'Barrel Zone'
    barrel_zone_position = (label_ev, label_la)  # Position the label at the specified exit velocity and middle launch angle

    plt.text(barrel_zone_position[0], barrel_zone_position[1], barrel_zone_label, fontsize=11, alpha=0.8, color='green',
            fontweight='bold', ha='right', va='center', rotation=270)

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

    # Graph the distributions of runs scored
    plt.figure(figsize=(10, 6))
    max_runs = max(max(home_runs_scored), max(away_runs_scored))
    bins = range(0, max_runs + 2)  # Start from 0 and include the maximum runs scored
    plt.hist(home_runs_scored, bins=bins, alpha=0.6, label=f'{home_team}', color=team_colors[home_team][0], edgecolor='black', linewidth=1)
    plt.hist(away_runs_scored, bins=bins, alpha=0.6, label=f'{away_team}', color=team_colors[away_team][0], edgecolor='black', linewidth=1)
    plt.xlabel('Runs Scored', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(f'Distribution of Runs Scored ({num_simulations} Simulations)\nActual Score:     {away_team} {str(away_score)} - {home_team} {str(home_score)}\nDeserve-to-Win: {away_team} {str(away_win_percentage_str)}%, {home_team} {str(home_win_percentage_str)}%, Tie {tie_percentage_str}%', fontsize=16, loc = 'left', pad=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Save the plot to the "images" folder in the repository
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    plt.savefig(os.path.join(images_dir, f'{away_team}_{home_team}_{str(away_score)}-{str(home_score)}--{str(away_win_percentage_str)}-{str(home_win_percentage_str)}_rd.png'), bbox_inches='tight')
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
