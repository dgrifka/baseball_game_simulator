import matplotlib.pyplot as plt
from constants import team_colors

import os

def wp_barplot(num_simulations, home_win_percentage, away_win_percentage, tie_percentage, home_team, away_team, home_score, away_score):

    # Create a bar plot for win percentages
    labels = [f'{home_team}', f'{away_team}', 'Tie']
    percentages = [home_win_percentage, away_win_percentage, tie_percentage]
    colors = [team_colors[home_team][0], team_colors[away_team][0], '#808080']  # Use hex codes for bar colors
    away_win_percentage_str = f"{away_win_percentage:.0f}"
    home_win_percentage_str = f"{home_win_percentage:.0f}"
    plt.figure(figsize=(8, 6))
    plt.bar(labels, percentages, color=colors)
    plt.xlabel('Team', fontsize=14)
    plt.ylabel('Win Percentage', fontsize=14)
    plt.title(f'Win Percentages ({num_simulations} Simulations)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().set_yticklabels([f'{x:.0f}%' for x in plt.gca().get_yticks()])

    # Save the plot to the "images" folder in the repository
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    plt.savefig(os.path.join(images_dir, f'{away_team}_{home_team}_{str(away_score)}-{str(home_score)}--{str(away_win_percentage_str)}-{str(home_win_percentage_str)}_wp.png'))
    plt.close()

def run_dist(num_simulations, home_runs_scored, away_runs_scored, home_team, away_team, home_score, away_score, home_win_percentage, away_win_percentage):
    away_win_percentage_str = f"{away_win_percentage:.0f}"
    home_win_percentage_str = f"{home_win_percentage:.0f}"
    # Graph the distributions of runs scored
    plt.figure(figsize=(10, 6))
    plt.hist(home_runs_scored, bins=range(max(home_runs_scored)+2), alpha=0.7, label=f'{home_team}', color=team_colors[home_team][0], edgecolor='black', linewidth=1)
    plt.hist(away_runs_scored, bins=range(max(away_runs_scored)+2), alpha=0.7, label=f'{away_team}', color=team_colors[away_team][0], edgecolor='black', linewidth=1)
    plt.xlabel('Runs Scored', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(f'Distribution of Runs Scored ({num_simulations} Simulations)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    # Save the plot to the "images" folder in the repository
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    plt.savefig(os.path.join(images_dir, f'{away_team}_{home_team}_{str(away_score)}-{str(home_score)}--{str(away_win_percentage_str)}-{str(home_win_percentage_str)}_rd.png'))
    plt.close()