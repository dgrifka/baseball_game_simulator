import matplotlib.pyplot as plt
from constants import team_colors

def wp_barplot(num_simulations, home_win_percentage, away_win_percentage, tie_percentage, home_team, away_team):

    # Create a bar plot for win percentages
    labels = [f'{home_team}', f'{away_team}', 'Tie']
    percentages = [home_win_percentage, away_win_percentage, tie_percentage]
    colors = [team_colors[home_team][0], team_colors[away_team][0], '#808080']  # Use hex codes for bar colors

    plt.figure(figsize=(8, 6))
    plt.bar(labels, percentages, color=colors)
    plt.xlabel('Team', fontsize=14)
    plt.ylabel('Win Percentage', fontsize=14)
    plt.title(f'Win Percentages ({num_simulations} Simulations)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().set_yticklabels([f'{x:.0f}%' for x in plt.gca().get_yticks()])
    plt.show()

def run_dist(num_simulations, home_runs_scored, away_runs_scored, home_team, away_team):
    # Graph the distributions of runs scored
    plt.figure(figsize=(10, 6))
    plt.hist(home_runs_scored, bins=range(max(home_runs_scored)+2), alpha=0.75, label=f'{home_team}', color=team_colors[home_team][0])
    plt.hist(away_runs_scored, bins=range(max(away_runs_scored)+2), alpha=0.75, label=f'{away_team}', color=team_colors[away_team][0])
    plt.xlabel('Runs Scored', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(f'Distribution of Runs Scored ({num_simulations} Simulations)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.show()