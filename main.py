import pandas as pd

from get_game_information import team_info, fetch_games, get_game_info
from game_simulator import outcomes, simulator
from visualizations import wp_barplot, run_dist
teams_df = team_info()

games_df, games_list, venues_list = fetch_games()

game_info_list = []

for game_id, venue in zip(games_list, venues_list):
  game_data = get_game_info(game_id)
  game_data['venue.name'] = venue
  ## Get actual game info
  game_info = games_df.copy()
  game_info = game_info[game_info['gamePk'] == game_id]

  game_info = pd.merge(game_info, teams_df[['team.id', 'teamName']], left_on='teams.away.team.id', right_on='team.id')
  game_info = game_info.rename(columns={'teamName': 'away.team'})
  game_info = pd.merge(game_info, teams_df[['team.id', 'teamName']], left_on='teams.home.team.id', right_on='team.id')
  game_info = game_info.rename(columns={'teamName': 'home.team'})
  game_info = game_info.drop(columns=['team.id_x', 'team.id_y'])

  home_team = game_info['home.team'].values[0]
  away_team = game_info['away.team'].values[0]

  home_outcomes = outcomes(game_data, 'home')
  away_outcomes = outcomes(game_data, 'away')

  num_simulations = 5000
  home_runs_scored, away_runs_scored, home_win_percentage, away_win_percentage, tie_percentage = simulator(num_simulations, home_outcomes, away_outcomes)
  wp_barplot(num_simulations, home_win_percentage, away_win_percentage, tie_percentage, home_team, away_team)
  run_dist(num_simulations, home_runs_scored, away_runs_scored, home_team, away_team)

  break