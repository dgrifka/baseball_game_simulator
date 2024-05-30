from get_game_information import fetch_games, get_game_info

games_list, venues_list = fetch_games()

game_info_list = []

for game_id, venue in zip(games_list, venues_list):
  game_data = get_game_info(game_id)
  ## Merge venue_name since we use this in our model
  break
  ## Add simulation

print(game_data)
