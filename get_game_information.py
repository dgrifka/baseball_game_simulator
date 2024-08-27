import requests
import pandas as pd
import json
from pandas import json_normalize
import pytz
import datetime

from constants import base_url, league, season, endpoint, team_ver, schedule_ver, game_ver, venue_names

def response_code(base_url, ver, endpoint):
    """
    Sends a GET request to the specified API endpoint and returns the response data.

    Args:
        base_url (str): The base URL of the API.
        ver (str): The version of the API.
        endpoint (str): The specific endpoint to make the request to.

    Returns:
        dict: The parsed JSON data from the API response.
    """
    url = f'{base_url}{ver}/{endpoint}'
    print(url)
    response = requests.get(url)
    data = response.json()
    return data

def team_info():
    endpoint = 'teams'
    team_info = response_code(base_url, team_ver, endpoint)
    flattened_teams = json_normalize(team_info['teams'])

    teams_df = flattened_teams.copy()
    teams_df = teams_df[teams_df['sport.id'] == float(league)].reset_index(drop=True)
    teams_df = teams_df[teams_df['season'] == season].reset_index(drop=True)

    teams_df = teams_df.rename(columns = {"id": "team.id"})

    return teams_df


def fetch_games(days_ago, all_columns = False):

    schedule = response_code(base_url, schedule_ver, endpoint)

    games_list = []
    for key in schedule['dates']:
        for games in key['games']:
            flattened_game = json_normalize(games)
            games_list.append(flattened_game)

    # Concatenate all the individual DataFrames into one DataFrame
    games_df = pd.concat(games_list, ignore_index=True)

    ## Filter to only finished games
    finished_games = games_df[(games_df['status.abstractGameState'] == 'Final')].reset_index(drop = True)

    ## Only include yesterday's games
    finished_games['gameDate'] = pd.to_datetime(finished_games['gameDate'])
    utc_timezone = pytz.UTC

    current_date_time = datetime.datetime.now(utc_timezone)
    one_day_ago = current_date_time - datetime.timedelta(days=days_ago)

    filtered_games_df = finished_games[(finished_games['gameDate'] >= one_day_ago) & (finished_games['gameDate'] <= current_date_time)]

    ## Return yesterday's game as a list
    games_list = filtered_games_df['gamePk'].unique()
    ## Return the venues as well, since we use this information in the model
    venues_list = filtered_games_df['venue.name'].unique()

    # Filter the DataFrame based on venue_names
    filtered_games_df = filtered_games_df[filtered_games_df['venue.name'].isin(venue_names)].reset_index(drop=True)

    ## Trim the columns down to save memory
    if all_columns == False:
        filtered_games_df = filtered_games_df[["gamePk", "officialDate", "teams.away.team.id", "teams.away.team.name", "teams.away.score", "teams.home.team.id", "teams.home.team.name", "teams.home.score", "teams.home.isWinner"]]
    
    ## Print game information for troubleshooting
    print(list(games_list))
    number_of_games = len(games_list)
    print(f"Number of games: {number_of_games}")
    print(filtered_games_df[["gamePk", "teams.away.team.name", "teams.home.team.name"]]
        .apply(lambda row: f"{row['gamePk']}: {row['teams.away.team.name']}-{row['teams.home.team.name']}", axis=1))

    return filtered_games_df, games_list, venues_list


## Now, we want to get each game's info
def _play_info(df, column):

  """
  This function reads the columns from the dataframe created after reading in
  data from MLB Stats API.
  Each column includes a dictionary that must be parsed.
  """
  df_list = []

  for events, ab in zip(df[column], df['atBatIndex']):
    pbp = pd.json_normalize(events)
    ## Must add 1, since starts at 0
    pbp['ab_num'] = ab + 1
    ## Add ab_num to the front of the df
    cols = ['ab_num'] + [col for col in pbp if col != 'ab_num']
    pbp = pbp[cols]
    df_list.append(pbp)

  total_pbp = pd.concat(df_list)

  return total_pbp


def get_game_info(game_id, all_columns = False):

    # Columns to process in the json
    columns_to_process = ['result', 'about', 'count', 'matchup', 'runners', 'playEvents']

    endpoint = f'game/{game_id}/feed/live'

    game = response_code(base_url, game_ver, endpoint)

    try:
        # Attempt to extract the 'allPlays' section
        all_plays = game['liveData']['plays']['allPlays']

    except KeyError:
        # This block executes if 'liveData' or any nested key is missing
        print("Key 'liveData' not found in game:", game)
        return  # This will skip to the next iteration of the loop

    # Creating a DataFrame from the extracted data
    all_plays_df = pd.DataFrame(all_plays)

    ## Sometimes games will not have info
    if len(all_plays_df) == 0:
        return

    # Initialize an empty DataFrame for the merged result
    total_pbp = pd.DataFrame()

    # Loop through each column, apply the function, and merge the result
    for col in columns_to_process:
        col_pbp = _play_info(all_plays_df, col)

        if col == 'playEvents':
            col_pbp = col_pbp.drop(columns = 'index')
            ## Drop some metadata
            col_pbp = col_pbp[col_pbp['details.event'] != "Game Advisory"].reset_index(drop = True)
            col_pbp = col_pbp.drop(columns = ['startTime', 'endTime', 'type', 'details.event', 'details.eventType'])

        # Merge on 'ab_num', using an outer join to ensure all data is included
        if total_pbp.empty:
            total_pbp = col_pbp
        else:
            total_pbp = total_pbp.merge(col_pbp, on='ab_num', how='left')

    ## Add game ID
    total_pbp['gamePk'] = game_id

    ## Filter for only balls put in play or if the play is a strikeout/walk
    total_pbp_filtered = total_pbp.copy()
    other_plays = ['walk', 'hit_by_pitch', 'strikeout']
    total_pbp_filtered = total_pbp_filtered[(total_pbp_filtered['details.isInPlay'] == True) | (total_pbp_filtered['eventType'].isin(other_plays))]

    ## Add non batted balls, such as errors, sb, cs, and pickoffs
    non_bb = total_pbp.copy()
    non_bb = non_bb[(non_bb['details.isInPlay'] != True)]

    ## Then, filter to the last record of this occurrence, since each playId includes other irrelevant information, such as stolen bases, etc.
    total_pbp_filtered = total_pbp_filtered.drop_duplicates(subset="ab_num", keep="last")

    # Replace event types that are not 'single', 'double', 'triple', or 'home_run' with 'out'
    total_pbp_filtered['eventType'] = total_pbp_filtered['eventType'].apply(lambda x: x if x in ['single', 'double', 'triple', 'home_run', 'walk', 'hit_by_pitch'] else 'out')
    total_pbp_filtered['eventType'] = total_pbp_filtered['eventType'].str.replace("hit_by_pitch", "walk")

    ## Filter to only columns needed
    if all_columns == False:
        cols_needed = ["gamePk", "batter.fullName", "playId", "ab_num", "eventType", "description", "outs",
                       "isOut", "isTopInning", "inning", "hitData.launchSpeed", "hitData.launchAngle"]
        total_pbp_filtered = total_pbp_filtered[cols_needed]

    return total_pbp_filtered, non_bb
