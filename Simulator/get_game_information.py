import requests
import pandas as pd
import json
from pandas import json_normalize
import pytz
import datetime

from Simulator.constants import base_url, league, season, endpoint, team_ver, schedule_ver, game_ver, venue_names

def response_code(base_url, ver, endpoint):
    """
    Sends a GET request to the MLB Stats API endpoint and returns JSON response.

    Args:
        base_url (str): Base MLB Stats API URL
        ver (str): API version number
        endpoint (str): API endpoint path
    
    Returns:
        dict: Parsed JSON response data
    """
    url = f'{base_url}{ver}/{endpoint}'
    print(url)
    response = requests.get(url)
    return response.json()

def team_info():
    """
    Fetches and filters MLB team information for specified season and league.
    
    Returns:
        pd.DataFrame: Team data with columns filtered for current season/league
    """
    endpoint = 'teams'
    team_info = response_code(base_url, team_ver, endpoint)
    flattened_teams = json_normalize(team_info['teams'])

    teams_df = (flattened_teams[flattened_teams['sport.id'] == float(league)]
                .loc[flattened_teams['season'] == season]
                .rename(columns={"id": "team.id"})
                .reset_index(drop=True))
    
    return teams_df, flattened_teams

def fetch_games(days_ago, all_columns=False):
    """
    Fetches and filters MLB game data from specified number of days ago.
    
    Args:
        days_ago (int): Number of days in past to fetch games from
        all_columns (bool): If True, returns all columns; if False, returns subset
    
    Returns:
        tuple: (filtered_games_df, games_list)
            - filtered_games_df: DataFrame of filtered game data
            - games_list: List of unique game IDs
    """
    schedule = response_code(base_url, schedule_ver, endpoint)
    
    games_list = [json_normalize(games) for key in schedule['dates'] 
                 for games in key['games']]
    games_df = pd.concat(games_list, ignore_index=True)
    
    # Filter to finished games within date range
    finished_games = games_df[games_df['status.abstractGameState'] == 'Final']
    finished_games['gameDate'] = pd.to_datetime(finished_games['gameDate'])
    
    current_date_time = datetime.datetime.now(pytz.UTC)
    one_day_ago = current_date_time - datetime.timedelta(days=days_ago)
    
    filtered_games_df = (finished_games[
        (finished_games['gameDate'] >= one_day_ago) & 
        (finished_games['gameDate'] <= current_date_time) &
        (finished_games['venue.name'].isin(venue_names))
    ].reset_index(drop=True))
    
    # Handle stadium name mapping
    filtered_games_df['venue.name'] = filtered_games_df['venue.name'].replace(
        'George M. Steinbrenner Field', 'Yankee Stadium')
    
    if not all_columns:
        filtered_games_df = filtered_games_df[[
            "gamePk", "officialDate", "venue.name", 
            "teams.away.team.id", "teams.away.team.name", "teams.away.score",
            "teams.home.team.id", "teams.home.team.name", "teams.home.score",
            "teams.home.isWinner"
        ]]
    
    games_list = filtered_games_df['gamePk'].unique()
    
    # Print game information
    print(list(games_list))
    print(f"Number of games: {len(games_list)}")
    print(filtered_games_df[["gamePk", "teams.away.team.name", "teams.home.team.name"]]
        .apply(lambda row: f"{row['gamePk']}: {row['teams.away.team.name']}-{row['teams.home.team.name']}", axis=1))
    
    return filtered_games_df, games_list

def play_info(df, column):
    """
    Parses play-by-play data from nested JSON columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing play-by-play data
        column (str): Name of column containing nested JSON
    
    Returns:
        pd.DataFrame: Flattened play-by-play data with at-bat numbers
    """
    df_list = []
    for events, ab in zip(df[column], df['atBatIndex']):
        pbp = pd.json_normalize(events)
        pbp['ab_num'] = ab + 1
        cols = ['ab_num'] + [col for col in pbp if col != 'ab_num']
        df_list.append(pbp[cols])
    
    return pd.concat(df_list)

def secondary_play_type(non_batted, play):
    """
    Returns DataFrame with non batted-ball data (i.e. steals, pickoffs).
    
    Args:
        non_batted (pd.DataFrame): DataFrame containing play-by-play data
        play (str): Name of secondary play type
    
    Returns:
        pd.DataFrame: DataFrame with steals or pickoffs
    """
    non_batted_play = non_batted[non_batted['details.movementReason'].str.contains(play, na=False)].copy()
    non_batted_play = non_batted_play[non_batted_play['isBaseRunningPlay'].notnull()]
    non_batted_play['play'] = play
    return non_batted_play
    
def get_game_info(game_id, all_columns=False):
    """
    Retrieves and processes detailed game play-by-play information.
    
    Args:
        game_id (int): MLB game ID
        all_columns (bool): If True, returns all columns; if False, returns subset
    
    Returns:
        tuple: (filtered_pbp, total_pbp, non_batted_balls) 
        - filtered_pbp: DataFrame of filtered play-by-play data
        - total_pbp: DataFrame of all play-by-play data
        - non_batted_balls: DataFrame of non-batted ball events
    """
    endpoint = f'game/{game_id}/feed/live'
    game = response_code(base_url, game_ver, endpoint)
    
    try:
        all_plays = game['liveData']['plays']['allPlays']
    except KeyError:
        print("Key 'liveData' not found in game:", game)
        return
        
    all_plays_df = pd.DataFrame(all_plays)
    if len(all_plays_df) == 0:
        return
        
    total_pbp = pd.DataFrame()
    columns_to_process = ['result', 'about', 'count', 'matchup', 'runners', 'playEvents']
    
    for col in columns_to_process:
        col_pbp = play_info(all_plays_df, col)
        
        if col == 'playEvents':
            col_pbp = (col_pbp[col_pbp['details.event'] != "Game Advisory"]
                      .drop(columns=['startTime', 'endTime', 'type', 'details.event', 'details.eventType', 'index'])
                      .reset_index(drop=True))
        
        total_pbp = col_pbp if total_pbp.empty else total_pbp.merge(col_pbp, on='ab_num', how='left')
    
    total_pbp['gamePk'] = game_id
    
    # Create copy for filtered batted balls
    total_pbp_filtered = total_pbp.copy()
    other_plays = ['walk', 'hit_by_pitch', 'strikeout']
    total_pbp_filtered = total_pbp_filtered[
        (total_pbp_filtered['details.isInPlay'] == True) |
        (total_pbp_filtered['eventType'].isin(other_plays))
    ]
    
    # Create DataFrame for non-batted balls (i.e. steals and pickoffs)
    non_batted_balls = total_pbp[total_pbp['details.isInPlay'] != True].copy()
    steals = secondary_play_type(non_batted_balls, "stolen_base")
    pickoffs = secondary_play_type(non_batted_balls, "pickoff")
    steals_and_pickoffs = pd.concat([steals, pickoffs])
    
    # Clean up event types for filtered data
    total_pbp_filtered = (total_pbp_filtered
        .drop_duplicates(subset="ab_num", keep="last")
        .assign(eventType=lambda x: x['eventType'].apply(
            lambda y: y if y in ['single', 'double', 'triple', 'home_run', 'walk', 'hit_by_pitch'] 
            else 'out'
        )))
    total_pbp_filtered['eventType'] = total_pbp_filtered['eventType'].str.replace("hit_by_pitch", "walk")
    
    if not all_columns:
        cols_needed = [
            "gamePk", "batter.fullName", "playId", "ab_num", "eventType", 
            "description", "outs", "isOut", "isTopInning", "inning",
            "hitData.launchSpeed", "hitData.launchAngle"
        ]
        total_pbp_filtered = total_pbp_filtered[cols_needed]
        non_batted_balls = non_batted_balls[cols_needed]
    
    return total_pbp_filtered, total_pbp, steals_and_pickoffs
