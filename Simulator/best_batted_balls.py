import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm

from Simulator.get_game_information import response_code, get_game_info, team_info
from Simulator.game_simulator import create_features_for_prediction, create_detailed_outcomes_df
from Simulator.visualizations import create_estimated_bases_table
from Simulator.constants import base_url, schedule_ver, venue_names, team_colors, mlb_team_logos

# Load the pipeline - handle both local and Colab paths
import pickle
import os

# Try multiple paths for the model
model_paths = [
    'Model/gb_classifier_pipeline_improved.pkl',
    '/content/baseball_game_simulator/Model/gb_classifier_pipeline_improved.pkl',
    'gb_classifier_pipeline_improved.pkl'
]

pipeline = None
for path in model_paths:
    if os.path.exists(path):
        with open(path, 'rb') as file:
            pipeline = pickle.load(file)
        print(f"Model loaded from: {path}")
        break

if pipeline is None:
    raise FileNotFoundError("Could not find the model file gb_classifier_pipeline_improved.pkl in any of the expected locations")


def parse_date_range(date_input):
    """
    Parse date input to get start and end dates.
    
    Args:
        date_input (str or tuple): Either a single date string ('2025-05-01') or 
                                   tuple of date strings ('2025-05-01', '2025-05-05')
    
    Returns:
        tuple: (start_date, end_date) as datetime objects
    """
    if isinstance(date_input, str):
        # Single date
        date = datetime.strptime(date_input, '%Y-%m-%d')
        date = pytz.UTC.localize(date)
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = date.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif isinstance(date_input, (list, tuple)) and len(date_input) == 2:
        # Date range
        start_date = datetime.strptime(date_input[0], '%Y-%m-%d')
        end_date = datetime.strptime(date_input[1], '%Y-%m-%d')
        start_date = pytz.UTC.localize(start_date).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = pytz.UTC.localize(end_date).replace(hour=23, minute=59, second=59, microsecond=999999)
    else:
        raise ValueError("date_input must be a single date string or a tuple of two date strings")
    
    return start_date, end_date


def fetch_games_by_date_range(start_date, end_date):
    """
    Fetch all games within a specific date range.
    
    Args:
        start_date (datetime): Start date (timezone-aware)
        end_date (datetime): End date (timezone-aware)
    
    Returns:
        tuple: (filtered_games_df, games_list)
    """
    # Calculate the date range for the API call
    days_diff = (end_date - start_date).days + 1
    
    # Format dates for the API
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Build endpoint with date range
    endpoint = f'schedule?sportId=1&startDate={start_str}&endDate={end_str}'
    
    # Get schedule data
    schedule = response_code(base_url, schedule_ver, endpoint)
    
    if 'dates' not in schedule:
        print(f"No games found between {start_str} and {end_str}")
        return pd.DataFrame(), []
    
    # Process games
    games_list = []
    for date_data in schedule['dates']:
        for game in date_data['games']:
            games_list.append(pd.json_normalize(game))
    
    if not games_list:
        return pd.DataFrame(), []
    
    games_df = pd.concat(games_list, ignore_index=True)
    
    # Filter to finished games at valid venues
    finished_games = games_df[games_df['status.abstractGameState'] == 'Final'].copy()
    finished_games['gameDate'] = pd.to_datetime(finished_games['gameDate'])
    
    filtered_games_df = finished_games[
        finished_games['venue.name'].isin(venue_names)
    ].reset_index(drop=True)
    
    # Handle stadium name mapping
    stadium_mapping = {
        'George M. Steinbrenner Field': 'Yankee Stadium',
        'Sutter Health Park': 'Oakland Coliseum',
        'Daikin Park': 'Minute Maid Park',
        'Rate Field': 'Guaranteed Rate Field'
    }
    
    for old_name, new_name in stadium_mapping.items():
        filtered_games_df['venue.name'] = filtered_games_df['venue.name'].replace(old_name, new_name)
    
    # Select relevant columns
    filtered_games_df = filtered_games_df[[
        "gamePk", "officialDate", "venue.name", 
        "teams.away.team.id", "teams.away.team.name", "teams.away.score",
        "teams.home.team.id", "teams.home.team.name", "teams.home.score",
        "teams.home.isWinner"
    ]]
    
    games_list = filtered_games_df['gamePk'].unique()
    
    print(f"Found {len(games_list)} games between {start_str} and {end_str}")
    
    return filtered_games_df, games_list


def process_game_batted_balls(game_id, game_info_df):
    """
    Process a single game to extract batted ball information with estimated bases.
    
    Args:
        game_id: Game ID
        game_info_df: DataFrame with game information
    
    Returns:
        pd.DataFrame: Batted balls with estimated bases and metadata
    """
    try:
        # Get game data
        game_row = game_info_df[game_info_df['gamePk'] == game_id].iloc[0]
        
        # Get play-by-play data
        total_pbp_filtered, total_pbp, steals_and_pickoffs = get_game_info(game_id)
        
        if total_pbp_filtered is None or total_pbp_filtered.empty:
            return pd.DataFrame()
        
        # Add venue information
        total_pbp_filtered['venue.name'] = game_row['venue.name']
        
        # Filter for batted balls with launch data
        batted_balls = total_pbp_filtered[
            ~total_pbp_filtered['hitData.launchSpeed'].isnull() & 
            ~total_pbp_filtered['hitData.launchAngle'].isnull()
        ].copy()
        
        if batted_balls.empty:
            return pd.DataFrame()
        
        # Calculate estimated bases for each batted ball
        estimated_bases_list = []
        
        for _, row in batted_balls.iterrows():
            features = create_features_for_prediction(
                row['hitData.launchSpeed'],
                row['hitData.launchAngle'], 
                row['venue.name']
            )
            
            # Get probabilities
            probs = pipeline.predict_proba(features)[0]
            class_labels = pipeline.classes_
            prob_dict = dict(zip(class_labels, probs))
            
            # Calculate estimated bases
            estimated_bases = (
                prob_dict.get('single', 0) * 1 + 
                prob_dict.get('double', 0) * 2 + 
                prob_dict.get('triple', 0) * 3 + 
                prob_dict.get('home_run', 0) * 4
            )
            
            estimated_bases_list.append({
                'Player': row['batter.fullName'],
                'Team': game_row['teams.home.team.name'] if not row['isTopInning'] else game_row['teams.away.team.name'],
                'Opponent': game_row['teams.away.team.name'] if not row['isTopInning'] else game_row['teams.home.team.name'],
                'Date': game_row['officialDate'],
                'Game': f"{game_row['teams.away.team.name']} @ {game_row['teams.home.team.name']}",
                'Launch Speed': row['hitData.launchSpeed'],
                'Launch Angle': row['hitData.launchAngle'],
                'Venue': row['venue.name'],
                'Result': row['eventType'],
                'Estimated Bases': estimated_bases,
                'Out Prob': f"{prob_dict.get('out', 0)*100:.1f}%",
                'Single Prob': f"{prob_dict.get('single', 0)*100:.1f}%",
                'Double Prob': f"{prob_dict.get('double', 0)*100:.1f}%",
                'Triple Prob': f"{prob_dict.get('triple', 0)*100:.1f}%",
                'Hr Prob': f"{prob_dict.get('home_run', 0)*100:.1f}%"
            })
        
        return pd.DataFrame(estimated_bases_list)
        
    except Exception as e:
        print(f"Error processing game {game_id}: {str(e)}")
        return pd.DataFrame()


def get_best_batted_balls_by_date(date_input, top_n=25, images_dir="images"):
    """
    Get the best batted balls across all games for a given date or date range.
    
    Args:
        date_input (str or tuple): Either a single date string ('2025-05-01') or 
                                   tuple of date strings ('2025-05-01', '2025-05-05')
        top_n (int): Number of top batted balls to return (default: 25)
        images_dir (str): Directory to save the visualization
    
    Returns:
        pd.DataFrame: Top batted balls ranked by estimated bases
    """
    # Parse dates
    start_date, end_date = parse_date_range(date_input)
    
    # Fetch games
    games_df, games_list = fetch_games_by_date_range(start_date, end_date)
    
    if len(games_list) == 0:
        print("No games found in the specified date range")
        return pd.DataFrame()
    
    # Process all games
    all_batted_balls = []
    
    print(f"Processing {len(games_list)} games...")
    for game_id in tqdm(games_list):
        game_balls = process_game_batted_balls(game_id, games_df)
        if not game_balls.empty:
            all_batted_balls.append(game_balls)
    
    if not all_batted_balls:
        print("No batted balls found in any games")
        return pd.DataFrame()
    
    # Combine all batted balls
    combined_df = pd.concat(all_batted_balls, ignore_index=True)
    
    # Sort by estimated bases (descending)
    combined_df = combined_df.sort_values('Estimated Bases', ascending=False).reset_index(drop=True)
    
    # Get top N
    top_balls = combined_df.head(top_n).copy()
    
    # Add team colors
    teams_df = team_info()
    team_color_dict = {}
    for _, team in teams_df.iterrows():
        team_name = team['name']
        if team_name in team_colors:
            team_color_dict[team_name] = team_colors[team_name][0]
    
    top_balls['team_color'] = top_balls['Team'].map(team_color_dict)
    
    # Create visualization
    if len(top_balls) >= 15:
        # Prepare title information
        date_str = start_date.strftime('%Y-%m-%d')
        if start_date.date() != end_date.date():
            date_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        
        # Create a modified version of the table for all games
        create_all_games_estimated_bases_table(
            top_balls.head(15),
            date_str,
            len(games_list),
            images_dir
        )
    
    return top_balls


def create_all_games_estimated_bases_table(df, date_str, num_games, images_dir):
    """
    Creates table visualization for best batted balls across all games.
    Modified version of create_estimated_bases_table for multiple games.
    """
    import matplotlib.pyplot as plt
    import os
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.close('all')
    
    # Prepare data
    df = df.copy()
    team_color_map = dict(zip(df['Team'], df['team_color']))
    
    # Format the data similar to the original function
    df.insert(0, 'Rank', range(1, len(df) + 1))
    df['Player'] = df['Player'].apply(lambda x: x.split()[0][0] + '. ' + ' '.join(x.split()[1:]))
    df['Result'] = df['Result'].str.replace('_', ' ').str.title()
    
    # Create xBA
    df['xBA'] = ((df['Single Prob'].str.rstrip('%').astype(float) + 
                  df['Double Prob'].str.rstrip('%').astype(float) + 
                  df['Triple Prob'].str.rstrip('%').astype(float) + 
                  df['Hr Prob'].str.rstrip('%').astype(float)) / 100).round(3)
    df['xBA'] = df['xBA'].apply(lambda x: f'.{str(int(x*1000)).zfill(3)}')
    
    df['HR%'] = df['Hr Prob']
    df['Launch Angle'] = df['Launch Angle'].astype(str) + '°'
    df['Launch Speed'] = df['Launch Speed'].astype(str) + ' mph'
    df['Estimated Bases'] = df['Estimated Bases'].round(2)
    
    # Select columns
    columns_to_keep = ['Rank', 'Team', 'Player', 'Launch Speed', 'Launch Angle', 
                       'Result', 'Estimated Bases', 'xBA', 'HR%', 'Game']
    df = df[columns_to_keep]
    
    # Create figure
    fig = plt.figure(figsize=(22, 10), dpi=150)
    ax = fig.add_subplot(111)
    ax.set_position([0.04, 0.05, 0.92, 0.65])
    ax.axis('off')
    
    # Create table with adjusted column widths for the Game column
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.05, 0.07, 0.12, 0.10, 0.10, 0.08, 0.12, 0.07, 0.07, 0.22])
    
    # Apply styling (similar to original but adapted for all games)
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header
            cell.set_height(0.06)
            cell.set_text_props(weight='bold', fontsize=14)
            cell.set_facecolor('#2C3E50')
            cell.get_text().set_color('white')
            cell.set_edgecolor('#1A252F')
            cell.set_linewidth(2)
        else:  # Data rows
            cell.set_height(0.055)
            cell.set_text_props(fontsize=12)
            cell.set_edgecolor('#E0E0E0')
            cell.set_linewidth(0.5)
            
            if row % 2 == 0:
                cell.set_facecolor('#F8F9FA')
            else:
                cell.set_facecolor('#FFFFFF')
            
            # Team colors
            if col == 1:  # Team column
                team = df.iloc[row-1]['Team']
                if team in team_color_map:
                    cell.set_facecolor(team_color_map[team])
                    # Check if dark color
                    from matplotlib.colors import to_rgb
                    r, g, b = to_rgb(team_color_map[team])
                    if (r * 0.299 + g * 0.587 + b * 0.114) < 0.5:
                        cell.get_text().set_color('white')
                cell.get_text().set_weight('bold')
            
            # Estimated bases gradient
            if col == 6:  # Estimated Bases column
                bases_value = df.iloc[row-1]['Estimated Bases']
                norm = plt.Normalize(0, 4)
                cmap = plt.cm.YlOrRd
                color = cmap(norm(bases_value))
                cell.set_facecolor(color)
                cell.get_text().set_weight('bold')
                cell.get_text().set_fontsize(14)
                if bases_value > 2.5:
                    cell.get_text().set_color('white')
    
    table.auto_set_font_size(False)
    table.scale(1.1, 1.5)
    
    # Title
    title_lines = [
        f"Top 15 Batted Balls by Estimated Total Bases",
        f"All Games • {date_str} • {num_games} Total Games"
    ]
    
    plt.text(0.5, 0.94, title_lines[0], transform=fig.transFigure,
             fontsize=22, fontweight='bold', ha='center', va='top')
    
    plt.text(0.5, 0.88, title_lines[1], transform=fig.transFigure,
             fontsize=16, ha='center', va='top', color='#333333')
    
    # Attribution
    plt.text(0.1, 0.92, 'Data: MLB', 
             transform=fig.transFigure, fontsize=17, 
             ha='left', va='top', color='#999999')
    
    plt.text(0.1, 0.895, 'By: @mlb_simulator', 
             transform=fig.transFigure, fontsize=17, 
             ha='left', va='top', color='#999999')
    
    # Save
    os.makedirs(images_dir, exist_ok=True)
    filename = f'all_games_best_batted_balls_{date_str.replace(" ", "_")}.png'
    plt.savefig(os.path.join(images_dir, filename), 
                bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Visualization saved to {os.path.join(images_dir, filename)}")
