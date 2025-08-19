import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm
import matplotlib.pyplot as plt

from Simulator.get_game_information import response_code, get_game_info, team_info
from Simulator.game_simulator import create_features_for_prediction, create_detailed_outcomes_df
from Simulator.visualizations import create_estimated_bases_table
from Simulator.constants import base_url, schedule_ver, venue_names, team_colors, mlb_team_logos
from Simulator.team_mapping import get_team_short_name, get_team_colors_key

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

# Test the pipeline with a simple prediction
try:
    test_features = create_features_for_prediction(100, 25, "Yankee Stadium")
    test_probs = pipeline.predict_proba(test_features)[0]
    print(f"Model test - Classes: {pipeline.classes_}")
    print(f"Model test - Test probabilities: {test_probs}")
    print(f"Model test - Sum of probabilities: {sum(test_probs)}")
except Exception as e:
    print(f"Model test failed: {e}")

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


def get_team_logo(team_name, mlb_team_logos, logo_cache={}):
    """
    Retrieves team logo URL with caching.
    
    Args:
        team_name (str): MLB team name
        mlb_team_logos (list): List of team logo URL dictionaries
        logo_cache (dict): Cache for logo URLs
        
    Returns:
        str: Logo URL or None if not found
    """
    if team_name in logo_cache:
        return logo_cache[team_name]
        
    logo_url = next((team['logo_url'] for team in mlb_team_logos 
                    if team['team'] == team_name), None)
    if logo_url:
        logo_cache[team_name] = logo_url
    else:
        print(f"Logo not found for {team_name}")
    return logo_url

def getImage(path, zoom=0.5, size=(50, 50), alpha=0.6, image_cache={}):
    """
    Processes team logo image with caching of raw image data.
    
    Args:
        path (str): Image URL
        zoom (float): Image zoom level
        size (tuple): Target image size
        alpha (float): Transparency level
        image_cache (dict): Cache for processed image data
        
    Returns:
        OffsetImage: Fresh OffsetImage instance for each call
    """
    try:
        # Cache the processed image data, not the OffsetImage
        if path not in image_cache:
            img = Image.open(BytesIO(requests.get(path).content))
            img = img.resize(size, Image.LANCZOS).convert("RGBA")
            
            # Enhance and crop
            img = ImageEnhance.Sharpness(img).enhance(2.5)
            data = np.array(img)
            mask = data[:, :, 3] > 0
            ymin, ymax = np.where(np.any(mask, axis=1))[0][[0, -1]]
            xmin, xmax = np.where(np.any(mask, axis=0))[0][[0, -1]]
            
            # Create transparent background
            new_img = Image.new('RGBA', size, (255, 255, 255, 0))
            cropped = img.crop((xmin, ymin, xmax+1, ymax+1))
            paste_pos = tuple((s - c) // 2 for s, c in zip(size, cropped.size))
            new_img.paste(cropped, paste_pos, cropped)
            
            # Apply alpha and store the processed data
            data = np.array(new_img)
            data[:, :, 3] = (data[:, :, 3] * alpha).astype(np.uint8)
            image_cache[path] = data
        
        # Create a fresh OffsetImage instance for each call
        return OffsetImage(Image.fromarray(image_cache[path].copy()), zoom=zoom)
    
    except Exception as e:
        print(f"Error loading image from {path}: {str(e)}")
        return None

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
        
        # Get team short names for color mapping
        home_team_full = game_row['teams.home.team.name']
        away_team_full = game_row['teams.away.team.name']
        home_team_short = get_team_short_name(home_team_full)
        away_team_short = get_team_short_name(away_team_full)
        
        # Calculate estimated bases for each batted ball
        estimated_bases_list = []
        
        for _, row in batted_balls.iterrows():
            try:
                # Validate input data
                launch_speed = float(row['hitData.launchSpeed'])
                launch_angle = float(row['hitData.launchAngle'])
                venue = str(row['venue.name'])
                
                # Create features
                features = create_features_for_prediction(launch_speed, launch_angle, venue)
                
                # Get probabilities
                probs = pipeline.predict_proba(features)[0]
                class_labels = pipeline.classes_
                
                # Map numeric classes to outcome names
                # Based on typical baseball outcome encoding: 0=out, 1=single, 2=double, 3=triple, 4=home_run
                class_mapping = {0: 'out', 1: 'single', 2: 'double', 3: 'triple', 4: 'home_run'}
                prob_dict = {}
                
                for i, prob in enumerate(probs):
                    if i < len(class_labels):
                        class_num = class_labels[i]
                        outcome_name = class_mapping.get(class_num, f'class_{class_num}')
                        prob_dict[outcome_name] = prob
                
                # Calculate estimated bases
                estimated_bases = (
                    prob_dict.get('single', 0) * 1 + 
                    prob_dict.get('double', 0) * 2 + 
                    prob_dict.get('triple', 0) * 3 + 
                    prob_dict.get('home_run', 0) * 4
                )
                    
            except Exception as e:
                print(f"Error processing batted ball: {e}")
                print(f"Launch Speed: {row['hitData.launchSpeed']}, Launch Angle: {row['hitData.launchAngle']}")
                # Set default values for failed predictions
                prob_dict = {'out': 1.0, 'single': 0, 'double': 0, 'triple': 0, 'home_run': 0}
                estimated_bases = 0
            
            # Use short team names for consistency with team_colors
            team_short = home_team_short if not row['isTopInning'] else away_team_short
            opponent_short = away_team_short if not row['isTopInning'] else home_team_short
            
            estimated_bases_list.append({
                'Player': row['batter.fullName'],
                'Team': team_short,
                'Opponent': opponent_short,
                'Date': game_row['officialDate'],
                'Game': f"{away_team_short} @ {home_team_short}",
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
    
    # Debug: Show unique result types
    print(f"Unique result types found: {sorted(combined_df['Result'].unique())}")
    
    # Get top N
    top_balls = combined_df.head(top_n).copy()
    
    # Add team colors using proper team name mapping
    team_color_dict = {}
    for team_name in top_balls['Team'].unique():
        if pd.notna(team_name):  # Check for NaN values
            colors_key = get_team_colors_key(team_name)
            if colors_key and colors_key in team_colors:
                team_color_dict[team_name] = team_colors[colors_key][0]
            else:
                # Default color if mapping fails
                team_color_dict[team_name] = "#666666"
                print(f"Warning: No color found for team '{team_name}', using default color")
        else:
            team_color_dict[team_name] = "#666666"
    
    top_balls['team_color'] = top_balls['Team'].map(team_color_dict)
    
    # Prepare date string for titles
    date_str = start_date.strftime('%Y-%m-%d')
    if start_date.date() != end_date.date():
        date_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    
    # Create visualizations
    if len(top_balls) >= 15:
        # Create Top 15 visualization
        create_all_games_estimated_bases_table(
            top_balls.head(15),
            date_str,
            len(games_list),
            images_dir,
            table_type="top"
        )
        
        # Create Bottom 15 visualization if we have enough data
        if len(combined_df) >= 30:
            bottom_balls = combined_df.tail(15).copy()
            # Add team colors for bottom balls too
            bottom_team_color_dict = {}
            for team_name in bottom_balls['Team'].unique():
                if pd.notna(team_name):
                    colors_key = get_team_colors_key(team_name)
                    if colors_key and colors_key in team_colors:
                        bottom_team_color_dict[team_name] = team_colors[colors_key][0]
                    else:
                        bottom_team_color_dict[team_name] = "#666666"
                else:
                    bottom_team_color_dict[team_name] = "#666666"
            
            bottom_balls['team_color'] = bottom_balls['Team'].map(bottom_team_color_dict)
            
            create_all_games_estimated_bases_table(
                bottom_balls,
                date_str,
                len(games_list),
                images_dir,
                table_type="bottom"
            )
    
    # Create Luckiest Hits visualization
    # Define what constitutes a hit
    hit_types = ['single', 'double', 'triple', 'home_run']
    hits_df = combined_df[combined_df['Result'].isin(hit_types)].copy()
    print(f"Found {len(hits_df)} hits out of {len(combined_df)} total batted balls")
    if len(hits_df) >= 15:
        luckiest_hits = hits_df.sort_values('Estimated Bases', ascending=True).head(15).copy()
        
        # Add team colors
        lucky_team_color_dict = {}
        for team_name in luckiest_hits['Team'].unique():
            if pd.notna(team_name):
                colors_key = get_team_colors_key(team_name)
                if colors_key and colors_key in team_colors:
                    lucky_team_color_dict[team_name] = team_colors[colors_key][0]
                else:
                    lucky_team_color_dict[team_name] = "#666666"
            else:
                lucky_team_color_dict[team_name] = "#666666"
        
        luckiest_hits['team_color'] = luckiest_hits['Team'].map(lucky_team_color_dict)
        
        create_all_games_estimated_bases_table(
            luckiest_hits,
            date_str,
            len(games_list),
            images_dir,
            table_type="luckiest"
        )
    else:
        print(f"Not enough hits to create luckiest hits visualization (need 15, have {len(hits_df)})")
    
    # Create Unluckiest Outs visualization
    # Define what constitutes an out (any result that's not a hit)
    hit_types = ['single', 'double', 'triple', 'home_run']
    outs_df = combined_df[~combined_df['Result'].isin(hit_types)].copy()
    print(f"Found {len(outs_df)} outs out of {len(combined_df)} total batted balls")
    if len(outs_df) >= 15:
        unluckiest_outs = outs_df.sort_values('Estimated Bases', ascending=False).head(15).copy()
        
        # Add team colors
        unlucky_team_color_dict = {}
        for team_name in unluckiest_outs['Team'].unique():
            if pd.notna(team_name):
                colors_key = get_team_colors_key(team_name)
                if colors_key and colors_key in team_colors:
                    unlucky_team_color_dict[team_name] = team_colors[colors_key][0]
                else:
                    unlucky_team_color_dict[team_name] = "#666666"
            else:
                unlucky_team_color_dict[team_name] = "#666666"
        
        unluckiest_outs['team_color'] = unluckiest_outs['Team'].map(unlucky_team_color_dict)
        
        create_all_games_estimated_bases_table(
            unluckiest_outs,
            date_str,
            len(games_list),
            images_dir,
            table_type="unluckiest"
        )
    else:
        print(f"Not enough outs to create unluckiest outs visualization (need 15, have {len(outs_df)})")
    
    return top_balls

def create_all_games_estimated_bases_table(df, date_str, num_games, images_dir, table_type="top"):
    """
    Creates table visualization for best/worst/luckiest/unluckiest batted balls across all games.
    Modified to match the style of create_estimated_bases_table with logos and enhanced formatting.
    
    Args:
        df: DataFrame with batted ball data
        date_str: Date string for title
        num_games: Number of games analyzed
        images_dir: Directory to save images
        table_type: "top", "bottom", "luckiest", or "unluckiest" for the type of table
    """
    import matplotlib.pyplot as plt
    import os
    from matplotlib.offsetbox import AnnotationBbox
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.close('all')
    
    # Prepare data
    df = df.copy()
    team_color_map = dict(zip(df['Team'], df['team_color']))
    
    # Store team names before data preparation (for logo lookup)
    team_names = df['Team'].tolist()
    
    # For bottom table, we want the worst 15 (lowest estimated bases)
    if table_type == "bottom":
        df = df.sort_values('Estimated Bases', ascending=True).head(15)
    else:
        # For all other types (top, luckiest, unluckiest), data is already sorted correctly
        df = df.head(15)
    
    # Format the data - KEEP FULL PLAYER NAMES (no abbreviation)
    if table_type == "bottom":
        df.insert(0, 'Rank', [f"W{i}" for i in range(1, len(df) + 1)])
    elif table_type == "luckiest":
        df.insert(0, 'Rank', [f"L{i}" for i in range(1, len(df) + 1)])
    elif table_type == "unluckiest":
        df.insert(0, 'Rank', [f"U{i}" for i in range(1, len(df) + 1)])
    else:
        df.insert(0, 'Rank', range(1, len(df) + 1))
    
    # REMOVED: Player name abbreviation - keeping full names now
    # df['Player'] = df['Player'].apply(lambda x: x.split()[0][0] + '. ' + ' '.join(x.split()[1:]))
    
    # Format Result column
    df['Result'] = df['Result'].str.replace('_', ' ').str.title()
    
    # Create xBA (expected batting average) from hit probabilities
    df['xBA'] = ((df['Single Prob'].str.rstrip('%').astype(float) + 
                  df['Double Prob'].str.rstrip('%').astype(float) + 
                  df['Triple Prob'].str.rstrip('%').astype(float) + 
                  df['Hr Prob'].str.rstrip('%').astype(float)) / 100).round(3)
    df['xBA'] = df['xBA'].apply(lambda x: f'.{str(int(x*1000)).zfill(3)}')
    
    # Simplify probability columns - keep only HR probability as it's most impactful
    df['HR%'] = df['Hr Prob']
    
    # Format launch angle to include degree symbol
    df['Launch Angle'] = df['Launch Angle'].astype(str) + '°'
    
    # Format launch speed
    df['Launch Speed'] = df['Launch Speed'].astype(str) + ' mph'
    
    # Round estimated bases to 2 decimals
    df['Estimated Bases'] = df['Estimated Bases'].round(2)
    
    # Select columns - Note: Game column makes this table wider
    columns_to_keep = ['Rank', 'Team', 'Player', 'Launch Speed', 'Launch Angle', 
                       'Result', 'Estimated Bases', 'xBA', 'HR%', 'Game']
    df = df[columns_to_keep]
    
    # Create figure with same proportions as individual game table
    fig = plt.figure(figsize=(20, 10), dpi=150)
    ax = fig.add_subplot(111)
    ax.set_position([0.05, 0.05, 0.9, 0.65])  # Match individual game table positioning
    ax.axis('off')
    
    # Adjusted column widths to match individual game table style
    # Tighter Team column for logo, wider Player column for full names
    col_widths = [0.05, 0.06, 0.20, 0.10, 0.10, 0.09, 0.11, 0.07, 0.07, 0.15]
    
    # Create table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    loc='center',
                    cellLoc='center',
                    colWidths=col_widths)
    
    # Apply enhanced styling with logos
    apply_all_games_enhanced_styling(table, df, team_color_map)
    
    # Scale table to match individual game table
    table.auto_set_font_size(False)
    table.scale(1.1, 1.5)
    
    # Add team logos after table is created and scaled
    add_logos_to_all_games_table(ax, table, team_names, mlb_team_logos, df)
    
    # Enhanced title with consistent formatting
    if table_type == "bottom":
        title_lines = [
            f"Bottom 15 Batted Balls by Estimated Total Bases",
            f"All Games • {date_str} • {num_games} Total Games"
        ]
    elif table_type == "luckiest":
        title_lines = [
            f"15 Luckiest Hits",
            f"All Games • {date_str} • {num_games} Total Games"
        ]
    elif table_type == "unluckiest":
        title_lines = [
            f"15 Unluckiest Outs",
            f"All Games • {date_str} • {num_games} Total Games"
        ]
    else:
        title_lines = [
            f"Top 15 Batted Balls by Estimated Total Bases",
            f"All Games • {date_str} • {num_games} Total Games"
        ]
    
    # Main title - consistent positioning with individual game table
    plt.text(0.5, 0.93, title_lines[0], transform=fig.transFigure,
             fontsize=23, fontweight='bold', ha='center', va='top')
    
    # Subtitle
    plt.text(0.5, 0.88, title_lines[1], transform=fig.transFigure,
             fontsize=16, ha='center', va='top', color='#333333')
    
    # Attribution - match individual game table style
    plt.text(0.1, 0.915, 'Data: MLB', 
             transform=fig.transFigure, fontsize=18, 
             ha='left', va='top', color='#999999')
    
    plt.text(0.1, 0.89, 'By: @mlb_simulator', 
             transform=fig.transFigure, fontsize=18, 
             ha='left', va='top', color='#999999')
    
    # Save with high quality
    os.makedirs(images_dir, exist_ok=True)
    if table_type == "bottom":
        filename = f'all_games_worst_batted_balls_{date_str.replace(" ", "_")}.png'
    elif table_type == "luckiest":
        filename = f'all_games_luckiest_hits_{date_str.replace(" ", "_")}.png'
    elif table_type == "unluckiest":
        filename = f'all_games_unluckiest_outs_{date_str.replace(" ", "_")}.png'
    else:
        filename = f'all_games_best_batted_balls_{date_str.replace(" ", "_")}.png'
        
    plt.savefig(os.path.join(images_dir, filename), 
                bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Visualization saved to {os.path.join(images_dir, filename)}")


def apply_all_games_enhanced_styling(table, df, team_color_map):
    """
    Apply enhanced styling to all games table cells, matching individual game table style.
    Hides team text for logo placement.
    """
    
    # Column indices
    rank_col = 0
    team_col = 1
    player_col = 2
    bases_col = df.columns.get_loc('Estimated Bases')
    result_col = df.columns.get_loc('Result')
    xba_col = df.columns.get_loc('xBA')
    hr_col = df.columns.get_loc('HR%')
    game_col = df.columns.get_loc('Game')
    
    # Helper function for color brightness
    def is_dark_color(color):
        from matplotlib.colors import to_rgb
        r, g, b = to_rgb(color)
        return (r * 0.299 + g * 0.587 + b * 0.114) < 0.5
    
    # Style all cells
    for (row, col), cell in table.get_celld().items():
        
        # Base cell styling
        if row == 0:  # Header row
            cell.set_height(0.06)
            # Adjust font size for Team header since column is narrower
            if col == team_col:
                cell.set_text_props(weight='bold', fontsize=14)
            else:
                cell.set_text_props(weight='bold', fontsize=15)
            cell.set_facecolor('#2C3E50')
            cell.get_text().set_color('white')
            cell.set_edgecolor('#1A252F')
            cell.set_linewidth(2)
        else:  # Data rows
            cell.set_height(0.055)
            cell.set_text_props(fontsize=14)
            cell.set_edgecolor('#E0E0E0')
            cell.set_linewidth(0.5)
            
            # Alternating row colors for better readability
            if row % 2 == 0:
                cell.set_facecolor('#F8F9FA')
            else:
                cell.set_facecolor('#FFFFFF')
    
    # Apply special formatting for data rows
    for row in range(1, len(df) + 1):
        
        # Rank column - bold and centered
        rank_cell = table[(row, rank_col)]
        rank_cell.get_text().set_weight('bold')
        rank_cell.get_text().set_fontsize(14)
        
        # Team column - hide text for logo placement
        team_cell = table[(row, team_col)]
        # Use alternating row colors to match other columns
        if row % 2 == 0:
            team_cell.set_facecolor('#F8F9FA')
        else:
            team_cell.set_facecolor('#FFFFFF')
        # Make text transparent/invisible since we'll add logo
        team_cell.get_text().set_alpha(0)
        
        # Player names - readable size for full names
        player_cell = table[(row, player_col)]
        player_cell.get_text().set_fontsize(13)  # Slightly smaller to fit full names
        
        # Game column - smaller font for space
        game_cell = table[(row, game_col)]
        game_cell.get_text().set_fontsize(12)
        
        # Estimated bases - gradient coloring
        bases_value = df.iloc[row-1]['Estimated Bases']
        bases_cell = table[(row, bases_col)]
        
        # Create gradient from light yellow to dark orange/red
        norm = plt.Normalize(0, 4)
        cmap = plt.cm.YlOrRd
        color = cmap(norm(bases_value))
        bases_cell.set_facecolor(color)
        bases_cell.get_text().set_weight('bold')
        bases_cell.get_text().set_fontsize(14)
        if bases_value > 2.5:
            bases_cell.get_text().set_color('white')
        
        # Result column - color coding
        result = df.iloc[row-1]['Result']
        result_cell = table[(row, result_col)]
        if result == 'Out':
            result_cell.set_facecolor('#FFE5E5')
            result_cell.get_text().set_color('#D32F2F')
        elif result == 'Home Run':
            result_cell.set_facecolor('#E8F5E9')
            result_cell.get_text().set_color('#2E7D32')
        elif result in ['Single', 'Double', 'Triple']:
            result_cell.set_facecolor('#E3F2FD')
            result_cell.get_text().set_color('#1565C0')
        
        # xBA column - highlight high values
        xba_value = float(df.iloc[row-1]['xBA'])
        xba_cell = table[(row, xba_col)]
        if xba_value >= 0.500:
            xba_cell.get_text().set_weight('bold')
            xba_cell.get_text().set_color('#2E7D32')
        elif xba_value >= 0.300:
            xba_cell.get_text().set_color('#1565C0')


def add_logos_to_all_games_table(ax, table, team_names, mlb_team_logos, df):
    """
    Add team logos to the all games table at the appropriate cell positions.
    Uses the same approach as add_team_logos_to_table from individual game tables.
    """
    from matplotlib.offsetbox import AnnotationBbox
    
    team_col = 1  # Team column index
    
    # Force a draw to ensure table is fully rendered
    ax.figure.canvas.draw()
    
    # Get renderer
    renderer = ax.figure.canvas.get_renderer()
    
    # Calculate positions for each team logo
    for row_idx, team_name in enumerate(team_names[:15], start=1):  # Limit to 15 rows
        try:
            # Get the cell for this team
            cell = table[(row_idx, team_col)]
            
            # Get cell position in display coordinates
            cell_bbox = cell.get_window_extent(renderer)
            
            # Calculate center of cell in display coordinates
            cell_center_display = [
                (cell_bbox.x0 + cell_bbox.x1) / 2,
                (cell_bbox.y0 + cell_bbox.y1) / 2
            ]
            
            # Transform to axes coordinates (0-1 range)
            inv_transform = ax.transAxes.inverted()
            cell_center_axes = inv_transform.transform([cell_center_display])[0]
            
            # Get team logo URL
            logo_url = get_team_logo(team_name, mlb_team_logos)
            
            if logo_url:
                # Get the logo image with smaller size for better fit
                logo_size = (28, 28)  # Same size as individual game table
                img = getImage(logo_url, zoom=0.75, size=logo_size, alpha=1.0)
                
                if img:
                    # Create annotation box for the logo using axes coordinates
                    ab = AnnotationBbox(img, cell_center_axes,
                                      frameon=False,
                                      xycoords='axes fraction',
                                      box_alignment=(0.5, 0.5))
                    ax.add_artist(ab)
        except Exception as e:
            print(f"Error adding logo for {team_name} at row {row_idx}: {e}")
