import random
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from Simulator.constants import team_colors

# Load the pipeline
with open('Model/gb_classifier_pipeline_improved.pkl', 'rb') as file:
    pipeline = pickle.load(file)

def create_features_for_prediction(launch_speed, launch_angle, stadium):
    """
    Create all required features for the improved model prediction.
    
    Args:
        launch_speed (float): Exit velocity in mph
        launch_angle (float): Launch angle in degrees
        stadium (str): Venue name
        
    Returns:
        pd.DataFrame: DataFrame with all required features
    """
    # Convert angle to radians for distance calculation
    angle_rad = np.radians(launch_angle)
    
    # Calculate distance proxy
    distance_proxy = launch_speed * np.sin(2 * angle_rad)
    
    # Categorize launch angle
    if launch_angle < 10:
        launch_angle_category = 'ground_ball'
    elif launch_angle < 25:
        launch_angle_category = 'line_drive'
    elif launch_angle < 50:
        launch_angle_category = 'fly_ball'
    else:
        launch_angle_category = 'popup'
    
    # Barrel zone indicator
    is_barrel = int((launch_speed >= 95) & (launch_angle >= 25) & (launch_angle <= 35))
    
    # Create DataFrame with all features
    return pd.DataFrame({
        'hitData_launchSpeed': [launch_speed],
        'hitData_launchAngle': [launch_angle],
        'distance_proxy': [distance_proxy],
        'launch_angle_category': [launch_angle_category],
        'is_barrel': [is_barrel],
        'venue_name': [stadium]
    })

def outcomes(game_data, steals_and_pickoffs, home_or_away):
    """
    Extract batting outcomes and baserunning events from game data for either home or away team.
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game events
        steals_and_pickoffs (pd.DataFrame): DataFrame containing stolen base and pickoff events
        home_or_away (str): 'home' or 'away' to filter team data
        
    Returns:
        list: Tuples of (outcome_data, event_type, player_name) where outcome_data is either
              a string ('strikeout'/'walk'/'stolen_base'/'pickoff') or 
              list [launch_speed, launch_angle, venue]
    """
    home_or_away_team = game_data.copy()
    if home_or_away == 'home':
        home_or_away_team = home_or_away_team[home_or_away_team['isTopInning'] == False]
        baserunning_events = steals_and_pickoffs[steals_and_pickoffs['isTopInning'] == False]
    else:
        home_or_away_team = home_or_away_team[home_or_away_team['isTopInning'] == True]
        baserunning_events = steals_and_pickoffs[steals_and_pickoffs['isTopInning'] == True]
    
    outcomes = []
    
    # Process batting outcomes
    automatic_outs = home_or_away_team[(home_or_away_team['eventType'] == 'out') & 
                                     (home_or_away_team['hitData.launchSpeed'].isnull())]
    for _, row in automatic_outs.iterrows():
        outcomes.append(("strikeout", row['eventType'], row['batter.fullName']))
        
    walks = home_or_away_team[home_or_away_team['eventType'] == 'walk']
    for _, row in walks.iterrows():
        outcomes.append(("walk", row['eventType'], row['batter.fullName']))
        
    put_in_play = home_or_away_team[~home_or_away_team['hitData.launchSpeed'].isnull()].reset_index(drop=True)
    for _, row in put_in_play.iterrows():
        outcomes.append(([row['hitData.launchSpeed'], row['hitData.launchAngle'], row['venue.name']], 
                       row['eventType'], row['batter.fullName']))
    
    # Process baserunning events
    if not baserunning_events.empty:
        for _, row in baserunning_events.iterrows():
            outcomes.append((row['play'], row['play'], row['batter.fullName']))
    
    # Sort outcomes by their original order
    return outcomes

def calculate_total_bases(outcomes):
    """
    Calculate expected bases and outcome probabilities for each batting/baserunning event.
    
    Args:
        outcomes (list): List of outcome tuples from outcomes()
        
    Returns:
        pd.DataFrame: Detailed stats including launch data, probabilities, and estimated bases
    """
    result_list = []
    for outcome, original_event_type, full_name in outcomes:
        if outcome == "strikeout":
            bases = 0
            event_type = "strikeout"
            probabilities = [1, 0, 0, 0, 0]
            launch_speed, launch_angle, stadium = None, None, None
        elif outcome == "walk":
            bases = 1
            event_type = "walk"
            probabilities = [0, 1, 0, 0, 0]
            launch_speed, launch_angle, stadium = None, None, None
        elif outcome == "stolen_base":
            bases = 1
            event_type = "stolen_base"
            probabilities = [0, 1, 0, 0, 0]
            launch_speed, launch_angle, stadium = None, None, None
        elif outcome == "pickoff":
            bases = 0
            event_type = "pickoff"
            probabilities = [1, 0, 0, 0, 0]
            launch_speed, launch_angle, stadium = None, None, None
        else:
            launch_speed, launch_angle, stadium = outcome
            event_type = "in_play"
            
            # Use the new feature creation function
            new_example = create_features_for_prediction(launch_speed, launch_angle, stadium)
            probabilities = pipeline.predict_proba(new_example)[0]
            
            bases = (
                probabilities[1] * 1 +
                probabilities[2] * 2 +
                probabilities[3] * 3 +
                probabilities[4] * 4
            )
        
        result_list.append({
            'player': full_name,
            'launch_speed': launch_speed,
            'launch_angle': launch_angle,
            'stadium': stadium,
            'event_type': event_type,
            'original_event_type': original_event_type,
            'estimated_bases': bases,
            'out_prob': probabilities[0],
            'single_prob': probabilities[1],
            'double_prob': probabilities[2],
            'triple_prob': probabilities[3],
            'hr_prob': probabilities[4]
        })
    
    return pd.DataFrame(result_list)

def create_detailed_outcomes_df(game_data, steals_and_pickoffs, home_or_away):
    """
    Create a detailed DataFrame of batting outcomes for specified team.
    
    Args:
        game_data (pd.DataFrame): Original game data
        home_or_away (str): 'home' or 'away' team selection
        
    Returns:
        pd.DataFrame: Processed and cleaned batting outcomes with probabilities
    """
    outcomes_list = outcomes(game_data, steals_and_pickoffs, home_or_away)
    detailed_df = calculate_total_bases(outcomes_list)
    # Filter out stolen bases and pickoffs
    detailed_df = detailed_df[~detailed_df['event_type'].isin(['stolen_base', 'pickoff'])]
    detailed_df = detailed_df.dropna().reset_index(drop=True)
    return detailed_df

def outcome_rankings(home_detailed_df, away_detailed_df):
    """
    Combine and rank batting outcomes from both teams by estimated bases.
    
    Args:
        home_detailed_df (pd.DataFrame): Home team detailed outcomes
        away_detailed_df (pd.DataFrame): Away team detailed outcomes
        
    Returns:
        pd.DataFrame: Top 10 outcomes ranked by estimated bases, formatted for display
    """
    # Combine team outcomes, excluding any remaining stolen bases or pickoffs
    home_detailed_df = home_detailed_df[~home_detailed_df['event_type'].isin(['stolen_base', 'pickoff'])]
    away_detailed_df = away_detailed_df[~away_detailed_df['event_type'].isin(['stolen_base', 'pickoff'])]
    
    total_team_outcomes = pd.concat([home_detailed_df, away_detailed_df])
    total_team_outcomes['launch_angle'] = total_team_outcomes['launch_angle'].astype(int)
    total_team_outcomes['estimated_bases'] = total_team_outcomes['estimated_bases'].round(2)
    
    prob_columns = [col for col in total_team_outcomes.columns if '_prob' in col]
    for col in prob_columns:
        total_team_outcomes[col] = (total_team_outcomes[col] * 100).round(0).astype(int).astype(str) + '%'
    
    selected_columns = ['team', 'player', 'launch_speed', 'launch_angle', "original_event_type", 'estimated_bases', 
                       "out_prob", "single_prob", "double_prob", "triple_prob", "hr_prob"]
    total_team_outcomes = total_team_outcomes[selected_columns]
    total_team_outcomes['original_event_type'] = total_team_outcomes['original_event_type'].str.title()
    total_team_outcomes.columns = total_team_outcomes.columns.str.replace('_', ' ').str.title()
    total_team_outcomes = total_team_outcomes.rename(columns={'Original Event Type': 'Result', 'Full Name': 'Player'})
    total_team_outcomes['team_color'] = total_team_outcomes['Team'].map({team: colors[0] for team, colors in team_colors.items()})
    return total_team_outcomes.sort_values(by='Estimated Bases', ascending=False).head(15)

def attempt_steal(bases):
    """
    Attempt to steal with the lead runner.
    
    Args:
        bases (list): Current base occupancy [1st, 2nd, 3rd]
        
    Returns:
        int: Runs scored (1 if stealing home successfully, 0 otherwise)
    """
    if bases[2]:  # Runner on third attempts to steal home
        bases[2] = False
        return 1
    elif bases[1]:  # Runner on second advances to third
        bases[2] = True
        bases[1] = False
        return 0
    elif bases[0]:  # Runner on first advances to second
        bases[1] = True
        bases[0] = False
        return 0
    return 0

def attempt_pickoff(bases):
    """
    Attempt pickoff of lead runner.
    
    Args:
        bases (list): Current base occupancy [1st, 2nd, 3rd]
        
    Returns:
        int: 1 if pickoff successful, 0 otherwise
    """
    if bases[2]:  # Pick off runner on third
        bases[2] = False
    elif bases[1]:  # Pick off runner on second
        bases[1] = False
    elif bases[0]:  # Pick off runner on first
        bases[0] = False
    return 1

def advance_runner(bases, count=1, is_walk=False):
    """
    Calculate runs scored and update base runners after a hit/walk.
    
    Args:
        bases (list): Current base occupancy [1st, 2nd, 3rd]
        count (int): Number of bases advanced (1-4)
        is_walk (bool): True if event is a walk
        
    Returns:
        int: Runs scored on this play
    """
    import random
    runs = 0
    
    if is_walk:
        # Walk logic stays the same
        if bases[2] and bases[1] and bases[0]:
            runs += 1
            bases[2] = bases[1]
            bases[1] = bases[0]
            bases[0] = True
        elif bases[1] and bases[0]:
            bases[2] = bases[1]
            bases[1] = bases[0]
            bases[0] = True
        elif bases[0]:
            bases[1] = bases[0]
            bases[0] = True
        else:
            bases[0] = True
    
    elif count == 4:  # Home run - special case
        # Count all runners on base plus the batter
        runs = sum(bases) + 1
        # Clear all bases
        bases[0] = False
        bases[1] = False
        bases[2] = False
    
    else:  # Singles, doubles, triples
        # Store original state to avoid conflicts
        original_bases = bases.copy()
        bases[0] = bases[1] = bases[2] = False
        
        # Process each runner with probabilistic advancements
        for i in range(2, -1, -1):  # Work backwards from 3rd to 1st
            if original_bases[i]:
                # Calculate advancement with special cases
                advancement = count
                
                # Special probabilistic advancements
                if count == 1 and i == 1:  # Single with runner on 2nd
                    advancement = 2 if random.random() < 0.5 else 1  # 50% scores
                elif count == 1 and i == 0 and not original_bases[1]:  # Single with runner on 1st (only if 2nd empty)
                    advancement = 2 if random.random() < 0.25 else 1  # 25% to 3rd
                elif count == 2 and i == 0:  # Double with runner on 1st
                    advancement = 3 if random.random() < 0.75 else 2  # 75% scores
                
                # Apply advancement
                new_position = i + advancement
                if new_position >= 3:
                    runs += 1
                else:
                    bases[new_position] = True
        
        # Put batter on base
        if count == 1:
            bases[0] = True
        elif count == 2:
            bases[1] = True
        elif count == 3:
            bases[2] = True
    
    return runs

def simulate_game(outcomes_list, prob_cache):
    """
    Simulate one game's worth of plate appearances and calculate runs scored using pre-computed probabilities.
    """
    outs = 0
    runs = 0
    bases = [False, False, False]
    
    # Use numpy for faster random sampling WITHOUT replacement
    n_outcomes = len(outcomes_list)
    indices = np.random.permutation(n_outcomes)  # Shuffle indices
    
    for idx in indices:
        if outs == 3:
            outs = 0
            bases = [False, False, False]
        
        outcome = outcomes_list[idx]
        
        if outcome == "strikeout":
            outs += 1
        elif outcome == "walk":
            runs += advance_runner(bases, is_walk=True)
        elif outcome == "stolen_base":
            if any(bases):
                runs += attempt_steal(bases)
        elif outcome == "pickoff":
            if any(bases):
                outs += attempt_pickoff(bases)
        elif isinstance(outcome, list) and len(outcome) == 3:
            # Use pre-computed probabilities
            probabilities = prob_cache[tuple(outcome)]
            random_value = random.random()
            
            if random_value < probabilities[0]:
                outs += 1
            elif random_value < probabilities[0] + probabilities[1]:
                runs += advance_runner(bases, 1)
            elif random_value < probabilities[0] + probabilities[1] + probabilities[2]:
                runs += advance_runner(bases, 2)
            elif random_value < probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3]:
                runs += advance_runner(bases, 3)
            else:
                runs += advance_runner(bases, 4)
    
    return runs

def simulator(num_simulations, home_outcomes, away_outcomes):
    """
    Simulator with pre-computed probabilities.
    
    Args:
        num_simulations (int): Number of games to simulate
        home_outcomes (list): List of possible home team batting outcomes
        away_outcomes (list): List of possible away team batting outcomes
        
    Returns:
        tuple: (home_runs_array, away_runs_array, home_win_pct, away_win_pct, tie_pct)
    """
    # Clean up outcomes format (handle tuples from original format)
    home_outcomes_clean = []
    away_outcomes_clean = []
    
    for outcome in home_outcomes:
        if isinstance(outcome, tuple):
            home_outcomes_clean.append(outcome[0])
        else:
            home_outcomes_clean.append(outcome)
    
    for outcome in away_outcomes:
        if isinstance(outcome, tuple):
            away_outcomes_clean.append(outcome[0])
        else:
            away_outcomes_clean.append(outcome)
    
    # Pre-compute ALL probabilities ONCE before simulations
    prob_cache = {}
    all_outcomes = home_outcomes_clean + away_outcomes_clean
    
    for outcome in all_outcomes:
        if isinstance(outcome, list) and len(outcome) == 3:
            key = tuple(outcome)
            if key not in prob_cache:
                launch_speed, launch_angle, stadium = outcome
                # Use the new feature creation function
                new_example = create_features_for_prediction(launch_speed, launch_angle, stadium)
                prob_cache[key] = pipeline.predict_proba(new_example)[0]
    
    # Initialize arrays for results
    home_runs_scored = np.zeros(num_simulations, dtype=int)
    away_runs_scored = np.zeros(num_simulations, dtype=int)
    
    # Run simulations with progress bar
    for i in tqdm(range(num_simulations), 
                  desc="Simulating games (fast)", 
                  unit="sim",
                  position=0,
                  leave=True,
                  ncols=80,
                  ascii=True):
        home_runs_scored[i] = simulate_game(home_outcomes_clean, prob_cache)
        away_runs_scored[i] = simulate_game(away_outcomes_clean, prob_cache)
    
    # Calculate win percentages
    home_wins = np.sum(home_runs_scored > away_runs_scored)
    away_wins = np.sum(home_runs_scored < away_runs_scored)
    ties = np.sum(home_runs_scored == away_runs_scored)
    
    home_win_percentage = home_wins / num_simulations * 100
    away_win_percentage = away_wins / num_simulations * 100
    tie_percentage = ties / num_simulations * 100
    
    return home_runs_scored, away_runs_scored, home_win_percentage, away_win_percentage, tie_percentage
