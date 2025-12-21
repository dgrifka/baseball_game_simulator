"""
MLB game simulation engine.
Simulates batted ball outcomes using a gradient boosting model trained on Statcast data.
"""

import random
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm

from Simulator.constants import team_colors, VENUE_NAME_TO_ID, DEFAULT_VENUE_ID
from Model.feature_engineering import (
    create_features_for_prediction,
    create_features_for_prediction_fallback
)

# =============================================================================
# LOAD MODEL
# =============================================================================

# Load the trained pipeline
pipeline = joblib.load('Model/batted_ball_model.pkl')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_venue_id(venue_name):
    """
    Convert venue name to venue ID for model prediction.
    
    Args:
        venue_name (str): Stadium name
    
    Returns:
        str: Venue ID as string
    """
    return VENUE_NAME_TO_ID.get(venue_name, DEFAULT_VENUE_ID)


def prepare_batted_ball_features(launch_speed, launch_angle, venue_name, 
                                  coord_x=None, coord_y=None, bat_side=None):
    """
    Prepare features for model prediction, handling missing spray data gracefully.
    
    Args:
        launch_speed (float): Exit velocity in mph
        launch_angle (float): Launch angle in degrees
        venue_name (str): Stadium name
        coord_x (float, optional): Hit coordinates X
        coord_y (float, optional): Hit coordinates Y
        bat_side (str, optional): 'L' or 'R' for batter handedness
    
    Returns:
        pd.DataFrame: Features ready for model prediction
    """
    venue_id = get_venue_id(venue_name)
    
    # Check if we have spray angle data
    has_spray_data = (
        coord_x is not None and 
        coord_y is not None and 
        bat_side is not None and
        not pd.isna(coord_x) and 
        not pd.isna(coord_y)
    )
    
    if has_spray_data:
        return create_features_for_prediction(
            launch_speed=launch_speed,
            launch_angle=launch_angle,
            coord_x=coord_x,
            coord_y=coord_y,
            bat_side=bat_side,
            venue_id=venue_id
        )
    else:
        # Fallback: use neutral spray angle
        return create_features_for_prediction_fallback(
            launch_speed=launch_speed,
            launch_angle=launch_angle,
            venue_id=venue_id
        )


# =============================================================================
# OUTCOME EXTRACTION
# =============================================================================

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
              dict with batted ball data
    """
    home_or_away_team = game_data.copy()
    if home_or_away == 'home':
        home_or_away_team = home_or_away_team[home_or_away_team['isTopInning'] == False]
        baserunning_events = steals_and_pickoffs[steals_and_pickoffs['isTopInning'] == False]
    else:
        home_or_away_team = home_or_away_team[home_or_away_team['isTopInning'] == True]
        baserunning_events = steals_and_pickoffs[steals_and_pickoffs['isTopInning'] == True]
    
    outcomes_list = []
    
    # Process batting outcomes
    automatic_outs = home_or_away_team[(home_or_away_team['eventType'] == 'out') & 
                                     (home_or_away_team['hitData.launchSpeed'].isnull())]
    for _, row in automatic_outs.iterrows():
        outcomes_list.append(("strikeout", row['eventType'], row['batter.fullName']))
        
    walks = home_or_away_team[home_or_away_team['eventType'] == 'walk']
    for _, row in walks.iterrows():
        outcomes_list.append(("walk", row['eventType'], row['batter.fullName']))
        
    put_in_play = home_or_away_team[~home_or_away_team['hitData.launchSpeed'].isnull()].reset_index(drop=True)
    for _, row in put_in_play.iterrows():
        # Create dict with all batted ball data (including spray angle fields)
        batted_ball_data = {
            'launch_speed': row['hitData.launchSpeed'],
            'launch_angle': row['hitData.launchAngle'],
            'venue_name': row['venue.name'],
            'coord_x': row.get('hitData.coordinates.coordX'),
            'coord_y': row.get('hitData.coordinates.coordY'),
            'bat_side': row.get('batSide.code')  # Flattened column name
        }
        outcomes_list.append((batted_ball_data, row['eventType'], row['batter.fullName']))
    
    # Process baserunning events
    if not baserunning_events.empty:
        for _, row in baserunning_events.iterrows():
            outcomes_list.append((row['play'], row['play'], row['batter.fullName']))
    
    return outcomes_list


def calculate_total_bases(outcomes_list):
    """
    Calculate expected bases and outcome probabilities for each batting/baserunning event.
    
    Args:
        outcomes_list (list): List of outcome tuples from outcomes()
        
    Returns:
        pd.DataFrame: Detailed stats including launch data, probabilities, and estimated bases
    """
    result_list = []
    
    for outcome, original_event_type, full_name in outcomes_list:
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
        elif isinstance(outcome, dict):
            # Batted ball with full data
            launch_speed = outcome['launch_speed']
            launch_angle = outcome['launch_angle']
            stadium = outcome['venue_name']
            event_type = "in_play"
            
            # Create features and predict
            features = prepare_batted_ball_features(
                launch_speed=launch_speed,
                launch_angle=launch_angle,
                venue_name=stadium,
                coord_x=outcome.get('coord_x'),
                coord_y=outcome.get('coord_y'),
                bat_side=outcome.get('bat_side')
            )
            probabilities = pipeline.predict_proba(features)[0]
            
            bases = (
                probabilities[1] * 1 +
                probabilities[2] * 2 +
                probabilities[3] * 3 +
                probabilities[4] * 4
            )
        else:
            # Legacy format: list [launch_speed, launch_angle, venue]
            launch_speed, launch_angle, stadium = outcome
            event_type = "in_play"
            
            features = prepare_batted_ball_features(
                launch_speed=launch_speed,
                launch_angle=launch_angle,
                venue_name=stadium
            )
            probabilities = pipeline.predict_proba(features)[0]
            
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
            'hr_prob': probabilities[4],
            'coord_x': outcome.get('coord_x') if isinstance(outcome, dict) else None,
            'coord_y': outcome.get('coord_y') if isinstance(outcome, dict) else None,
            'bat_side': outcome.get('bat_side') if isinstance(outcome, dict) else None
        })
    
    return pd.DataFrame(result_list)


def create_detailed_outcomes_df(game_data, steals_and_pickoffs, home_or_away):
    """
    Create a detailed DataFrame of batting outcomes for specified team.
    
    Args:
        game_data (pd.DataFrame): Original game data
        steals_and_pickoffs (pd.DataFrame): Steals and pickoffs data
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
    total_team_outcomes = total_team_outcomes.rename(columns={
        "launch_speed": "EV", "launch_angle": "LA", "original_event_type": "Result", 
        "out_prob": "Out%", "single_prob": "1B%", "double_prob": "2B%", 
        "triple_prob": "3B%", "hr_prob": "HR%", "estimated_bases": "xBases", "team": "Team"
    })
    total_team_outcomes.columns = [col.title() for col in total_team_outcomes.columns]
    return total_team_outcomes.sort_values(by='Xbases', ascending=False).head(15).reset_index(drop=True)


# =============================================================================
# BASERUNNING
# =============================================================================

def attempt_steal(bases):
    """
    Attempt steal of next base for lead runner.
    
    Args:
        bases (list): Current base occupancy [1st, 2nd, 3rd]
        
    Returns:
        int: Runs scored (1 if runner on 3rd steals home, else 0)
    """
    if bases[2]:  # Steal home (rare)
        bases[2] = False
        return 1
    elif bases[1]:  # Steal third
        bases[2] = True
        bases[1] = False
    elif bases[0]:  # Steal second
        bases[1] = True
        bases[0] = False
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
    runs = 0
    
    if is_walk:
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
    
    elif count == 4:  # Home run
        runs = sum(bases) + 1
        bases[0] = False
        bases[1] = False
        bases[2] = False
    
    else:  # Singles, doubles, triples
        original_bases = bases.copy()
        bases[0] = bases[1] = bases[2] = False
        
        for i in range(2, -1, -1):  # Work backwards from 3rd to 1st
            if original_bases[i]:
                advancement = count
                
                # Probabilistic advancements
                if count == 1 and i == 1:  # Single with runner on 2nd
                    advancement = 2 if random.random() < 0.5 else 1
                elif count == 1 and i == 0 and not original_bases[1]:
                    advancement = 2 if random.random() < 0.25 else 1
                elif count == 2 and i == 0:  # Double with runner on 1st
                    advancement = 3 if random.random() < 0.75 else 2
                
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


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def simulate_game(outcomes_list, prob_cache):
    """
    Simulate one game using pre-computed probabilities.
    
    Args:
        outcomes_list (list): List of outcome data
        prob_cache (dict): Pre-computed probabilities keyed by outcome
    
    Returns:
        int: Runs scored in simulated game
    """
    outs = 0
    runs = 0
    bases = [False, False, False]
    
    n_outcomes = len(outcomes_list)
    indices = np.random.permutation(n_outcomes)
    
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
        elif isinstance(outcome, (dict, tuple)):
            # Get cache key
            if isinstance(outcome, dict):
                cache_key = (outcome['launch_speed'], outcome['launch_angle'], outcome['venue_name'])
            else:
                cache_key = outcome
            
            probabilities = prob_cache.get(cache_key)
            if probabilities is None:
                continue
                
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
    Run multiple game simulations with pre-computed probabilities.
    
    Args:
        num_simulations (int): Number of games to simulate
        home_outcomes (list): List of home team batting outcomes
        away_outcomes (list): List of away team batting outcomes
        
    Returns:
        tuple: (home_runs_array, away_runs_array, home_win_pct, away_win_pct, tie_pct)
    """
    # Clean up outcomes format
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
        if isinstance(outcome, dict):
            cache_key = (outcome['launch_speed'], outcome['launch_angle'], outcome['venue_name'])
            if cache_key not in prob_cache:
                features = prepare_batted_ball_features(
                    launch_speed=outcome['launch_speed'],
                    launch_angle=outcome['launch_angle'],
                    venue_name=outcome['venue_name'],
                    coord_x=outcome.get('coord_x'),
                    coord_y=outcome.get('coord_y'),
                    bat_side=outcome.get('bat_side')
                )
                prob_cache[cache_key] = pipeline.predict_proba(features)[0]
        elif isinstance(outcome, list) and len(outcome) == 3:
            # Legacy format
            cache_key = tuple(outcome)
            if cache_key not in prob_cache:
                launch_speed, launch_angle, stadium = outcome
                features = prepare_batted_ball_features(
                    launch_speed=launch_speed,
                    launch_angle=launch_angle,
                    venue_name=stadium
                )
                prob_cache[cache_key] = pipeline.predict_proba(features)[0]
    
    # Initialize arrays for results
    home_runs_scored = np.zeros(num_simulations, dtype=int)
    away_runs_scored = np.zeros(num_simulations, dtype=int)
    
    # Run simulations with progress bar
    for i in tqdm(range(num_simulations), 
                  desc="Simulating games", 
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
