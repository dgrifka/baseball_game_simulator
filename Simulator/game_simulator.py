import random
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from Simulator.constants import team_colors

# Load the pipeline
with open('Model/gb_classifier_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

def outcomes(game_data, home_or_away):
    """
    Extracts batting outcomes for either home or away team from game data.
    
    Args:
        game_data (pd.DataFrame): Complete game data
        home_or_away (str): Either 'home' or 'away' to filter team data
        
    Returns:
        list: Tuples of (outcome_data, event_type, batter_name)
    """
    home_or_away_team = game_data[game_data['isTopInning'] != (home_or_away == 'home')].copy()
    outcomes = []
    
    # Process automatic outs (strikeouts)
    mask = (home_or_away_team['eventType'] == 'out') & (home_or_away_team['hitData.launchSpeed'].isnull())
    outcomes.extend([("strikeout", row['eventType'], row['batter.fullName']) 
                    for _, row in home_or_away_team[mask].iterrows()])
    
    # Process walks
    mask = home_or_away_team['eventType'] == 'walk'
    outcomes.extend([("walk", row['eventType'], row['batter.fullName']) 
                    for _, row in home_or_away_team[mask].iterrows()])
    
    # Process balls in play
    mask = ~home_or_away_team['hitData.launchSpeed'].isnull()
    outcomes.extend([([row['hitData.launchSpeed'], row['hitData.launchAngle'], row['venue.name']], 
                     row['eventType'], row['batter.fullName']) 
                    for _, row in home_or_away_team[mask].iterrows()])
    
    return outcomes

def calculate_total_bases(outcomes):
    """
    Calculates total bases and outcome probabilities for each batting event.
    
    Args:
        outcomes (list): List of outcome tuples from outcomes()
        
    Returns:
        pd.DataFrame: Detailed statistics for each batting event
    """
    result_list = []
    
    for outcome, original_event_type, full_name in outcomes:
        if outcome == "strikeout":
            result = {'bases': 0, 'event_type': "strikeout", 
                     'probabilities': [1, 0, 0, 0, 0], 
                     'stats': [None, None, None]}
        elif outcome == "walk":
            result = {'bases': 1, 'event_type': "walk", 
                     'probabilities': [0, 1, 0, 0, 0], 
                     'stats': [None, None, None]}
        else:
            probabilities = pipeline.predict_proba(pd.DataFrame({
                'hitData_launchSpeed': [outcome[0]],
                'hitData_launchAngle': [outcome[1]],
                'venue_name': [outcome[2]]
            }))[0]
            
            bases = np.dot(probabilities[1:], [1, 2, 3, 4])
            result = {'bases': bases, 'event_type': "in_play", 
                     'probabilities': probabilities, 'stats': outcome}
            
        result_list.append({
            'player': full_name,
            'launch_speed': result['stats'][0] if result['stats'] else None,
            'launch_angle': result['stats'][1] if result['stats'] else None,
            'stadium': result['stats'][2] if result['stats'] else None,
            'event_type': result['event_type'],
            'original_event_type': original_event_type,
            'estimated_bases': result['bases'],
            'out_prob': result['probabilities'][0],
            'single_prob': result['probabilities'][1],
            'double_prob': result['probabilities'][2],
            'triple_prob': result['probabilities'][3],
            'hr_prob': result['probabilities'][4]
        })
    
    return pd.DataFrame(result_list)

def create_detailed_outcomes_df(game_data, home_or_away):
    """
    Creates a detailed DataFrame of batting outcomes for specified team.
    
    Args:
        game_data (pd.DataFrame): Complete game data
        home_or_away (str): Either 'home' or 'away' to filter team data
        
    Returns:
        pd.DataFrame: Detailed batting outcomes
    """
    return calculate_total_bases(outcomes(game_data, home_or_away)).dropna().reset_index(drop=True)

def outcome_rankings(home_detailed_df, away_detailed_df):
    """
    Ranks batting outcomes by estimated bases and formats for display.
    
    Args:
        home_detailed_df (pd.DataFrame): Home team detailed outcomes
        away_detailed_df (pd.DataFrame): Away team detailed outcomes
        
    Returns:
        pd.DataFrame: Top 10 outcomes ranked by estimated bases
    """
    total_team_outcomes = pd.concat([home_detailed_df, away_detailed_df])
    
    # Format columns
    total_team_outcomes['launch_angle'] = total_team_outcomes['launch_angle'].astype(int)
    total_team_outcomes['estimated_bases'] = total_team_outcomes['estimated_bases'].round(2)
    
    prob_columns = [col for col in total_team_outcomes.columns if '_prob' in col]
    total_team_outcomes[prob_columns] = (total_team_outcomes[prob_columns] * 100).round(0).astype(int).astype(str) + '%'
    
    # Select and rename columns
    selected_columns = ['team', 'player', 'launch_speed', 'launch_angle', 
                       "original_event_type", 'estimated_bases', 
                       "out_prob", "single_prob", "double_prob", "triple_prob", "hr_prob"]
    
    total_team_outcomes = (total_team_outcomes[selected_columns]
                          .assign(original_event_type=lambda x: x['original_event_type'].str.title())
                          .rename(columns=lambda x: x.replace('_', ' ').title())
                          .rename(columns={'Original Event Type': 'Result', 'Full Name': 'Player'}))
    
    # Add team colors and sort
    total_team_outcomes['team_color'] = total_team_outcomes['Team'].map({team: colors[0] for team, colors in team_colors.items()})
    
    return total_team_outcomes.nlargest(10, 'Estimated Bases')

def advance_runner(bases, count=1, is_walk=False):
    """
    Advances runners on base according to the play outcome.
    
    Args:
        bases (list): Current base state [first, second, third]
        count (int): Number of bases to advance
        is_walk (bool): Whether the advance is due to a walk
        
    Returns:
        int: Number of runs scored
    """
    runs = 0
    bases = bases.copy()
    
    if is_walk:
        # Force in run if bases loaded
        if all(bases):
            runs += 1
        # Shift runners
        for i in range(len(bases)-1, 0, -1):
            bases[i] = bases[i-1]
        bases[0] = True
    else:
        # Score runs and advance runners
        for _ in range(count):
            if bases[2]:  # Runner on third scores
                runs += 1
            # Shift runners
            bases[2] = bases[1]
            bases[1] = bases[0]
            bases[0] = True
            
    return runs

def simulate_game(outcomes_df):
    """
    Simulates a single game based on batting outcomes.
    
    Args:
        outcomes_df (pd.DataFrame): Batting outcomes to simulate
        
    Returns:
        int: Runs scored in simulation
    """
    outs = 0
    runs = 0
    bases = [False] * 3
    outcomes_copy = outcomes_df.copy()
    
    # Pre-calculate probabilities
    probabilities_dict = {tuple(outcome): pipeline.predict_proba(pd.DataFrame({
        'hitData_launchSpeed': [outcome[0]],
        'hitData_launchAngle': [outcome[1]],
        'venue_name': [outcome[2]]
    }))[0] for outcome in outcomes_copy if isinstance(outcome, list)}
    
    while outcomes_copy:
        if outs == 3:
            outs = 0
            bases = [False] * 3
            
        outcome = outcomes_copy.pop(random.randrange(len(outcomes_copy)))
        
        if outcome == "strikeout":
            outs += 1
        elif outcome == "walk":
            runs += advance_runner(bases, is_walk=True)
        else:
            prob = probabilities_dict[tuple(outcome)]
            rand_val = random.random()
            cumsum_prob = np.cumsum(prob)
            
            for i, threshold in enumerate(cumsum_prob):
                if rand_val < threshold:
                    if i == 0:
                        outs += 1
                    else:
                        runs += advance_runner(bases, i)
                    break
    
    return runs

def simulator(num_simulations, home_outcomes, away_outcomes):
    """
    Runs multiple game simulations and calculates win probabilities.
    
    Args:
        num_simulations (int): Number of games to simulate
        home_outcomes (list): Home team batting outcomes
        away_outcomes (list): Away team batting outcomes
        
    Returns:
        tuple: Arrays of runs scored and win/loss percentages
    """
    results = np.zeros((num_simulations, 2), dtype=int)
    
    for i in tqdm(range(num_simulations), desc="Simulating games", unit="sim"):
        results[i] = [simulate_game(home_outcomes), simulate_game(away_outcomes)]
    
    home_wins = np.sum(results[:, 0] > results[:, 1])
    away_wins = np.sum(results[:, 0] < results[:, 1])
    ties = num_simulations - home_wins - away_wins
    
    return (results[:, 0], results[:, 1],
            home_wins * 100 / num_simulations,
            away_wins * 100 / num_simulations,
            ties * 100 / num_simulations)
