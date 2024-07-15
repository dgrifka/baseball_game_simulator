import random
import pickle
import pandas as pd
import numpy as np

from constants import team_colors

# Load the saved model and fitted preprocessor
with open('logistic_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

def outcomes(game_data, home_or_away):
    home_or_away_team = game_data.copy()
    if home_or_away == 'home':
        home_or_away_team = home_or_away_team[home_or_away_team['isTopInning'] == False]
    else:
        home_or_away_team = home_or_away_team[home_or_away_team['isTopInning'] == True]
    
    outcomes = []

    # Handle automatic outs (strikeouts)
    automatic_outs = home_or_away_team[(home_or_away_team['eventType'] == 'out') & (home_or_away_team['hitData.launchSpeed'].isnull())]
    for _, row in automatic_outs.iterrows():
        outcomes.append(("strikeout", row['eventType'], row['batter.fullName']))

    # Handle walks
    walks = home_or_away_team[home_or_away_team['eventType'] == 'walk']
    for _, row in walks.iterrows():
        outcomes.append(("walk", row['eventType'], row['batter.fullName']))

    # Handle balls put in play
    put_in_play = home_or_away_team[~home_or_away_team['hitData.launchSpeed'].isnull()].reset_index(drop=True)
    for _, row in put_in_play.iterrows():
        outcomes.append(([row['hitData.launchSpeed'], row['hitData.launchAngle'], row['venue.name']], row['eventType'], row['batter.fullName']))

    return outcomes

def calculate_total_bases(outcomes):
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
        else:
            launch_speed, launch_angle, stadium = outcome
            event_type = "in_play"
            # Create a DataFrame with the new example
            new_example = pd.DataFrame({
                'hitData_launchSpeed': [launch_speed],
                'hitData_launchAngle': [launch_angle],
                'venue_name': [stadium]
            })
            # Preprocess the new example using the loaded preprocessor
            new_example_preprocessed = preprocessor.transform(new_example)
            # Get predicted probabilities
            probabilities = loaded_model.predict_proba(new_example_preprocessed)[0]
            # Calculate bases for the current outcome
            bases = (
                probabilities[1] * 1 +  # Single
                probabilities[2] * 2 +  # Double
                probabilities[3] * 3 +  # Triple
                probabilities[4] * 4    # Home Run
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

def create_detailed_outcomes_df(game_data, home_or_away):
    # Get outcomes
    outcomes_list = outcomes(game_data, home_or_away)
    
    # Calculate total bases and get detailed DataFrame
    detailed_df = calculate_total_bases(outcomes_list)
    detailed_df = detailed_df.dropna().reset_index(drop=True)

    return detailed_df

def outcome_rankings(home_detailed_df, away_detailed_df, luck_type):
    # Create total dataframe
    total_team_outcomes = pd.concat([home_detailed_df, away_detailed_df])

    # Filter based on luck_type
    if luck_type == "unlucky":
        total_team_outcomes = total_team_outcomes[total_team_outcomes['original_event_type'] == 'out']
        total_team_outcomes = total_team_outcomes.sort_values(by='estimated_bases', ascending=False).reset_index(drop=True)
    elif luck_type == "lucky":
        total_team_outcomes = total_team_outcomes[total_team_outcomes['original_event_type'] != 'out']
        total_team_outcomes = total_team_outcomes.sort_values(by='estimated_bases', ascending=True).reset_index(drop=True)

    # Convert launch_angle to int and round estimated_bases
    total_team_outcomes['launch_angle'] = total_team_outcomes['launch_angle'].astype(int)
    total_team_outcomes['estimated_bases'] = total_team_outcomes['estimated_bases'].round(2)

    # Clean up probability columns
    prob_columns = [col for col in total_team_outcomes.columns if '_prob' in col]
    for col in prob_columns:
        total_team_outcomes[col] = (total_team_outcomes[col] * 100).round(0).astype(int).astype(str) + '%'

    # Select top 10 rows and relevant columns
    total_team_outcomes = total_team_outcomes.head(7)
    selected_columns = ['team', 'player', 'launch_speed', 'launch_angle', "original_event_type", 'estimated_bases', "out_prob", "single_prob", "double_prob", "triple_prob", "hr_prob"]
    total_team_outcomes = total_team_outcomes[selected_columns]

    # Capitalize original_event_type values
    total_team_outcomes['original_event_type'] = total_team_outcomes['original_event_type'].str.title()

    # Rename columns
    total_team_outcomes.columns = total_team_outcomes.columns.str.replace('_', ' ').str.title()
    total_team_outcomes = total_team_outcomes.rename(columns={'Original Event Type': 'Result', 'Full Name': 'Player'})

    # Extract the team names from the lucky_outcomes DataFrame
    # Assuming 'Team' column exists in lucky_outcomes DataFrame
    team_names = total_team_outcomes['Team']
    
    # Create a dictionary mapping team names to their corresponding colors
    color_mapping = {team: colors[0] for team, colors in team_colors.items()}
    
    # Add the 'team_color' column to the lucky_outcomes DataFrame
    total_team_outcomes['team_color'] = total_team_outcomes['Team'].map(color_mapping)
    return total_team_outcomes
    
def simulate_game(outcomes_df):
    outs = 0
    runs = 0
    bases = [False, False, False]  # First, Second, Third base
    
    outcomes_copy = outcomes_df.copy()  # Create a copy of the outcomes list
    
    # Calculate probabilities for each outcome before the loop
    probabilities_dict = {}
    for outcome in outcomes_copy:
        if isinstance(outcome, list) and len(outcome) == 3:
            launch_speed, launch_angle, stadium = outcome
            new_example = pd.DataFrame({
                'hitData_launchSpeed': [launch_speed],
                'hitData_launchAngle': [launch_angle],
                'venue_name': [stadium]
            })
            new_example_preprocessed = preprocessor.transform(new_example)
            probabilities = loaded_model.predict_proba(new_example_preprocessed)[0]
            probabilities_dict[tuple(outcome)] = probabilities
    
    while outcomes_copy:  # Continue until all outcomes are used
        if outs == 3:
            outs = 0
            bases = [False, False, False]  # Clear the bases after 3 outs
        
        # Sample an outcome from the list
        outcome = random.choice(outcomes_copy)
        outcomes_copy.remove(outcome)  # Remove the sampled outcome from the copy

        if outcome == "out":
            outs += 1
        elif outcome == "walk":
            advance_runner(bases)
        elif isinstance(outcome, list) and len(outcome) == 3:
            # Get the pre-calculated probabilities for the outcome
            probabilities = probabilities_dict[tuple(outcome)]

            # Generate a random value between 0 and 1
            random_value = random.random()

            # Determine the outcome based on the probabilities
            if random_value < probabilities[0]:
                outs += 1
            elif random_value < probabilities[0] + probabilities[1]:
                runs += advance_runner(bases)
                bases[0] = True
            elif random_value < probabilities[0] + probabilities[1] + probabilities[2]:
                runs += advance_runner(bases, 2)
                bases[1] = True
            elif random_value < probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3]:
                runs += advance_runner(bases, 3)
                bases[2] = True
            else:
                runs += advance_runner(bases, 4)
                bases = [False, False, False]
    
    return runs

def advance_runner(bases, count=1):
    runs = 0
    for _ in range(count):
        if bases[2]:
            runs += 1
        bases[2] = bases[1]
        bases[1] = bases[0]
        bases[0] = True
    return runs

def simulator(num_simulations, home_outcomes, away_outcomes):
    ## Clean up the tuples
    home_outcomes = [outcome[0] if isinstance(outcome[0], tuple) else outcome[0] for outcome in home_outcomes]
    away_outcomes = [outcome[0] if isinstance(outcome[0], tuple) else outcome[0] for outcome in away_outcomes]
    
    # Simulate the game for home_outcomes and away_outcomes using NumPy
    home_runs_scored = np.zeros(num_simulations, dtype=int)
    away_runs_scored = np.zeros(num_simulations, dtype=int)

    for i in range(num_simulations):
        home_runs_scored[i] = simulate_game(home_outcomes)
        away_runs_scored[i] = simulate_game(away_outcomes)

    # Compare the scores and calculate win/tie/loss percentages
    home_wins = np.sum(home_runs_scored > away_runs_scored)
    away_wins = np.sum(home_runs_scored < away_runs_scored)
    ties = np.sum(home_runs_scored == away_runs_scored)

    home_win_percentage = home_wins / num_simulations * 100
    away_win_percentage = away_wins / num_simulations * 100
    tie_percentage = ties / num_simulations * 100

    return home_runs_scored, away_runs_scored, home_win_percentage, away_win_percentage, tie_percentage
