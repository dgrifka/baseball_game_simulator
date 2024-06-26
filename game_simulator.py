import random
import pickle
import pandas as pd
import numpy as np

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

    ## Calculate number of automatic outs (i.e. strikeouts)
    ## We're going to assume these outs stay the same in the simulations
    automatic_outs = home_or_away_team.copy()
    automatic_outs = automatic_outs[(automatic_outs['eventType'] == 'out') & (automatic_outs['hitData.launchSpeed'].isnull())]
    strikeouts = len(automatic_outs)
    ## Calculate the number of walks
    walks = home_or_away_team.copy()
    walks = walks[walks['eventType'] == 'walk']
    walk_len = len(walks)

    ## Now let's create a df with balls put in play
    put_in_play = home_or_away_team.copy()
    put_in_play = put_in_play[~put_in_play['hitData.launchSpeed'].isnull()].reset_index(drop=True)
    put_in_play = put_in_play[['hitData.launchSpeed', 'hitData.launchAngle', 'venue.name']]

    ## Now, we'll create a list of outcomes to sample from
    # Convert the DataFrame to a list of lists
    pip_list = put_in_play[['hitData.launchSpeed', 'hitData.launchAngle', 'venue.name']].values.tolist()

    # Create a list of "strikeout" and "walk" strings
    strikeout_list = ["strikeout"] * strikeouts
    walk_list = ["walk"] * walk_len

    # Combine the two lists
    outcomes = pip_list + strikeout_list + walk_list

    return outcomes

## For the estimated total bases graph
def calculate_total_bases(outcomes):
    total_bases = 0
    for outcome in outcomes:
        if outcome == "strikeout":
            bases = 0
        elif outcome == "walk":
            bases = 1
        else:
            launch_speed, launch_angle, stadium = outcome
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
        total_bases += bases
    return total_bases

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