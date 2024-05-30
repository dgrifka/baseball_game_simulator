## In the get_game_information file, I think we should filter to include only where the ball was put in play or a walk/hbp or a strikeout
## Then make sure to remove duplicates
## Then, we create a new df that includes venue_name, balls, strikes, outs, bases occupied, launch angle, exit velo
## Here, we could merge run expectancy, launch angle and exit velo at the specific venue
## A walk/hbp automatically moves everyone up 1 base
## A strikeout adds 1 out
## In main, we use the model to predict the outcome of the of the launch angle and exit velo at the specific venue
## Then, we roll the dice to see what the outcome is
## Then, we adjust the bases, scores, outs
## Repeat 10,000 times and compare winners

import pandas as pd
import pickle

# Load the game data from the pandas DataFrame
game_data = pd.read_csv('game_data.csv')

# Load the logistic regression model from the pickle file
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize variables for runs and outs
away_runs = 0
home_runs = 0
outs = 0

# Function to simulate a half-inning
def simulate_half_inning(team):
    global outs
    outs = 0
    while outs < 3:
        # Get the next batter's data from the game_data DataFrame
        batter_data = game_data.loc[game_data['team'] == team].iloc[0]
        
        # Extract the relevant features for the logistic regression model
        launch_speed = batter_data['hitData.launchSpeed']
        launch_angle = batter_data['hitData.launchAngle']
        venue_name = batter_data['venue.name']
        
        # Prepare the input features for the model
        input_features = [[launch_speed, launch_angle, venue_name]]
        
        # Use the logistic regression model to predict the outcome
        outcome = model.predict(input_features)
        
        # Update runs and outs based on the outcome
        if outcome == 'hit':
            if team == 'away':
                global away_runs
                away_runs += 1
            else:
                global home_runs
                home_runs += 1
        else:
            outs += 1
        
        # Remove the current batter from the game_data DataFrame
        game_data.drop(game_data.index[0], inplace=True)

# Simulate the game
for inning in range(1, 10):
    print(f"Inning {inning}")
    
    # Top half of the inning (away team)
    print("Top of the inning")
    simulate_half_inning('away')
    
    # Bottom half of the inning (home team)
    print("Bottom of the inning")
    simulate_half_inning('home')
    
    print(f"Score: Away {away_runs} - Home {home_runs}")
    print()

# Final score
print("Final Score:")
print(f"Away: {away_runs}")
print(f"Home: {home_runs}")