# baseball_simulator
MLB Deserve-to-Win Simulator

The idea is to utilize launch angle and exit velocity 

## In the get_game_information file, I think we should filter to include only where the ball was put in play or a walk/hbp or a strikeout
## Then make sure to remove duplicates
## Then, we create a new df that includes venue_name, balls, strikes, outs, bases occupied, launch angle, exit velo
## Here, we could merge run expectancy, launch angle and exit velo at the specific venue
## A walk automatically moves everyone up 1 base
## A strikeout adds 1 out
## In main, we use the model to predict the outcome of the of the launch angle and exit velo at the specific venue
## Then, we roll the dice to see what the outcome is
## Then, we adjust the bases, scores, outs
## Repeat 10,000 times and compare winners
