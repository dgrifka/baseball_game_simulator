# baseball_simulator
MLB Deserve-to-Win Simulator -- STILL IN PRODUCTION

The idea is to utilize launch angle and exit velocity 

Reasons actual vs predicted can vary:

This simulation is lineup agnostic, so different outcomes can be assigned to each player. So, even if Player A hits a home run in the real game, this home run could be assigned to a different player in the simulation. So, a team can outperform the simulation if the roster is constructed better than the simulation.

This model uses launch angle and exit velocity to predict outcomes, so a great defensive play in real life can be a double in the simulation.

## Then, we create a new df that includes venue_name, balls, strikes, outs, bases occupied, launch angle, exit velo
## Here, we could merge run expectancy, launch angle and exit velo at the specific venue
## A walk automatically moves everyone up 1 base
## A strikeout adds 1 out
## In main, we use the model to predict the outcome of the of the launch angle and exit velo at the specific venue
## Then, we roll the dice to see what the outcome is
## Then, we adjust the bases, scores, outs
## Repeat 10,000 times and compare winners
