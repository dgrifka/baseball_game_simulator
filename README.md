# baseball_simulator
MLB Deserve-to-Win Simulator -- STILL IN PRODUCTION

First, we're utilizing launch angle and exit velocity, since I wanted to reward teams that hit the ball hard. I'd like to add a feature that resamples the existing outcomes, instead of LA/EV, but selfishly I just wanted to learn how to importing a model into a simulation via pickle.
We're essentially determining a team's "luck" factor by how their hits were dispersed in a game vs many simulations.

Assumptions

The number of strikeouts and walks (incl. hbp) remain the same.

The simulation can have fewer/more ABs/innings than the real game, since the outcomes will be distributed differently.
##
Reasons actual vs simulation can vary:

This simulation is outcome agnostic, so different outcomes can be assigned to each player. So, even if Player A hits a home run in the real game, this home run could be assigned to a different player in the simulation. So, a team can outperform the simulation if the roster is constructed better than the simulation.

This model uses launch angle and exit velocity to predict outcomes, so a great defensive play in real life can be a double in the simulation.

If the away team went up to bat in the 9th inning, while the home team did not, then the away team could be favored more than the home team in the simulation.

Fielding errors are not accounted for, so a team can outperform or underperform in the simulation depending on errors.

## Then, we create a new df that includes venue_name, balls, strikes, outs, bases occupied, launch angle, exit velo
## Here, we could merge run expectancy, launch angle and exit velo at the specific venue
## A walk automatically moves everyone up 1 base
## A strikeout adds 1 out
## In main, we use the model to predict the outcome of the of the launch angle and exit velo at the specific venue
## Then, we roll the dice to see what the outcome is
## Then, we adjust the bases, scores, outs
## Repeat 10,000 times and compare winners
