# baseball_simulator
MLB Deserve-to-Win Simulator - https://x.com/mlb_simulator - (inspired by @MoneyPuckdotcom)

I wanted to analyze how game outcomes can change, depending on when a hit occurred. For example, I wanted to determine if a team hit well, but struggled to string together hits. I essentially wanted to determine a team's "luck" factor by how their hits were dispersed in a game vs many simulations. For example, leaving 15 batters on base to end the game can be incredibly frustrating for fans. So, this can let them know that they should have won!

Additionally, I wanted to utilize a model that uses launch angle, exit velocity, and ballpark to determine luck, since I wanted to reward teams that hit the ball hard. In the future, I'd like to add a new sim that only resamples the existing outcomes, instead of using LA/EV, but selfishly I also wanted to my modeling skills to the test. I've included the methodology behind my model as an ipynb. Please, feel free to rip it up and improve it!

In the future, I'd like to use a Bayesian Multiniomial model to utilize better priors for batted ball outcomes.

The data was gathered from the MLB Stats API.

The main.ipynb will run the simulations and create visualizations about the run distributions and batted ball outcomes, which are then saved in the images folder.

2025 Additions:

- Games played at George M. Steinbrenner Field will be converted to Yankee Stadium in the model, since the dimensions are similar.
##
Future ideas:

Incorporate double plays into the simulations, depending on ground ball probabilities.

Utilize linear algebra/matrices to speed up simulations.

##
Assumptions:

The number of strikeouts and walks/hbp remain the same.

The simulation can have fewer/more ABs/innings than the real game, since the outcomes will be distributed differently.
##
Reasons actual vs simulation can vary:

This simulation is outcome agnostic, so different outcomes can be assigned to each player. So, even if Player A hits a home run in the real game, this home run could be assigned to a different player in the simulation. So, a team can outperform the simulation if the roster is constructed better than the simulation.

This model uses launch angle and exit velocity to predict outcomes, so a great defensive play in real life can be a hit in the simulation.

If the away team went up to bat in the 9th inning, while the home team did not, then the away team could be favored more than the home team in the simulation.

Fielding errors are not accounted for, so a team can outperform or underperform in the simulation depending on errors.

##
Model Testing (Expected Bases):

![image](https://github.com/user-attachments/assets/4c8390a4-3467-4992-b160-f6d54e4af679)



##
Outputs:

![image](https://github.com/user-attachments/assets/21e0a3b1-2114-4134-88d6-0284c76af581)
![image](https://github.com/user-attachments/assets/94a04426-ba4e-40a0-ac36-06533fba7b6e)
![image](https://github.com/user-attachments/assets/75a29e0d-7eff-490b-bac1-34fe6c9af2d9)

##
2024 Research:

![image](https://github.com/user-attachments/assets/c3482c9c-cbfd-426d-b2a2-71b01165d0fb)

![image](https://github.com/user-attachments/assets/e74267f4-fe24-47f3-a075-1f9405c27612)




