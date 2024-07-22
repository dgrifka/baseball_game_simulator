# baseball_simulator
MLB Deserve-to-Win Simulator (inspired by @MoneyPuckdotcom)

I wanted to analyze how game outcomes can change, depending on when a hit occurred. For example, I wanted to determine if a team hit well, but struggled to string together hits. I essentially wanted to determine a team's "luck" factor by how their hits were dispersed in a game vs many simulations. For example, leaving 15 batters on base to end the game can be incredibly frustrating for fans. So, this can let them know that they should have won!

Additionally, I wanted to utilize a model that uses launch angle, exit velocity, and ballpark to determine luck, since I wanted to reward teams that hit the ball hard. In the future, I'd like to add a new sim that only resamples the existing outcomes, instead of using LA/EV, but selfishly I also wanted to my modeling skills to the test. I've included the methodology behind my model as an ipynb. Please, feel free to rip it up and improve it!

In the future, I'd like to use a Bayesian Multiniomial model to utilize better priors for batted ball outcomes.

The data was gathered from the MLB Stats API.

The main.py will run the simulations and create visualizations about the run distributions and batted ball outcomes, which are then saved in the images folder.

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
Model Testing:

![download](https://github.com/user-attachments/assets/29443ee2-81e6-4e5f-86cc-7e71f5607f5e)


##
Outputs:

![img1](https://github.com/user-attachments/assets/8bcbe33c-66ff-40da-a8cd-216b3af7f988)
![img2](https://github.com/user-attachments/assets/79f93289-a542-433f-923a-670255bd55ce)
![img3](https://github.com/user-attachments/assets/bc8291fd-1610-4e43-aab1-d8e01c363065)
![img4](https://github.com/user-attachments/assets/bcc016b9-4045-4d16-9a4b-293f45d13f61)







