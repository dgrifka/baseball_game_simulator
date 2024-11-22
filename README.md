# MLB Deserve-to-Win Simulator
### https://x.com/mlb_simulator
### [@mlb-simulator.bsky.social](https://bsky.app/profile/mlb-simulator.bsky.social)

## Table of Contents
- [Description](#description)
- [Assumptions](#assumptions)
- [Reasons Actual vs Simulation Can Vary](#reasons-actual-vs-simulation-can-vary)
- [Future Ideas](#future-ideas)
- [2025 Additions](#2025-additions)
- [Project Structure](#project-structure)
- [Model Testing](#model-testing-expected-bases)
- [Outputs](#outputs)
- [2024 Research](#2024-research)
  
## Description

This project simulates alternative outcomes of MLB games by resampling batted ball events, walks, strikeouts, steals, and pickoffs to determine who "should have won" a game. The simulator uses advanced metrics including launch angle, exit velocity, and ballpark-specific factors to provide a more nuanced analysis of team performance and game outcomes. 

Regarding the model, I use a Gradient Boosting Classifier trained on 2023-24 MLB batted ball data, which is then wrapped in a scikit-learn pipeline. The model can correctly predict the outcome 77% of the time (w/test data), but more importantly, provides probabilities for all outcomes. For more info, check out Model/Base_Model.ipynb

The main.ipynb will run the simulations and create visualizations about the run distributions and batted ball outcomes. I maintain a separate Colab file that saves and Tweets the images automatically. 

This account was inspired by @MoneyPuckdotcom. The model and account are derived from data gathered from MLB Stats API.

Feel free to submit a pull request with modifications/improvements!

## Future ideas:

Layer the expected outcome model by predicting outcomes conditional on other outcome probabilities.

Add sprint speed or stolen bases per PA to account for player speed.

Incorporate double plays into the simulations, depending on ground ball probabilities.

Incorporate prior run distribution into likelihood of the simulations.

## Assumptions:

The number of strikeouts and walks/hbp remain the same in each simulation.

The simulation can have fewer/more innings than the real game, since the outcomes will be distributed differently.

## Reasons actual vs simulation can vary:

This simulation assigns the outcomes and batted balls randomly throughout the lineup, so different outcomes can be assigned to each player. So, even if Player A hits a home run in the real game, this home run could be assigned to a different player in the simulation. So, a team can outperform the simulation if the roster is constructed better than the simulation. Although roster construction is an important aspect of the game, we're mostly focused on the randomness of sequencing.

This model uses launch angle and exit velocity to predict outcomes, so a great defensive play in the game can be a hit in the simulation.

If the away team put the ball in play in the 9th inning, while the home team did not, then the away team could be favored more than the home team in the simulation.

Fielding errors are not accounted for, so a team can outperform or underperform in the simulation, depending on errors.

## 2025 Additions:

- Improved code docstrings and documentation.
- Added steals and pickoffs to simulations
- Cleaner and improved visuals
- Games played at George M. Steinbrenner Field will be converted to Yankee Stadium in the model, since the dimensions are similar.

## Project Structure

├── baseball_game_simulator/

├── .gitignore

├── README.md # Data and project documentation

└── main.ipynb # Colab file used to run and save the visualizations

├── Model/

│   ├── Base_Model.ipynb # Colab file used to create and save model

│   └── gb_classifier_pipeline.pkl # Model used to assign outcome probabilities

├── Research/

│   └── 2024_Season_WP_Model.ipynb # Analyzing batted ball outcomes by team with a Bayesian hierarchical model

├── Simulator/

│   ├── constants.py # Hard-coded values used in the simulator

│   ├── game_simulator.py # Code used to simulate batted ball outcomes

│   ├── get_game_information.py # Scrapes MLB Stats API for game info

│   ├── utils.py # Misc helper code

│   └── visualizations.py # Code for all plots

├── Data/

│   └── contour_data.csv # Underlying contour data used in the EV/LA graph

## Contact

Derek Grifka – dmgrifka@gmail.com - https://dgrifka.github.io/

## Model Testing (Expected Bases):

![image](https://github.com/user-attachments/assets/4c8390a4-3467-4992-b160-f6d54e4af679)

## Outputs:

![Dodgers_Yankees_7-6--27-66_bb](https://github.com/user-attachments/assets/378f8eba-5450-46bf-b430-96ec639d3960)

![Dodgers_Yankees_7-6--27-66_rd](https://github.com/user-attachments/assets/3b4921b7-0e00-4cee-a3ad-c431877e2766)

![Dodgers_Yankees_7-6--27-66_estimated_bases](https://github.com/user-attachments/assets/329c809c-5eba-49de-8c3a-38fd6ce688d5)


## 2024 Research:

![image](https://github.com/user-attachments/assets/c3482c9c-cbfd-426d-b2a2-71b01165d0fb)

![image](https://github.com/user-attachments/assets/e74267f4-fe24-47f3-a075-1f9405c27612)


