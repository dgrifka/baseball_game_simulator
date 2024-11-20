# MLB Deserve-to-Win Simulator - https://x.com/mlb_simulator

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

I wanted to analyze how game outcomes can change, depending on when a hit occurred. For example, I wanted to determine if a team hit well, but struggled to string together hits. I essentially wanted to determine a team's "luck" factor by how their hits were dispersed in a game vs many simulations. For example, leaving 15 batters on base to end the game can be incredibly frustrating for fans. So, this can let them know that they should have won!

Additionally, I wanted to utilize a model that uses launch angle, exit velocity, and ballpark to determine luck, since I wanted to reward teams that hit the ball hard. In the future, I'd like to add a new sim that only resamples the existing outcomes, instead of using LA/EV, but selfishly I also wanted to my modeling skills to the test. I've included the methodology behind my model as an ipynb. Please, feel free to rip it up and improve it!

The main.ipynb will run the simulations and create visualizations about the run distributions and batted ball outcomes. I maintain a separate Colab file that saves and Tweets the images automatically. 

This account was inspired by @MoneyPuckdotcom. The data was gathered from MLB Stats API.

## Future ideas:

Incorporate double plays into the simulations, depending on ground ball probabilities.

Utilize linear algebra/matrices to speed up simulations.

## Assumptions:

The number of strikeouts and walks/hbp remain the same.

The simulation can have fewer/more ABs/innings than the real game, since the outcomes will be distributed differently.

## Reasons actual vs simulation can vary:

This simulation is outcome agnostic, so different outcomes can be assigned to each player. So, even if Player A hits a home run in the real game, this home run could be assigned to a different player in the simulation. So, a team can outperform the simulation if the roster is constructed better than the simulation.

This model uses launch angle and exit velocity to predict outcomes, so a great defensive play in real life can be a hit in the simulation.

If the away team went up to bat in the 9th inning, while the home team did not, then the away team could be favored more than the home team in the simulation.

Fielding errors are not accounted for, so a team can outperform or underperform in the simulation depending on errors.

## 2025 Additions:

- Improved code docstrings and documentation.
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


