def outcomes(game_data, steals_and_pickoffs, home_or_away):
    """
    Extract batting outcomes and baserunning events from game data for either home or away team.
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game events
        steals_and_pickoffs (pd.DataFrame): DataFrame containing stolen base and pickoff events
        home_or_away (str): 'home' or 'away' to filter team data
        
    Returns:
        list: Tuples of (outcome_data, event_type, player_name) where outcome_data is either
              a string ('strikeout'/'walk'/'stolen_base'/'pickoff') or 
              list [launch_speed, launch_angle, venue]
    """
    home_or_away_team = game_data.copy()
    if home_or_away == 'home':
        home_or_away_team = home_or_away_team[home_or_away_team['isTopInning'] == False]
        baserunning_events = steals_and_pickoffs[steals_and_pickoffs['isTopInning'] == False]
    else:
        home_or_away_team = home_or_away_team[home_or_away_team['isTopInning'] == True]
        baserunning_events = steals_and_pickoffs[steals_and_pickoffs['isTopInning'] == True]
    
    outcomes = []
    
    # Process batting outcomes
    automatic_outs = home_or_away_team[(home_or_away_team['eventType'] == 'out') & 
                                     (home_or_away_team['hitData.launchSpeed'].isnull())]
    for _, row in automatic_outs.iterrows():
        outcomes.append(("strikeout", row['eventType'], row['batter.fullName']))
        
    walks = home_or_away_team[home_or_away_team['eventType'] == 'walk']
    for _, row in walks.iterrows():
        outcomes.append(("walk", row['eventType'], row['batter.fullName']))
        
    put_in_play = home_or_away_team[~home_or_away_team['hitData.launchSpeed'].isnull()].reset_index(drop=True)
    for _, row in put_in_play.iterrows():
        outcomes.append(([row['hitData.launchSpeed'], row['hitData.launchAngle'], row['venue.name']], 
                       row['eventType'], row['batter.fullName']))
    
    # Process baserunning events
    if not baserunning_events.empty:
        for _, row in baserunning_events.iterrows():
            outcomes.append((row['play'], row['play'], row['batter.fullName']))
    
    # Sort outcomes by their original order
    return outcomes

def calculate_total_bases(outcomes):
    """
    Calculate expected bases and outcome probabilities for each batting event.
    
    Args:
        outcomes (list): List of outcome tuples from outcomes()
        
    Returns:
        pd.DataFrame: Detailed stats including launch data, probabilities, and estimated bases
    """
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
            new_example = pd.DataFrame({
                'hitData_launchSpeed': [launch_speed],
                'hitData_launchAngle': [launch_angle],
                'venue_name': [stadium]
            })
            probabilities = pipeline.predict_proba(new_example)[0]
            bases = (
                probabilities[1] * 1 +
                probabilities[2] * 2 +
                probabilities[3] * 3 +
                probabilities[4] * 4
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
    """
    Create a detailed DataFrame of batting outcomes for specified team.
    
    Args:
        game_data (pd.DataFrame): Original game data
        home_or_away (str): 'home' or 'away' team selection
        
    Returns:
        pd.DataFrame: Processed and cleaned batting outcomes with probabilities
    """
    outcomes_list = outcomes(game_data, home_or_away)
    detailed_df = calculate_total_bases(outcomes_list)
    detailed_df = detailed_df.dropna().reset_index(drop=True)
    return detailed_df

def outcome_rankings(home_detailed_df, away_detailed_df):
    """
    Combine and rank batting outcomes from both teams by estimated bases.
    
    Args:
        home_detailed_df (pd.DataFrame): Home team detailed outcomes
        away_detailed_df (pd.DataFrame): Away team detailed outcomes
        
    Returns:
        pd.DataFrame: Top 10 outcomes ranked by estimated bases, formatted for display
    """
    total_team_outcomes = pd.concat([home_detailed_df, away_detailed_df])
    total_team_outcomes['launch_angle'] = total_team_outcomes['launch_angle'].astype(int)
    total_team_outcomes['estimated_bases'] = total_team_outcomes['estimated_bases'].round(2)
    
    prob_columns = [col for col in total_team_outcomes.columns if '_prob' in col]
    for col in prob_columns:
        total_team_outcomes[col] = (total_team_outcomes[col] * 100).round(0).astype(int).astype(str) + '%'
    
    selected_columns = ['team', 'player', 'launch_speed', 'launch_angle', "original_event_type", 'estimated_bases', 
                       "out_prob", "single_prob", "double_prob", "triple_prob", "hr_prob"]
    total_team_outcomes = total_team_outcomes[selected_columns]
    total_team_outcomes['original_event_type'] = total_team_outcomes['original_event_type'].str.title()
    total_team_outcomes.columns = total_team_outcomes.columns.str.replace('_', ' ').str.title()
    total_team_outcomes = total_team_outcomes.rename(columns={'Original Event Type': 'Result', 'Full Name': 'Player'})
    total_team_outcomes['team_color'] = total_team_outcomes['Team'].map({team: colors[0] for team, colors in team_colors.items()})
    return total_team_outcomes.sort_values(by='Estimated Bases', ascending=False).head(10)

def simulate_game(outcomes_df):
    """
    Simulate one game's worth of plate appearances and calculate runs scored.
    
    Args:
        outcomes_df (pd.DataFrame): DataFrame containing possible batting outcomes
        
    Returns:
        int: Total runs scored in the simulated game
    """
    outs = 0
    runs = 0
    bases = [False, False, False]
    outcomes_copy = outcomes_df.copy()
    
    probabilities_dict = {}
    for outcome in outcomes_copy:
        if isinstance(outcome, list) and len(outcome) == 3:
            launch_speed, launch_angle, stadium = outcome
            new_example = pd.DataFrame({
                'hitData_launchSpeed': [launch_speed],
                'hitData_launchAngle': [launch_angle],
                'venue_name': [stadium]
            })
            probabilities = pipeline.predict_proba(new_example)[0]
            probabilities_dict[tuple(outcome)] = probabilities
    
    while outcomes_copy:
        if outs == 3:
            outs = 0
            bases = [False, False, False]
        
        outcome = random.choice(outcomes_copy)
        outcomes_copy.remove(outcome)
        
        if outcome == "strikeout":
            outs += 1
        elif outcome == "walk":
            runs += advance_runner(bases, is_walk=True)
        elif isinstance(outcome, list) and len(outcome) == 3:
            probabilities = probabilities_dict[tuple(outcome)]
            random_value = random.random()
            if random_value < probabilities[0]:
                outs += 1
            elif random_value < probabilities[0] + probabilities[1]:
                runs += advance_runner(bases, 1)
            elif random_value < probabilities[0] + probabilities[1] + probabilities[2]:
                runs += advance_runner(bases, 2)
            elif random_value < probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3]:
                runs += advance_runner(bases, 3)
            else:
                runs += advance_runner(bases, 4)
    
    return runs

def advance_runner(bases, count=1, is_walk=False):
    """
    Calculate runs scored and update base runners after a hit/walk.
    
    Args:
        bases (list): Current base occupancy [1st, 2nd, 3rd]
        count (int): Number of bases advanced (1-4)
        is_walk (bool): True if event is a walk
        
    Returns:
        int: Runs scored on this play
    """
    runs = 0
    if is_walk:
        if bases[2] and bases[1] and bases[0]:
            runs += 1
            bases[2] = bases[1]
            bases[1] = bases[0]
            bases[0] = True
        elif bases[1] and bases[0]:
            bases[2] = bases[1]
            bases[1] = bases[0]
            bases[0] = True
        elif bases[0]:
            bases[1] = bases[0]
            bases[0] = True
        else:
            bases[0] = True
    else:
        for _ in range(count):
            if bases[2]:
                runs += 1
            bases[2] = bases[1]
            bases[1] = bases[0]
            bases[0] = True
    
    return runs

def simulator(num_simulations, home_outcomes, away_outcomes):
    """
    Run multiple game simulations and calculate win probabilities.
    
    Args:
        num_simulations (int): Number of games to simulate
        home_outcomes (list): List of possible home team batting outcomes
        away_outcomes (list): List of possible away team batting outcomes
        
    Returns:
        tuple: (home_runs_array, away_runs_array, home_win_pct, away_win_pct, tie_pct)
    """
    home_runs_scored = np.zeros(num_simulations, dtype=int)
    away_runs_scored = np.zeros(num_simulations, dtype=int)
    
    for i in tqdm(range(num_simulations), desc="Simulating games", unit="sim"):
        home_runs_scored[i] = simulate_game(home_outcomes)
        away_runs_scored[i] = simulate_game(away_outcomes)

    home_wins = np.sum(home_runs_scored > away_runs_scored)
    away_wins = np.sum(home_runs_scored < away_runs_scored)
    ties = np.sum(home_runs_scored == away_runs_scored)

    home_win_percentage = home_wins / num_simulations * 100
    away_win_percentage = away_wins / num_simulations * 100
    tie_percentage = ties / num_simulations * 100

    return home_runs_scored, away_runs_scored, home_win_percentage, away_win_percentage, tie_percentage
