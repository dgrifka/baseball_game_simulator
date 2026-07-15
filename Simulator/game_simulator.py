"""
MLB game simulation engine.
Simulates batted ball outcomes using a gradient boosting model trained on Statcast data.
"""

import itertools
import random
import warnings
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm

from Simulator.constants import team_colors, VENUE_NAME_TO_ID, DEFAULT_VENUE_ID
from Simulator import vector_engine
from Model.feature_engineering import (
    create_features_for_prediction,
    create_features_for_prediction_fallback
)

# =============================================================================
# HR TAIL CORRECTION
# =============================================================================
# Post-hoc correction for systematic HR under-prediction at high exit velocities.
# The GBC model under-predicts HR probability at 100+ mph (validated on 32k batted
# balls from 2025: ~1.04x at 100-105 mph, ~1.10x at 105+ mph). We apply a
# conservative half-correction to the HR probability and redistribute from out_prob.
#
# SCOPE — simulation only, by design. The correction is applied in simulator()
# and simulator_by_inning() (the win% draws), but deliberately NOT in
# calculate_total_bases(), which produces the estimated_bases / *_prob values
# shown in the Estimated Bases table AND exported downstream as the data
# standard (batted-balls parquet → player evaluations, luck ledgers, weekly
# content). Applying it there would change the scoring scale mid-season and
# desync new rows from historical data; doing so requires a deliberate
# decision plus a full historical re-score of the exported parquets.
HR_TAIL_CORRECTIONS = {
    (100, 105): 1.02,  # actual/model ratio ~1.04, apply half
    (105, 200): 1.05,  # actual/model ratio ~1.10, apply half
}

# =============================================================================
# LOAD MODEL
# =============================================================================

# Load the trained pipeline
import os as _os
_this_dir = _os.path.dirname(_os.path.abspath(__file__))
_model_path = _os.path.join(_this_dir, '..', 'Model', 'batted_ball_model.pkl')
pipeline = joblib.load(_model_path)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _apply_hr_tail_correction(probabilities, launch_speed):
    """Apply post-hoc HR probability correction for high exit velocities.

    Boosts HR probability and redistributes the increase from out_prob to keep
    the probability vector summing to 1. Only affects 100+ mph batted balls.

    Args:
        probabilities: array of [out, single, double, triple, hr] probabilities
        launch_speed: exit velocity in mph

    Returns:
        Corrected probability array (new copy if modified, original otherwise)
    """
    if launch_speed < 100:
        return probabilities

    for (lo, hi), factor in HR_TAIL_CORRECTIONS.items():
        if lo <= launch_speed < hi:
            probs = probabilities.copy()
            hr_boost = probs[4] * (factor - 1.0)
            probs[4] += hr_boost
            probs[0] = max(0.0, probs[0] - hr_boost)
            return probs

    return probabilities

def get_venue_id(venue_name):
    """
    Convert venue name to venue ID for model prediction.
    
    Args:
        venue_name (str): Stadium name
    
    Returns:
        str: Venue ID as string
    """
    return VENUE_NAME_TO_ID.get(venue_name, DEFAULT_VENUE_ID)


def prepare_batted_ball_features(launch_speed, launch_angle, venue_name, 
                                  coord_x=None, coord_y=None, bat_side=None):
    """
    Prepare features for model prediction, handling missing spray data gracefully.
    
    Args:
        launch_speed (float): Exit velocity in mph
        launch_angle (float): Launch angle in degrees
        venue_name (str): Stadium name
        coord_x (float, optional): Hit coordinates X
        coord_y (float, optional): Hit coordinates Y
        bat_side (str, optional): 'L' or 'R' for batter handedness
    
    Returns:
        pd.DataFrame: Features ready for model prediction
    """
    venue_id = get_venue_id(venue_name)
    
    # Check if we have spray angle data
    has_spray_data = (
        coord_x is not None and 
        coord_y is not None and 
        bat_side is not None and
        not pd.isna(coord_x) and 
        not pd.isna(coord_y)
    )
    
    if has_spray_data:
        return create_features_for_prediction(
            launch_speed=launch_speed,
            launch_angle=launch_angle,
            coord_x=coord_x,
            coord_y=coord_y,
            bat_side=bat_side,
            venue_id=venue_id
        )
    else:
        # Fallback: use neutral spray angle
        return create_features_for_prediction_fallback(
            launch_speed=launch_speed,
            launch_angle=launch_angle,
            venue_id=venue_id
        )


# =============================================================================
# OUTCOME EXTRACTION
# =============================================================================

def outcomes(game_data, steals_and_pickoffs, home_or_away):
    """
    Extract batting outcomes and baserunning events from game data for either home or away team.
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game events
        steals_and_pickoffs (pd.DataFrame): DataFrame containing stolen base and pickoff events
        home_or_away (str): 'home' or 'away' to filter team data
        
    Returns:
        list: Tuples of (outcome_data, event_type, player_name, pitcher_name) where
              outcome_data is either a string ('strikeout'/'walk'/'stolen_base'/'pickoff')
              or dict with batted ball data
    """
    home_or_away_team = game_data.copy()
    if home_or_away == 'home':
        home_or_away_team = home_or_away_team[home_or_away_team['isTopInning'] == False]
        baserunning_events = steals_and_pickoffs[steals_and_pickoffs['isTopInning'] == False]
    else:
        home_or_away_team = home_or_away_team[home_or_away_team['isTopInning'] == True]
        baserunning_events = steals_and_pickoffs[steals_and_pickoffs['isTopInning'] == True]

    # Check if pitcher column is available
    has_pitcher = 'pitcher.fullName' in home_or_away_team.columns

    outcomes_list = []

    # Process batting outcomes
    automatic_outs = home_or_away_team[(home_or_away_team['eventType'] == 'out') &
                                     (home_or_away_team['hitData.launchSpeed'].isnull())]
    for _, row in automatic_outs.iterrows():
        pitcher_name = row.get('pitcher.fullName') if has_pitcher else None
        outcomes_list.append(("strikeout", row['eventType'], row['batter.fullName'], pitcher_name))

    walks = home_or_away_team[home_or_away_team['eventType'] == 'walk']
    for _, row in walks.iterrows():
        pitcher_name = row.get('pitcher.fullName') if has_pitcher else None
        outcomes_list.append(("walk", row['eventType'], row['batter.fullName'], pitcher_name))

    put_in_play = home_or_away_team[~home_or_away_team['hitData.launchSpeed'].isnull()].reset_index(drop=True)
    for _, row in put_in_play.iterrows():
        pitcher_name = row.get('pitcher.fullName') if has_pitcher else None
        # Create dict with all batted ball data (including spray angle fields)
        batted_ball_data = {
            'launch_speed': row['hitData.launchSpeed'],
            'launch_angle': row['hitData.launchAngle'],
            'total_distance': row.get('hitData.totalDistance'),
            'venue_name': row['venue.name'],
            'coord_x': row.get('hitData.coordinates.coordX'),
            'coord_y': row.get('hitData.coordinates.coordY'),
            'bat_side': row.get('batSide.code'),  # Flattened column name
            'pitcher_hand': row.get('pitchHand.code'),
            'pitcher_id': row.get('pitcher.id'),
            'batter_id': row.get('batter.id'),
            'play_id': row.get('playId'),
            'inning': row.get('inning'),
            'is_top_inning': row.get('isTopInning'),
        }
        outcomes_list.append((batted_ball_data, row['eventType'], row['batter.fullName'], pitcher_name))

    # Process baserunning events
    if not baserunning_events.empty:
        for _, row in baserunning_events.iterrows():
            pitcher_name = row.get('pitcher.fullName') if has_pitcher else None
            outcomes_list.append((row['play'], row['play'], row['batter.fullName'], pitcher_name))

    return outcomes_list


def calculate_total_bases(outcomes_list):
    """
    Calculate expected bases and outcome probabilities for each batting/baserunning event.

    Uses RAW model probabilities — no HR_TAIL_CORRECTIONS — on purpose. This
    output is the scoring standard for all estimated-bases data products
    (display table, S3 batted-balls export, player evaluations). See the
    HR_TAIL_CORRECTIONS comment at the top of this module before changing.

    Args:
        outcomes_list (list): List of outcome tuples from outcomes()

    Returns:
        pd.DataFrame: Detailed stats including launch data, probabilities, and estimated bases
    """
    result_list = []

    for item in outcomes_list:
        # Support both 3-element (legacy) and 4-element (with pitcher) tuples
        if len(item) == 4:
            outcome, original_event_type, full_name, pitcher_name = item
        else:
            outcome, original_event_type, full_name = item
            pitcher_name = None
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
        elif outcome == "stolen_base":
            bases = 1
            event_type = "stolen_base"
            probabilities = [0, 1, 0, 0, 0]
            launch_speed, launch_angle, stadium = None, None, None
        elif outcome == "pickoff":
            bases = 0
            event_type = "pickoff"
            probabilities = [1, 0, 0, 0, 0]
            launch_speed, launch_angle, stadium = None, None, None
        elif isinstance(outcome, dict):
            # Batted ball with full data
            launch_speed = outcome['launch_speed']
            launch_angle = outcome['launch_angle']
            stadium = outcome['venue_name']
            event_type = "in_play"
            
            # Create features and predict
            features = prepare_batted_ball_features(
                launch_speed=launch_speed,
                launch_angle=launch_angle,
                venue_name=stadium,
                coord_x=outcome.get('coord_x'),
                coord_y=outcome.get('coord_y'),
                bat_side=outcome.get('bat_side')
            )
            probabilities = pipeline.predict_proba(features)[0]
            
            bases = (
                probabilities[1] * 1 +
                probabilities[2] * 2 +
                probabilities[3] * 3 +
                probabilities[4] * 4
            )
        else:
            # Legacy format: list [launch_speed, launch_angle, venue]
            launch_speed, launch_angle, stadium = outcome
            event_type = "in_play"
            
            features = prepare_batted_ball_features(
                launch_speed=launch_speed,
                launch_angle=launch_angle,
                venue_name=stadium
            )
            probabilities = pipeline.predict_proba(features)[0]
            
            bases = (
                probabilities[1] * 1 +
                probabilities[2] * 2 +
                probabilities[3] * 3 +
                probabilities[4] * 4
            )
        
        row_dict = {
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
            'hr_prob': probabilities[4],
            'coord_x': outcome.get('coord_x') if isinstance(outcome, dict) else None,
            'coord_y': outcome.get('coord_y') if isinstance(outcome, dict) else None,
            'bat_side': outcome.get('bat_side') if isinstance(outcome, dict) else None,
            'pitcher_hand': outcome.get('pitcher_hand') if isinstance(outcome, dict) else None,
            'pitcher_id': outcome.get('pitcher_id') if isinstance(outcome, dict) else None,
            'batter_id': outcome.get('batter_id') if isinstance(outcome, dict) else None,
            'inning': outcome.get('inning') if isinstance(outcome, dict) else None,
            'is_top_inning': outcome.get('is_top_inning') if isinstance(outcome, dict) else None,
        }
        # Include play_id if available (used for Statcast video links)
        play_id = outcome.get('play_id') if isinstance(outcome, dict) else None
        if play_id is not None:
            row_dict['play_id'] = play_id
        # Only include pitcher if available (avoids dropna() removing batted balls)
        if pitcher_name is not None:
            row_dict['pitcher'] = pitcher_name
        result_list.append(row_dict)
    
    return pd.DataFrame(result_list)


def create_detailed_outcomes_df(game_data, steals_and_pickoffs, home_or_away):
    """
    Create a detailed DataFrame of batting outcomes for specified team.
    
    Args:
        game_data (pd.DataFrame): Original game data
        steals_and_pickoffs (pd.DataFrame): Steals and pickoffs data
        home_or_away (str): 'home' or 'away' team selection
        
    Returns:
        pd.DataFrame: Processed and cleaned batting outcomes with probabilities
    """
    outcomes_list = outcomes(game_data, steals_and_pickoffs, home_or_away)
    detailed_df = calculate_total_bases(outcomes_list)
    # Filter out stolen bases and pickoffs
    detailed_df = detailed_df[~detailed_df['event_type'].isin(['stolen_base', 'pickoff'])]
    detailed_df = detailed_df.dropna().reset_index(drop=True)
    return detailed_df


def outcome_rankings(home_detailed_df, away_detailed_df):
    """
    Combine and rank batting outcomes from both teams by estimated bases.
    
    Args:
        home_detailed_df (pd.DataFrame): Home team detailed outcomes
        away_detailed_df (pd.DataFrame): Away team detailed outcomes
        
    Returns:
        pd.DataFrame: Top 15 outcomes ranked by estimated bases, formatted for display
    """
    # Combine team outcomes, excluding any remaining stolen bases or pickoffs
    home_detailed_df = home_detailed_df[~home_detailed_df['event_type'].isin(['stolen_base', 'pickoff'])]
    away_detailed_df = away_detailed_df[~away_detailed_df['event_type'].isin(['stolen_base', 'pickoff'])]
    
    total_team_outcomes = pd.concat([home_detailed_df, away_detailed_df])
    total_team_outcomes['launch_angle'] = total_team_outcomes['launch_angle'].astype(int)
    total_team_outcomes['estimated_bases'] = total_team_outcomes['estimated_bases'].round(2)
    
    prob_columns = [col for col in total_team_outcomes.columns if '_prob' in col]
    for col in prob_columns:
        total_team_outcomes[col] = (total_team_outcomes[col] * 100).round(0).astype(int).astype(str) + '%'
    
    # Include coord_x, coord_y, bat_side for spray direction calculation
    selected_columns = ['team', 'player', 'launch_speed', 'launch_angle', 'original_event_type', 'estimated_bases', 
                       'out_prob', 'single_prob', 'double_prob', 'triple_prob', 'hr_prob',
                       'coord_x', 'coord_y', 'bat_side']
    
    # Only include columns that exist (for backward compatibility)
    selected_columns = [col for col in selected_columns if col in total_team_outcomes.columns]
    total_team_outcomes = total_team_outcomes[selected_columns]
    
    total_team_outcomes['original_event_type'] = total_team_outcomes['original_event_type'].str.title()
    total_team_outcomes = total_team_outcomes.rename(columns={
        'launch_speed': 'Launch Speed', 
        'launch_angle': 'Launch Angle', 
        'original_event_type': 'Result', 
        'out_prob': 'Out Prob', 
        'single_prob': 'Single Prob', 
        'double_prob': 'Double Prob', 
        'triple_prob': 'Triple Prob', 
        'hr_prob': 'Hr Prob', 
        'estimated_bases': 'Estimated Bases', 
        'team': 'Team',
        'player': 'Player'
    })
    
    return total_team_outcomes.sort_values(by='Estimated Bases', ascending=False).head(15).reset_index(drop=True)


def outcomes_by_inning(game_data, steals_and_pickoffs, home_or_away):
    """
    Like outcomes(), but iterates game_data AND steals_and_pickoffs in chronological
    order and tags every outcome (batting AND baserunning) with its real inning.
    Returns a list of (outcome_data, inning) tuples, stably sorted by inning and,
    within an inning, by at-bat number (ab_num) — with a stolen_base/pickoff event
    sorting BEFORE the plate-appearance outcome sharing its ab_num, since a
    baserunning event is attempted mid-at-bat, before that at-bat resolves.

    ab_num is expected to always be present in production data. If a row's
    ab_num is missing/NaN, this raises a RuntimeWarning (via warnings.warn) and
    substitutes a synthetic ab_num drawn from a single counter SHARED across
    both the plate-appearance loop and the baserunning loop below — not two
    independently-reset counters — so synthetic values can never collide
    between the two event types. Synthetic values increase monotonically in
    the order rows are encountered: all plate-appearance rows (in game_data
    order) first, then all baserunning rows (in steals_and_pickoffs order).
    This guarantees a deterministic, collision-free sort, but it is only an
    approximation of true chronological order — it does NOT reconstruct real
    interleaving between plate appearances and baserunning events when ab_num
    is unavailable for both. Covers strikeouts, walks, batted balls, stolen
    bases, and pickoffs. Used by the per-inning win-probability simulator;
    does NOT replace outcomes().
    """
    df = game_data.copy()
    df = df[df['isTopInning'] == (home_or_away == 'away')]

    if home_or_away == 'home':
        baserunning_events = steals_and_pickoffs[steals_and_pickoffs['isTopInning'] == False]
    else:
        baserunning_events = steals_and_pickoffs[steals_and_pickoffs['isTopInning'] == True]

    # (inning, ab_num, is_pa, outcome_data) — is_pa=0 for baserunning events so
    # they sort before a same-ab_num plate-appearance outcome (is_pa=1).
    # `fallback_ab_num` is a single counter shared by BOTH loops below so a
    # synthetic ab_num (substituted only when the real ab_num is missing) can
    # never collide between a plate-appearance row and a baserunning row.
    tagged = []
    fallback_ab_num = itertools.count()
    for _, row in df.iterrows():
        inning = row.get('inning')
        if inning is None or pd.isna(inning):
            continue
        inning = int(inning)
        ab_num = row.get('ab_num')
        if pd.isna(ab_num):
            warnings.warn(
                f"outcomes_by_inning: plate-appearance row missing ab_num "
                f"(inning {inning}) — substituting a synthetic ab_num from a "
                f"shared fallback counter; production data should always "
                f"have ab_num.",
                RuntimeWarning,
                stacklevel=2,
            )
            ab_num = next(fallback_ab_num)
        event_type = row.get('eventType')
        launch_speed = row.get('hitData.launchSpeed')

        # 'out' with no launch data → non-contact out (strikeout-like); contact outs (sac fly, fielder's choice) have launch data and fall through to the batted-ball branch
        if event_type == 'out' and pd.isna(launch_speed):
            outcome_data = "strikeout"
        elif event_type == 'walk':
            outcome_data = "walk"
        elif not pd.isna(launch_speed):
            outcome_data = {
                'launch_speed': launch_speed,
                'launch_angle': row.get('hitData.launchAngle'),
                'total_distance': row.get('hitData.totalDistance'),
                'venue_name': row.get('venue.name'),
                'coord_x': row.get('hitData.coordinates.coordX'),
                'coord_y': row.get('hitData.coordinates.coordY'),
                'bat_side': row.get('batSide.code'),
                'pitcher_hand': row.get('pitchHand.code'),
                'pitcher_id': row.get('pitcher.id'),
                'play_id': row.get('playId'),
                'inning': inning,
                'is_top_inning': row.get('isTopInning'),
            }
        else:
            continue  # HBP, interference, etc. — not modeled (mirrors outcomes())

        tagged.append((inning, ab_num, 1, outcome_data))

    for _, row in baserunning_events.iterrows():
        inning = row.get('inning')
        if inning is None or pd.isna(inning):
            continue
        inning = int(inning)
        ab_num = row.get('ab_num')
        if pd.isna(ab_num):
            warnings.warn(
                f"outcomes_by_inning: baserunning row missing ab_num "
                f"(inning {inning}) — substituting a synthetic ab_num from a "
                f"shared fallback counter; production data should always "
                f"have ab_num.",
                RuntimeWarning,
                stacklevel=2,
            )
            ab_num = next(fallback_ab_num)
        tagged.append((inning, ab_num, 0, row['play']))

    # Stable sort: inning, then ab_num, then baserunning (0) before PA (1) at a tie.
    tagged.sort(key=lambda t: (t[0], t[1], t[2]))
    return [(outcome_data, inning) for inning, _, _, outcome_data in tagged]


# =============================================================================
# BASERUNNING
# =============================================================================

def attempt_steal(bases):
    """
    Attempt steal of next base for lead runner.
    
    Args:
        bases (list): Current base occupancy [1st, 2nd, 3rd]
        
    Returns:
        int: Runs scored (1 if runner on 3rd steals home, else 0)
    """
    if bases[2]:  # Steal home (rare)
        bases[2] = False
        return 1
    elif bases[1]:  # Steal third
        bases[2] = True
        bases[1] = False
    elif bases[0]:  # Steal second
        bases[1] = True
        bases[0] = False
    return 0


def attempt_pickoff(bases):
    """
    Attempt pickoff of lead runner.
    
    Args:
        bases (list): Current base occupancy [1st, 2nd, 3rd]
        
    Returns:
        int: 1 if pickoff successful, 0 otherwise
    """
    if bases[2]:  # Pick off runner on third
        bases[2] = False
    elif bases[1]:  # Pick off runner on second
        bases[1] = False
    elif bases[0]:  # Pick off runner on first
        bases[0] = False
    return 1


# Probabilistic runner-advancement rates (halfway between original rules and
# MLB averages). Module constants so the vectorized engine reads the same
# numbers the scalar code uses — edit here, nowhere else.
SINGLE_ADV_FROM_2ND_P = 0.56    # runner on 2nd takes home on a single
SINGLE_ADV_FROM_1ST_P = 0.265   # runner on 1st takes 3rd on a single (no runner on 2nd)
DOUBLE_ADV_FROM_1ST_P = 0.675   # runner on 1st scores on a double


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
    
    elif count == 4:  # Home run
        runs = sum(bases) + 1
        bases[0] = False
        bases[1] = False
        bases[2] = False
    
    else:  # Singles, doubles, triples
        original_bases = bases.copy()
        bases[0] = bases[1] = bases[2] = False
        
        for i in range(2, -1, -1):  # Work backwards from 3rd to 1st
            if original_bases[i]:
                advancement = count
                
                # Probabilistic advancements (rates defined above advance_runner)
                if count == 1 and i == 1:  # Single with runner on 2nd
                    advancement = 2 if random.random() < SINGLE_ADV_FROM_2ND_P else 1
                elif count == 1 and i == 0 and not original_bases[1]:
                    advancement = 2 if random.random() < SINGLE_ADV_FROM_1ST_P else 1
                elif count == 2 and i == 0:  # Double with runner on 1st
                    advancement = 3 if random.random() < DOUBLE_ADV_FROM_1ST_P else 2
                
                new_position = i + advancement
                if new_position >= 3:
                    runs += 1
                else:
                    bases[new_position] = True
        
        # Put batter on base
        if count == 1:
            bases[0] = True
        elif count == 2:
            bases[1] = True
        elif count == 3:
            bases[2] = True
    
    return runs


# =============================================================================
# SIMULATION ENGINE
# =============================================================================
# simulate_game / simulate_game_by_inning are the scalar REFERENCE
# implementations. Production runs the vectorized engine (vector_engine.py),
# whose transition tables are built from these functions at import time —
# behavior is defined here, executed there. DTW_SCALAR_SIM=1 forces the
# scalar loops (escape hatch).

def _outcome_cache_key(outcome):
    """The cache-key tuple shared by the prob cache and both engines."""
    return (outcome['launch_speed'], outcome['launch_angle'],
            outcome['venue_name'], outcome.get('coord_x'),
            outcome.get('coord_y'), outcome.get('bat_side'))


def _use_scalar_sim():
    return _os.environ.get('DTW_SCALAR_SIM', '') == '1'


_transition_tables = None


def _get_transition_tables():
    """Build (once) the vector engine's tables from the scalar functions above."""
    global _transition_tables
    if _transition_tables is None:
        branch_probs = {}
        for s in range(8):
            if s & 2:                # runner on 2nd: single draws the 0.56 rule
                branch_probs[(s, 1)] = SINGLE_ADV_FROM_2ND_P
            elif s & 1:              # runner on 1st, none on 2nd: the 0.265 rule
                branch_probs[(s, 1)] = SINGLE_ADV_FROM_1ST_P
            if s & 1:                # runner on 1st: double draws the 0.675 rule
                branch_probs[(s, 2)] = DOUBLE_ADV_FROM_1ST_P
        _transition_tables = vector_engine.build_transition_tables(
            advance_runner, attempt_steal, attempt_pickoff, branch_probs)
    return _transition_tables


def simulate_game(outcomes_list, prob_cache):
    """
    Simulate one game using pre-computed probabilities.
    
    Args:
        outcomes_list (list): List of outcome data
        prob_cache (dict): Pre-computed probabilities keyed by outcome
    
    Returns:
        int: Runs scored in simulated game
    """
    outs = 0
    runs = 0
    bases = [False, False, False]
    
    n_outcomes = len(outcomes_list)
    indices = np.random.permutation(n_outcomes)
    
    for idx in indices:
        if outs == 3:
            outs = 0
            bases = [False, False, False]
        
        outcome = outcomes_list[idx]
        
        if outcome == "strikeout":
            outs += 1
        elif outcome == "walk":
            runs += advance_runner(bases, is_walk=True)
        elif outcome == "stolen_base":
            if any(bases):
                runs += attempt_steal(bases)
        elif outcome == "pickoff":
            if any(bases):
                outs += attempt_pickoff(bases)
        elif isinstance(outcome, (dict, tuple)):
            # Get cache key
            if isinstance(outcome, dict):
                cache_key = (outcome['launch_speed'], outcome['launch_angle'],
                             outcome['venue_name'], outcome.get('coord_x'),
                             outcome.get('coord_y'), outcome.get('bat_side'))
            else:
                cache_key = outcome
            
            probabilities = prob_cache.get(cache_key)
            if probabilities is None:
                continue
                
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


def simulate_game_by_inning(outcomes_with_inning, prob_cache, n_innings):
    """
    Simulate one team's game respecting real innings.

    Args:
        outcomes_with_inning (list): (outcome_data, inning) tuples from outcomes_by_inning,
            sorted by inning (within-inning chronological, baserunning events
            interleaved before the same-at-bat plate-appearance outcome).
        prob_cache (dict): same key/value scheme as simulator()'s cache.
        n_innings (int): number of inning buckets (max inning across both teams).

    Returns:
        np.ndarray of shape (n_innings,): CUMULATIVE deserved runs through each inning.
    """
    runs_by_inning = np.zeros(n_innings, dtype=float)
    bases = [False, False, False]
    prev_inning = None

    for outcome, inning in outcomes_with_inning:
        if inning != prev_inning:
            bases = [False, False, False]  # new half-inning for this team
            prev_inning = inning
        i = inning - 1
        if i < 0 or i >= n_innings:
            raise IndexError(
                f"simulate_game_by_inning: outcome in inning {inning} exceeds n_innings={n_innings}"
            )

        if outcome == "strikeout":
            continue  # out: no runs, bases unchanged
        elif outcome == "walk":
            runs_by_inning[i] += advance_runner(bases, is_walk=True)
        elif outcome == "stolen_base":
            if any(bases):
                runs_by_inning[i] += attempt_steal(bases)
        elif outcome == "pickoff":
            # The headline simulate_game() also charges an out here; this
            # inning-structured sim has no out counter (bases only reset at
            # half-inning boundaries), so a pickoff just removes the lead runner.
            if any(bases):
                attempt_pickoff(bases)
        elif isinstance(outcome, dict):
            cache_key = (outcome['launch_speed'], outcome['launch_angle'],
                         outcome['venue_name'], outcome.get('coord_x'),
                         outcome.get('coord_y'), outcome.get('bat_side'))
            probabilities = prob_cache.get(cache_key)
            if probabilities is None:
                continue
            rv = random.random()
            if rv < probabilities[0]:
                pass  # out
            elif rv < probabilities[0] + probabilities[1]:
                runs_by_inning[i] += advance_runner(bases, 1)
            elif rv < probabilities[0] + probabilities[1] + probabilities[2]:
                runs_by_inning[i] += advance_runner(bases, 2)
            elif rv < probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3]:
                runs_by_inning[i] += advance_runner(bases, 3)
            else:
                runs_by_inning[i] += advance_runner(bases, 4)
        # Any other unrecognized outcome is silently skipped.

    return np.cumsum(runs_by_inning)


def simulator(num_simulations, home_outcomes, away_outcomes):
    """
    Run multiple game simulations with pre-computed probabilities.
    
    Args:
        num_simulations (int): Number of games to simulate
        home_outcomes (list): List of home team batting outcomes
        away_outcomes (list): List of away team batting outcomes
        
    Returns:
        tuple: (home_runs_array, away_runs_array, home_win_pct, away_win_pct, tie_pct)
    """
    # Clean up outcomes format
    home_outcomes_clean = []
    away_outcomes_clean = []
    
    for outcome in home_outcomes:
        if isinstance(outcome, tuple):
            home_outcomes_clean.append(outcome[0])
        else:
            home_outcomes_clean.append(outcome)
    
    for outcome in away_outcomes:
        if isinstance(outcome, tuple):
            away_outcomes_clean.append(outcome[0])
        else:
            away_outcomes_clean.append(outcome)
    
    # Pre-compute ALL probabilities ONCE before simulations
    prob_cache = {}
    all_outcomes = home_outcomes_clean + away_outcomes_clean
    
    for outcome in all_outcomes:
        if isinstance(outcome, dict):
            cache_key = (outcome['launch_speed'], outcome['launch_angle'],
                         outcome['venue_name'], outcome.get('coord_x'),
                         outcome.get('coord_y'), outcome.get('bat_side'))
            if cache_key not in prob_cache:
                features = prepare_batted_ball_features(
                    launch_speed=outcome['launch_speed'],
                    launch_angle=outcome['launch_angle'],
                    venue_name=outcome['venue_name'],
                    coord_x=outcome.get('coord_x'),
                    coord_y=outcome.get('coord_y'),
                    bat_side=outcome.get('bat_side')
                )
                probs = pipeline.predict_proba(features)[0]
                prob_cache[cache_key] = _apply_hr_tail_correction(probs, outcome['launch_speed'])
        elif isinstance(outcome, list) and len(outcome) == 3:
            # Legacy format
            cache_key = tuple(outcome)
            if cache_key not in prob_cache:
                launch_speed, launch_angle, stadium = outcome
                features = prepare_batted_ball_features(
                    launch_speed=launch_speed,
                    launch_angle=launch_angle,
                    venue_name=stadium
                )
                prob_cache[cache_key] = pipeline.predict_proba(features)[0]
    
    if _use_scalar_sim():
        # Reference path (DTW_SCALAR_SIM=1): the original per-sim loop.
        home_runs_scored = np.zeros(num_simulations, dtype=int)
        away_runs_scored = np.zeros(num_simulations, dtype=int)
        for i in tqdm(range(num_simulations),
                      desc="Simulating games",
                      unit="sim",
                      position=0,
                      leave=True,
                      ncols=80,
                      ascii=True):
            home_runs_scored[i] = simulate_game(home_outcomes_clean, prob_cache)
            away_runs_scored[i] = simulate_game(away_outcomes_clean, prob_cache)
    else:
        rng = np.random.default_rng()
        tables = _get_transition_tables()
        home_ev, home_cdf = vector_engine.translate_outcomes(
            home_outcomes_clean, prob_cache, _outcome_cache_key)
        away_ev, away_cdf = vector_engine.translate_outcomes(
            away_outcomes_clean, prob_cache, _outcome_cache_key)
        home_runs_scored = vector_engine.simulate_games_vectorized(
            home_ev, home_cdf, tables, num_simulations, rng).astype(int)
        away_runs_scored = vector_engine.simulate_games_vectorized(
            away_ev, away_cdf, tables, num_simulations, rng).astype(int)


    # Calculate win percentages
    home_wins = np.sum(home_runs_scored > away_runs_scored)
    away_wins = np.sum(home_runs_scored < away_runs_scored)
    ties = np.sum(home_runs_scored == away_runs_scored)
    
    home_win_percentage = home_wins / num_simulations * 100
    away_win_percentage = away_wins / num_simulations * 100
    tie_percentage = ties / num_simulations * 100
    
    return home_runs_scored, away_runs_scored, home_win_percentage, away_win_percentage, tie_percentage


def simulator_by_inning(num_simulations, home_outcomes_inn, away_outcomes_inn, prob_cache=None):
    """
    Per-inning deserved-run trajectories via one nested batch of sims.

    Args:
        num_simulations (int)
        home_outcomes_inn, away_outcomes_inn (list): (outcome_data, inning) tuples
            from outcomes_by_inning.
        prob_cache (dict, optional): pre-computed probabilities keyed the same way
            as simulator()'s cache. If None (default), built here exactly as
            simulator() does (same cache-key tuple, prepare_batted_ball_features,
            HR tail correction). If provided, used as-is — callers that already
            built an identical cache for simulator() should pass it in to avoid
            duplicate predict_proba calls.

    Returns:
        (innings, home_cum, away_cum):
            innings: list[int] 1..n_innings
            home_cum, away_cum: np.ndarray of shape (num_simulations, n_innings) —
                CUMULATIVE deserved runs per simulation through each inning boundary.
        Collapsing these into win/tie percentages (and the tie-counting convention
        for a simulation level on cumulative runs at inning N) is the caller's
        responsibility.
    """
    all_innings = [inn for _, inn in home_outcomes_inn] + [inn for _, inn in away_outcomes_inn]
    n_innings = max(all_innings) if all_innings else 1

    if prob_cache is None:
        # Build prob cache (duplicate of simulator()'s loop, with HR tail correction).
        prob_cache = {}
        for outcome, _ in list(home_outcomes_inn) + list(away_outcomes_inn):
            if isinstance(outcome, dict):
                cache_key = (outcome['launch_speed'], outcome['launch_angle'],
                             outcome['venue_name'], outcome.get('coord_x'),
                             outcome.get('coord_y'), outcome.get('bat_side'))
                if cache_key not in prob_cache:
                    features = prepare_batted_ball_features(
                        launch_speed=outcome['launch_speed'],
                        launch_angle=outcome['launch_angle'],
                        venue_name=outcome['venue_name'],
                        coord_x=outcome.get('coord_x'),
                        coord_y=outcome.get('coord_y'),
                        bat_side=outcome.get('bat_side'),
                    )
                    probs = pipeline.predict_proba(features)[0]
                    prob_cache[cache_key] = _apply_hr_tail_correction(probs, outcome['launch_speed'])

    if _use_scalar_sim():
        # Reference path (DTW_SCALAR_SIM=1): the original per-sim loop.
        home_cum = np.zeros((num_simulations, n_innings), dtype=float)
        away_cum = np.zeros((num_simulations, n_innings), dtype=float)
        for s in range(num_simulations):
            home_cum[s] = simulate_game_by_inning(home_outcomes_inn, prob_cache, n_innings)
            away_cum[s] = simulate_game_by_inning(away_outcomes_inn, prob_cache, n_innings)
    else:
        rng = np.random.default_rng()
        tables = _get_transition_tables()
        home_ev, home_cdf, home_inns = vector_engine.translate_outcomes(
            [o for o, _ in home_outcomes_inn], prob_cache, _outcome_cache_key,
            innings=[inn for _, inn in home_outcomes_inn])
        away_ev, away_cdf, away_inns = vector_engine.translate_outcomes(
            [o for o, _ in away_outcomes_inn], prob_cache, _outcome_cache_key,
            innings=[inn for _, inn in away_outcomes_inn])
        home_cum = vector_engine.simulate_games_by_inning_vectorized(
            home_ev, home_cdf, home_inns, tables, num_simulations, n_innings, rng)
        away_cum = vector_engine.simulate_games_by_inning_vectorized(
            away_ev, away_cdf, away_inns, tables, num_simulations, n_innings, rng)

    return list(range(1, n_innings + 1)), home_cum, away_cum
