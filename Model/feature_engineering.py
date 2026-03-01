"""
Feature engineering functions for MLB batted ball outcome prediction.
Used by both model training (Base_Model.ipynb) and inference (game_simulator.py).
"""

import numpy as np
import pandas as pd

# =============================================================================
# CONSTANTS
# =============================================================================

# MLB coordinate system: home plate location
HOME_PLATE_X = 125.42
HOME_PLATE_Y = 199.02

# Spray angle thresholds (in degrees, after handedness adjustment)
PULL_THRESHOLD = -15  # More negative = more pulled
OPPO_THRESHOLD = 15   # More positive = more opposite field

# Barrel zone definition
BARREL_MIN_EV = 95
BARREL_MIN_ANGLE = 25
BARREL_MAX_ANGLE = 35

# Launch angle category thresholds
GROUND_BALL_MAX = 10
LINE_DRIVE_MAX = 25
FLY_BALL_MAX = 50


# =============================================================================
# SPRAY ANGLE FUNCTIONS
# =============================================================================

def calculate_spray_angle(coord_x, coord_y):
    """
    Calculate spray angle in degrees from Statcast coordinates.
    
    Args:
        coord_x (float): hitData.coordinates.coordX from MLB API
        coord_y (float): hitData.coordinates.coordY from MLB API
    
    Returns:
        float: Spray angle in degrees
            0° = straight to center field
            Negative = toward left field
            Positive = toward right field
    """
    delta_x = coord_x - HOME_PLATE_X
    delta_y = HOME_PLATE_Y - coord_y  # Flip Y since it decreases into outfield
    
    angle_rad = np.arctan2(delta_x, delta_y)
    return np.degrees(angle_rad)


def adjust_spray_for_handedness(spray_angle, bat_side):
    """
    Adjust spray angle based on batter handedness.
    
    After adjustment:
        Negative = pulled (toward batter's pull side)
        Positive = opposite field
        ~0 = up the middle
    
    Args:
        spray_angle (float): Raw spray angle from calculate_spray_angle()
        bat_side (str): 'L' or 'R' for batter handedness
    
    Returns:
        float: Handedness-adjusted spray angle
    """
    if bat_side == 'L':
        return -spray_angle
    return spray_angle


def categorize_spray_direction(spray_angle_adj):
    """
    Categorize adjusted spray angle into pull/center/oppo zones.
    
    Args:
        spray_angle_adj (float): Handedness-adjusted spray angle
    
    Returns:
        str: 'pull', 'center', or 'oppo'
    """
    if spray_angle_adj < PULL_THRESHOLD:
        return 'pull'
    elif spray_angle_adj > OPPO_THRESHOLD:
        return 'oppo'
    else:
        return 'center'


# =============================================================================
# LAUNCH ANGLE FUNCTIONS
# =============================================================================

def categorize_launch_angle(launch_angle):
    """
    Classify batted balls by launch angle.
    
    Args:
        launch_angle (float): Launch angle in degrees
    
    Returns:
        str: 'ground_ball', 'line_drive', 'fly_ball', or 'popup'
    """
    if launch_angle < GROUND_BALL_MAX:
        return 'ground_ball'
    elif launch_angle < LINE_DRIVE_MAX:
        return 'line_drive'
    elif launch_angle < FLY_BALL_MAX:
        return 'fly_ball'
    else:
        return 'popup'


def calculate_distance_proxy(launch_speed, launch_angle):
    """
    Calculate physics-based distance estimate.
    
    Args:
        launch_speed (float): Exit velocity in mph
        launch_angle (float): Launch angle in degrees
    
    Returns:
        float: Distance proxy value
    """
    angle_rad = np.radians(launch_angle)
    return launch_speed * np.sin(2 * angle_rad)


def is_barrel(launch_speed, launch_angle):
    """
    Check if batted ball is in the barrel zone (optimal HR zone).
    
    Args:
        launch_speed (float): Exit velocity in mph
        launch_angle (float): Launch angle in degrees
    
    Returns:
        int: 1 if barrel, 0 otherwise
    """
    return int(
        (launch_speed >= BARREL_MIN_EV) and 
        (launch_angle >= BARREL_MIN_ANGLE) and 
        (launch_angle <= BARREL_MAX_ANGLE)
    )


# =============================================================================
# MAIN FEATURE CREATION FUNCTION
# =============================================================================

def create_features_for_prediction(launch_speed, launch_angle, coord_x, coord_y, 
                                    bat_side, venue_id):
    """
    Create all required features for model prediction.
    
    This is the main function used by game_simulator.py to prepare data
    for the batted ball outcome model.
    
    Args:
        launch_speed (float): Exit velocity in mph
        launch_angle (float): Launch angle in degrees
        coord_x (float): hitData.coordinates.coordX
        coord_y (float): hitData.coordinates.coordY
        bat_side (str): 'L' or 'R' for batter handedness
        venue_id (str): Stadium venue ID (as string)
    
    Returns:
        pd.DataFrame: Single-row DataFrame with all features for model prediction
    """
    # Core features
    distance_proxy = calculate_distance_proxy(launch_speed, launch_angle)
    launch_category = categorize_launch_angle(launch_angle)
    barrel = is_barrel(launch_speed, launch_angle)
    
    # Spray angle features
    spray_angle = calculate_spray_angle(coord_x, coord_y)
    spray_angle_adj = adjust_spray_for_handedness(spray_angle, bat_side)
    spray_angle_abs = abs(spray_angle_adj)
    spray_direction = categorize_spray_direction(spray_angle_adj)
    
    # Binary spray indicators
    is_pulled = int(spray_angle_adj < PULL_THRESHOLD)
    is_opposite = int(spray_angle_adj > OPPO_THRESHOLD)
    
    # Interaction features
    pulled_hard = int(is_pulled and launch_speed >= 95)
    oppo_hard = int(is_opposite and launch_speed >= 95)
    spray_ev_interaction = spray_angle_adj * launch_speed
    pulled_ground_ball = int(is_pulled and launch_category == 'ground_ball')
    oppo_line_drive = int(is_opposite and launch_category == 'line_drive')

    # Nonlinear EV features for tail accuracy
    launch_speed_squared = launch_speed ** 2
    angle_rad = np.radians(launch_angle)
    hr_distance_proxy = launch_speed * np.sin(angle_rad)

    # Create DataFrame matching model's expected feature order
    return pd.DataFrame({
        # Numeric features (order must match training)
        'hitData_launchSpeed': [launch_speed],
        'hitData_launchAngle': [launch_angle],
        'distance_proxy': [distance_proxy],
        'hr_distance_proxy': [hr_distance_proxy],
        'launch_speed_squared': [launch_speed_squared],
        'spray_angle_adj': [spray_angle_adj],
        'spray_angle_abs': [spray_angle_abs],
        'is_barrel': [barrel],
        'is_pulled': [is_pulled],
        'is_opposite': [is_opposite],
        'pulled_hard': [pulled_hard],
        'oppo_hard': [oppo_hard],
        'spray_ev_interaction': [spray_ev_interaction],
        'pulled_ground_ball': [pulled_ground_ball],
        'oppo_line_drive': [oppo_line_drive],
        # Categorical features
        'launch_angle_category': [launch_category],
        'spray_direction': [spray_direction],
        'venue_id': [str(venue_id)]
    })


def create_features_for_prediction_fallback(launch_speed, launch_angle, venue_id):
    """
    Fallback feature creation when spray angle data is unavailable.
    Uses average/neutral spray values.
    
    Args:
        launch_speed (float): Exit velocity in mph
        launch_angle (float): Launch angle in degrees
        venue_id (str): Stadium venue ID (as string)
    
    Returns:
        pd.DataFrame: Single-row DataFrame with all features for model prediction
    """
    # Use neutral spray angle (center field)
    return create_features_for_prediction(
        launch_speed=launch_speed,
        launch_angle=launch_angle,
        coord_x=HOME_PLATE_X,  # Center = 0 spray angle
        coord_y=HOME_PLATE_Y - 100,  # Into outfield
        bat_side='R',  # Doesn't matter for center
        venue_id=venue_id
    )