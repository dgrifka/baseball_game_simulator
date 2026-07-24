"""
Feature engineering functions for MLB batted ball outcome prediction.
Used by both model training (Base_Model.ipynb) and inference (game_simulator.py).
"""

import math
import os

import numpy as np
import pandas as pd

from Model.bbe_physics import nathan_spin, spin_aware_carry

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
# PARK GEOMETRY + CARRY PHYSICS
# =============================================================================
#
# These features encode stadium geometry (outfield-wall distance at the ball's
# spray angle, park altitude) and a drag-adjusted carry estimate, all derivable
# at inference from {EV, launch_angle, coord_x, coord_y, bat_side, venue_id}.
# They replace the old `venue_id` one-hot blob in the F3 model. Training and
# inference both call this same function, so they produce identical feature
# values (no train/serve skew).

# hc-units -> feet (non-corner-calibrated; matches Bill Petti R package).
COORD_TO_FT = 2.495

# League-median wall distance (feet) used as the fallback when a venue is not in
# the park-wall table (e.g. neutral-site games, relocated parks) or the ray-cast
# misses. Derived once as the median of wall_distance_ft over known-venue batted
# balls across 2024-2026 (= 374.009 ft); a fixed constant so single-row inference
# matches without needing a dataset-level median.
LEAGUE_MEDIAN_WALL_FT = 374.0

# --- drag-carry physics constants --------------------------------------------
_MASS = 0.145          # kg
_CD = 0.30             # drag coefficient
_AREA = 0.00426        # m^2 ball cross-section
_G = 9.81              # m/s^2
_RHO0 = 1.225          # kg/m^3 air density at sea level
_SCALE_HEIGHT = 8500.0  # m, exponential atmosphere scale height
_MPH_TO_MS = 0.44704
_FT_PER_M = 3.28084
_FT_TO_M = 0.3048
_CONTACT_HEIGHT_M = 1.0  # contact point height
_DT = 0.005            # integration timestep (s)
_MAX_STEPS = 4000      # 20 s ceiling — far beyond any real fly ball

# Magnus lift is OFF: the Phase-2 bake-off showed it does NOT improve log-loss,
# ECE, or the HR tail. The shipped carry is drag-only.
_USE_LIFT = False


def _load_walls(path):
    """Return {venue_id(str): Nx2 ndarray of (x, y) wall vertices in hc units}."""
    w = pd.read_csv(path, dtype={"venue_id": str})
    w = w.sort_values(["venue_id", "point_order"])
    return {vid: g[["x", "y"]].to_numpy() for vid, g in w.groupby("venue_id")}


def _load_altitudes(path):
    """Return {venue_id(str): elevation_ft(float)}."""
    a = pd.read_csv(path, dtype={"venue_id": str})
    return dict(zip(a["venue_id"], a["elevation_ft"].astype(float)))


_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
try:
    _WALLS = _load_walls(os.path.join(_DATA_DIR, "mlb_park_walls.csv"))
    _ALTITUDES = _load_altitudes(os.path.join(_DATA_DIR, "park_altitudes.csv"))
except (FileNotFoundError, OSError):
    # Graceful degradation: without the CSVs, geo features fall back to the
    # league-median wall / zero altitude (carry still computes). The model's
    # ColumnTransformer selects by name, so this only affects the geo columns.
    _WALLS = {}
    _ALTITUDES = {}


def _ray_polygon_distance(pts, angle_deg):
    """Cast a ray from home plate at `angle_deg`; return distance (hc units) to
    the nearest wall-polygon intersection, or None.

    Ray: P(t) = HOME + t * (sin(theta), -cos(theta)), t >= 0 (0deg -> CF,
    +angle -> right field, matching calculate_spray_angle).
    """
    th = np.radians(angle_deg)
    dx, dy = np.sin(th), -np.cos(th)
    closed = np.vstack([pts, pts[0]])
    best = None
    for i in range(len(closed) - 1):
        x1, y1 = closed[i]
        x2, y2 = closed[i + 1]
        ex, ey = x2 - x1, y2 - y1
        denom = dx * (-ey) - dy * (-ex)
        if abs(denom) < 1e-9:
            continue
        t = ((x1 - HOME_PLATE_X) * (-ey) - (y1 - HOME_PLATE_Y) * (-ex)) / denom
        u = (dx * (y1 - HOME_PLATE_Y) - dy * (x1 - HOME_PLATE_X)) / denom
        if t >= 0 and -1e-9 <= u <= 1 + 1e-9:
            if best is None or t < best:
                best = t
    return best


def wall_distance_at_spray(venue_id, spray_angle_deg):
    """Distance (FEET) from home plate to the outfield wall at `spray_angle_deg`
    for `venue_id`. Falls back to LEAGUE_MEDIAN_WALL_FT for unknown venues or
    ray-cast misses (so the feature never produces NaN).

    The ray-cast uses spray rounded to 1 decimal, matching the training-side
    feature builder's per-(venue, rounded-spray) cache.
    """
    pts = _WALLS.get(str(venue_id))
    if pts is None:
        return LEAGUE_MEDIAN_WALL_FT
    d_hc = _ray_polygon_distance(pts, round(float(spray_angle_deg), 1))
    if d_hc is None:
        return LEAGUE_MEDIAN_WALL_FT
    return d_hc * COORD_TO_FT


def _drag_carry_scalar(launch_speed_mph, launch_angle_deg, elevation_ft):
    """Single ball: drag-adjusted carry distance in feet (Euler integration)."""
    if not np.isfinite(launch_speed_mph) or not np.isfinite(launch_angle_deg):
        return np.nan
    elevation_m = elevation_ft * _FT_TO_M
    rho = _RHO0 * np.exp(-elevation_m / _SCALE_HEIGHT)
    k = 0.5 * rho * _CD * _AREA

    v0 = launch_speed_mph * _MPH_TO_MS
    theta = np.radians(launch_angle_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    x = 0.0
    y = _CONTACT_HEIGHT_M

    for _ in range(_MAX_STEPS):
        speed = np.hypot(vx, vy)
        ax = -(k / _MASS) * speed * vx
        ay = -_G - (k / _MASS) * speed * vy
        if _USE_LIFT and speed > 1e-9:
            S = (0.0366 * 37.0) / speed
            Cl = 1.0 / (2.32 + 0.40 / S)
            k_lift = 0.5 * rho * Cl * _AREA * speed * speed
            ux, uy = vx / speed, vy / speed
            ax += (k_lift / _MASS) * (-uy)
            ay += (k_lift / _MASS) * (ux)
        x_new = x + vx * _DT
        y_new = y + vy * _DT
        if y_new <= 0.0:
            frac = y / (y - y_new) if (y - y_new) != 0 else 0.0
            x = x + (x_new - x) * frac
            return x * _FT_PER_M
        x, y = x_new, y_new
        vx += ax * _DT
        vy += ay * _DT
    return x * _FT_PER_M


def drag_carry(launch_speed_mph, launch_angle_deg, elevation_ft=0.0):
    """Drag-adjusted projectile carry distance in FEET (drag-only; lift off).

    Strictly increasing in launch_speed at fixed angle and increasing with
    elevation (thinner air -> more carry). Scalar in, scalar out.
    """
    return _drag_carry_scalar(
        float(launch_speed_mph), float(launch_angle_deg), float(elevation_ft)
    )


# =============================================================================
# GAME-TIME TEMPERATURE GUARD (shared by training, re-score, and live scoring)
# =============================================================================

ROOF_CLOSED_CONDITIONS = {"Dome", "Roof Closed"}
_ROOF_PLAUSIBLE_RANGE = (50.0, 95.0)   # indoor readings outside this are garbage
_TEMP_CLIP_RANGE = (35.0, 105.0)       # open-air plausibility clip
_TEMP_DEFAULT = 70.0                   # legacy fixed value (temp_f=70 == no-temp physics)
_ROOF_INDOOR_TEMP = 72.0               # canonical climate-controlled reading


def is_roof_closed(condition):
    """True when the weather `condition` string means climate-controlled air."""
    return condition in ROOF_CLOSED_CONDITIONS


def sanitize_temp(temp_f, roof_closed=False):
    """Return a plausible game-time temperature (F) for the carry physics.

    NaN/None -> 70.0; roof-closed readings outside [50, 95] -> 72.0 (the API
    sometimes reports outdoor temp for closed-roof games, e.g. Chase 2022
    106-109F, and 0F for two 2024 dome games); everything else clipped to
    [35, 105].
    """
    if temp_f is None:
        return _TEMP_DEFAULT
    t = float(temp_f)
    if math.isnan(t):
        return _TEMP_DEFAULT
    if roof_closed and not (_ROOF_PLAUSIBLE_RANGE[0] <= t <= _ROOF_PLAUSIBLE_RANGE[1]):
        return _ROOF_INDOOR_TEMP
    return float(min(max(t, _TEMP_CLIP_RANGE[0]), _TEMP_CLIP_RANGE[1]))


# =============================================================================
# MAIN FEATURE CREATION FUNCTION
# =============================================================================

def create_features_for_prediction(launch_speed, launch_angle, coord_x, coord_y,
                                    bat_side, venue_id, temp_f=70.0, roof_closed=False):
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
        temp_f (float): Game-time temperature in Fahrenheit (default 70.0)
        roof_closed (bool): Whether the stadium roof is closed (default False)

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

    # Park-geometry + carry features (F3 model). venue_id is kept in the output
    # below as well so the legacy 18-feature model keeps working — each model's
    # ColumnTransformer selects only the columns it was trained on.
    altitude_ft = float(_ALTITUDES.get(str(venue_id), 0.0))
    wall_distance_ft = wall_distance_at_spray(venue_id, spray_angle)
    carry_ft = drag_carry(launch_speed, launch_angle, altitude_ft)
    over_fence_margin = carry_ft - wall_distance_ft

    # Spin + temperature carry features (F6 model). Additive: older models'
    # ColumnTransformers select by name and ignore these.
    temp_f_safe = sanitize_temp(temp_f, roof_closed)
    backspin_rpm, sidespin_rpm = nathan_spin(launch_angle, spray_angle_adj, bat_side)
    carry_ft_spin = spin_aware_carry(
        launch_speed, launch_angle, spray_angle_adj, bat_side, altitude_ft)
    carry_ft_spin_temp = spin_aware_carry(
        launch_speed, launch_angle, spray_angle_adj, bat_side, altitude_ft,
        temp_f_safe)

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
        # Park-geometry + carry features (F3 model)
        'altitude_ft': [altitude_ft],
        'wall_distance_ft': [wall_distance_ft],
        'carry_ft': [carry_ft],
        'over_fence_margin': [over_fence_margin],
        # Spin + temperature carry features (F6 model)
        'carry_ft_spin': [carry_ft_spin],
        'over_fence_margin_spin': [carry_ft_spin - wall_distance_ft],
        'total_spin_rpm': [float(np.sqrt(backspin_rpm ** 2 + sidespin_rpm ** 2))],
        'sidespin_abs_rpm': [abs(sidespin_rpm)],
        'carry_ft_spin_temp': [carry_ft_spin_temp],
        'over_fence_margin_spin_temp': [carry_ft_spin_temp - wall_distance_ft],
        'temp_f': [temp_f_safe],
        # Categorical features
        'launch_angle_category': [launch_category],
        'spray_direction': [spray_direction],
        'venue_id': [str(venue_id)]
    })


def create_features_for_prediction_fallback(launch_speed, launch_angle, venue_id,
                                             temp_f=70.0, roof_closed=False):
    """
    Fallback feature creation when spray angle data is unavailable.
    Uses average/neutral spray values.

    Args:
        launch_speed (float): Exit velocity in mph
        launch_angle (float): Launch angle in degrees
        venue_id (str): Stadium venue ID (as string)
        temp_f (float): Game-time temperature in Fahrenheit (default 70.0)
        roof_closed (bool): Whether the stadium roof is closed (default False)

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
        venue_id=venue_id,
        temp_f=temp_f,
        roof_closed=roof_closed
    )