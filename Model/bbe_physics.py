"""
Physics-informed batted-ball spin + trajectory model, seeded from Alan
Nathan's public research (Trajectory Calculator spreadsheet,
`TrajectoryCalculator-new-3D.xlsx`, baseball.physics.illinois.edu, and
"Analysis of Baseball Trajectories", 2017).

Two public functions:
    nathan_spin(launch_angle, spray_adj, bat_side) -> (backspin_rpm, sidespin_rpm)
    spin_aware_carry(ev_mph, launch_angle, spray_adj, bat_side, altitude_ft,
                      temp_f=70.0) -> carry_ft

Built for docs/superpowers/specs/2026-07-15-bbe-tail-round2-physics-design.md
(Task 1). All inputs are array-like or scalar; both functions preserve the
scalar/array shape of their inputs.

CONVENTION MAPPING (spray_adj <-> Nathan's raw phi)
-----------------------------------------------------
Nathan's phi: 0 = dead center, positive = toward 1B/RF, negative = toward
3B/LF (foul lines at +/-45).

Our repo's raw spray angle (`Model/feature_engineering.py`,
`calculate_spray_angle`) is defined identically: 0 = center field, negative
= left field, positive = right field. So Nathan's phi === our RAW
(pre-handedness-adjustment) spray angle directly, with no sign flip.

`spray_adj` (our repo's `spray_angle_adj` column, what THIS module receives)
is the handedness-ADJUSTED spray from `adjust_spray_for_handedness`:

    spray_adj = spray_angle          if bat_side == 'R'
    spray_adj = -spray_angle         if bat_side == 'L'

  (after adjustment: negative = pulled, positive = opposite field, for
  BOTH handedness — see that function's docstring).

Inverting for phi (= raw spray_angle) in terms of spray_adj:

    phi = spray_adj                  if bat_side == 'R'   (sign=+1)
    phi = -spray_adj                 if bat_side == 'L'   (sign=-1)

  i.e. phi = sign * spray_adj, where sign = +1 for R, -1 for L — the same
  `sign` used in Nathan's spin formulas. This also means phi*sign =
  sign*sign*spray_adj = spray_adj (sign**2 == 1), so the backspin formula's
  `phi*sign` term literally equals `spray_adj` — consistent with the brief's
  note "phi*sign > 0 means toward opposite field", since spray_adj > 0 IS
  opposite field by construction.

Worked example — RHB pulls to LEFT field:
  RHB, spray_adj = -20 (pulled, per adjust_spray_for_handedness convention).
  sign = +1 (R)  =>  phi = sign * spray_adj = 1 * (-20) = -20.
  Nathan's phi = -20 = toward LEFT field/3B line. Correct: RHB pull side is
  left field, and Nathan's phi is negative there. Matches.

Worked example — LHB pulls to RIGHT field:
  LHB, spray_adj = -20 (pulled, same adjusted-spray convention: negative
  always means pulled, regardless of hand).
  sign = -1 (L)  =>  phi = sign * spray_adj = -1 * (-20) = +20.
  Nathan's phi = +20 = toward RIGHT field/1B line. Correct: LHB pull side is
  right field, and Nathan's phi is positive there. Matches.

A useful corollary used by the trajectory model: since |sidespin| depends on
phi only through `sign*849 + 94*phi` in absolute value, and phi = sign *
spray_adj, the `sign` factors cancel entirely in the magnitude — |sidespin|
as a function of spray_adj is IDENTICAL for R and L batters. Handedness only
flips the ROTATIONAL SENSE (direction the ball curves), never the magnitude.
This is what test 3 (oppo > pull sidespin) exercises for both hands.

TRAJECTORY COORDINATE SYSTEM
-----------------------------------------------------
x: horizontal, positive toward 1B/RF (same direction as Nathan's phi = +).
y: horizontal, positive toward dead center field (phi = 0 direction).
z: vertical, positive up. Launch point is the origin in (x, y); z starts at
   LAUNCH_HEIGHT_FT. This keeps the phi -> (vx, vy) decomposition structurally
   identical to `calculate_spray_angle`'s own arctan2(delta_x, delta_y).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (English/mixed units to match Nathan's spreadsheet).
# ---------------------------------------------------------------------------
G_FT_S2 = 32.174               # gravitational acceleration, ft/s^2
MASS_OZ = 5.125                # standard baseball mass, oz
CIRC_IN = 9.125                # standard baseball circumference, in
BALL_RADIUS_FT = (CIRC_IN / (2.0 * np.pi)) / 12.0  # ft

DT = 0.01                      # Euler integration step, s (per spec)
LAUNCH_HEIGHT_FT = 3.0         # initial height of contact, ft
MAX_STEPS = 3000               # safety cap: 30s of flight (real flights ~4-7s)

MPH_TO_FTS = 5280.0 / 3600.0

# Standard atmospheric conditions used for the air-density formula (spec).
_T_C = 21.1     # 70 F (legacy reference anchor -- see air_density_K temp_f)
_T_F_REF = 70.0  # Fahrenheit reference point _T_C was calibrated at
_P_MMHG = 760.0
_RH_PCT = 50.0

# kg/m^3 -> lb/ft^3 mass-density conversion factor.
_KGM3_TO_LBFT3 = 0.062428


# ---------------------------------------------------------------------------
# internal helper (defined early: used by both public functions below)
# ---------------------------------------------------------------------------
def _validate_bat_side(bat_side_arr):
    """Raise ValueError if any element of `bat_side_arr` is not 'R' or 'L'.

    Without this guard, `sign = np.where(bat_side_arr == "R", 1.0, -1.0)`
    silently maps EVERY non-'R' value — typos, other handedness codes, or
    NaN — to the L (-1) branch. bat_side has no documented NaN-as-default
    convention here (unlike spray_adj -> center or altitude_ft -> sea level
    above), so NaN is rejected too, not silently treated as L."""
    valid = np.isin(bat_side_arr, ["R", "L"])
    if not valid.all():
        bad = np.unique(np.asarray(bat_side_arr, dtype=object)[~valid])
        raise ValueError(
            f"bat_side must be 'R' or 'L' for every row; got invalid "
            f"value(s): {bad.tolist()}"
        )


# ---------------------------------------------------------------------------
# nathan_spin
# ---------------------------------------------------------------------------
def nathan_spin(launch_angle, spray_adj, bat_side):
    """
    Seed backspin/sidespin (rpm) from Nathan's Trajectory Calculator
    regressions.

    Args:
        launch_angle: launch angle, degrees.
        spray_adj: handedness-adjusted spray angle, degrees (our repo's
            `spray_angle_adj` convention: negative = pulled, positive =
            opposite field). NaN is treated as center (0).
        bat_side: 'R' or 'L' (array-like of strings, or scalar). Any other
            value (including NaN) raises ValueError.

    Returns:
        (backspin_rpm, sidespin_rpm) — same shape as the broadcast inputs.
    """
    scalar_out = _all_scalar(launch_angle, spray_adj, bat_side)

    theta = np.atleast_1d(np.asarray(launch_angle, dtype=float))
    spray_adj_arr = np.atleast_1d(np.asarray(spray_adj, dtype=float))
    spray_adj_arr = np.where(np.isnan(spray_adj_arr), 0.0, spray_adj_arr)
    bat_side_arr = np.atleast_1d(np.asarray(bat_side))
    _validate_bat_side(bat_side_arr)

    theta, spray_adj_arr, bat_side_arr = np.broadcast_arrays(
        theta, spray_adj_arr, bat_side_arr
    )

    sign = np.where(bat_side_arr == "R", 1.0, -1.0)  # +1 RHB, -1 LHB
    phi = sign * spray_adj_arr  # Nathan's raw spray angle (see module docstring)

    backspin_rpm = -763.0 + 120.0 * theta + 21.0 * (phi * sign)
    sidespin_rpm = -sign * 849.0 - 94.0 * phi

    if scalar_out:
        return float(backspin_rpm[0]), float(sidespin_rpm[0])
    return backspin_rpm, sidespin_rpm


# ---------------------------------------------------------------------------
# air_density_K — air density -> drag/lift scale constant K = c0 = 0.5*rho*A/m
# ---------------------------------------------------------------------------
def air_density_K(altitude_ft, temp_f=70.0):
    """
    K = c0 (ft^-1), the aerodynamic scale constant in the equations of
    motion: K = 0.5 * rho * A / m, computed from Nathan's air-density
    formula at standard pressure/humidity, adjusted for park altitude and
    game-time temperature.

    NaN/missing altitude defaults to 0 ft (sea level). NaN/missing temp_f
    defaults to 70 F (the historical fixed value this module used before
    the temp_f parameter existed).

    temp_f == 70.0 (the default) reproduces the pre-round-4 output
    bit-for-bit: t_c is computed as an offset from the legacy _T_C = 21.1
    reference (t_c = _T_C + (temp_f - 70) * 5/9) rather than a fresh
    F->C conversion, so the zero-offset case collapses to exactly _T_C.
    The offset formula has the same 5/9 C-per-F slope as a direct
    conversion, so deltas (e.g. 90F vs 70F carry) are unaffected by this
    choice.
    """
    altitude_ft_arr = np.atleast_1d(np.asarray(altitude_ft, dtype=float))
    altitude_ft_arr = np.where(np.isnan(altitude_ft_arr), 0.0, altitude_ft_arr)

    temp_f_arr = np.atleast_1d(np.asarray(temp_f, dtype=float))
    temp_f_arr = np.where(np.isnan(temp_f_arr), _T_F_REF, temp_f_arr)

    altitude_ft_arr, temp_f_arr = np.broadcast_arrays(altitude_ft_arr, temp_f_arr)
    elev_m = altitude_ft_arr * 0.3048
    t_c = _T_C + (temp_f_arr - _T_F_REF) * 5.0 / 9.0

    svp = 4.5841 * np.exp((18.687 - t_c / 234.5) * t_c / (257.14 + t_c))
    rho_kgm3 = (
        1.2929
        * (273.0 / (t_c + 273.0))
        * (_P_MMHG * np.exp(-1.217e-4 * elev_m) - 0.3783 * _RH_PCT * svp / 100.0)
        / 760.0
    )
    rho_lbft3 = rho_kgm3 * _KGM3_TO_LBFT3

    # K = 0.07182 * rho[lb/ft^3] * (5.125/mass_oz) * (circ_in/9.125)^2;
    # with mass_oz=5.125, circ_in=9.125 the mass/circumference correction
    # factors are both 1.0, and this reproduces the sea-level anchor
    # K ~ 5.37e-3 ft^-1 to within 0.3% — equivalent to the direct physical
    # derivation K = 0.5*rho*A/m in slug/ft units.
    K = 0.07182 * rho_lbft3

    if np.ndim(altitude_ft) == 0 and np.ndim(temp_f) == 0:
        return float(K[0])
    return K


# ---------------------------------------------------------------------------
# spin_aware_carry
# ---------------------------------------------------------------------------
def spin_aware_carry(ev_mph, launch_angle, spray_adj, bat_side, altitude_ft, temp_f=70.0):
    """
    Horizontal landing distance (feet) from a Nathan-seeded-spin,
    Magnus-lift trajectory, integrated with Euler dt=0.01s and a
    position half-step, matching Nathan's spreadsheet.

    Args:
        ev_mph: exit velocity, mph.
        launch_angle: launch angle, degrees.
        spray_adj: handedness-adjusted spray angle, degrees (NaN -> center).
        bat_side: 'R' or 'L'. Any other value (including NaN) raises
            ValueError.
        altitude_ft: park altitude, feet (NaN/missing -> 0).
        temp_f: game-time temperature, Fahrenheit (NaN/missing -> 70.0).
            Passed through to `air_density_K`; temp_f=70.0 (the default)
            reproduces pre-round-4 output bit-for-bit.

    Returns:
        carry_ft_spin — same shape as the broadcast inputs.
    """
    scalar_out = _all_scalar(ev_mph, launch_angle, spray_adj, bat_side, altitude_ft, temp_f)

    ev_arr = np.atleast_1d(np.asarray(ev_mph, dtype=float))
    la_arr = np.atleast_1d(np.asarray(launch_angle, dtype=float))
    spray_arr = np.atleast_1d(np.asarray(spray_adj, dtype=float))
    spray_arr = np.where(np.isnan(spray_arr), 0.0, spray_arr)
    bat_side_arr = np.atleast_1d(np.asarray(bat_side))
    _validate_bat_side(bat_side_arr)
    alt_arr = np.atleast_1d(np.asarray(altitude_ft, dtype=float))
    alt_arr = np.where(np.isnan(alt_arr), 0.0, alt_arr)
    temp_arr = np.atleast_1d(np.asarray(temp_f, dtype=float))

    ev_arr, la_arr, spray_arr, bat_side_arr, alt_arr, temp_arr = np.broadcast_arrays(
        ev_arr, la_arr, spray_arr, bat_side_arr, alt_arr, temp_arr
    )
    shape = ev_arr.shape
    n = ev_arr.size

    ev_flat = ev_arr.reshape(n)
    la_flat = la_arr.reshape(n)
    spray_flat = spray_arr.reshape(n)
    bat_side_flat = bat_side_arr.reshape(n)
    alt_flat = alt_arr.reshape(n)
    temp_flat = temp_arr.reshape(n)

    backspin, sidespin = nathan_spin(la_flat, spray_flat, bat_side_flat)
    backspin = np.asarray(backspin, dtype=float)
    sidespin = np.asarray(sidespin, dtype=float)

    sign = np.where(bat_side_flat == "R", 1.0, -1.0)
    phi = sign * spray_flat
    phi_rad = np.radians(phi)
    theta_rad = np.radians(la_flat)

    omega_total_rpm = np.sqrt(backspin ** 2 + sidespin ** 2)
    omega_rad_s = omega_total_rpm * 2.0 * np.pi / 60.0

    # Spin components (Nathan Eq., rpm units — the ratio in the equations of
    # motion below cancels units, so rpm throughout is consistent).
    wx = backspin * np.cos(phi_rad) - sidespin * np.sin(theta_rad) * np.sin(phi_rad)
    wy = -backspin * np.sin(phi_rad) - sidespin * np.sin(theta_rad) * np.cos(phi_rad)
    wz = sidespin * np.cos(theta_rad)

    CD = 0.297 + 0.0292 * (omega_total_rpm / 1000.0)
    K = air_density_K(alt_flat, temp_flat)

    has_spin = omega_total_rpm > 0.0
    omega_safe = np.where(has_spin, omega_total_rpm, 1.0)

    V = ev_flat * MPH_TO_FTS
    vx = V * np.cos(theta_rad) * np.sin(phi_rad)
    vy = V * np.cos(theta_rad) * np.cos(phi_rad)
    vz = V * np.sin(theta_rad)

    x = np.zeros(n)
    y = np.zeros(n)
    z = np.full(n, LAUNCH_HEIGHT_FT)

    landed = np.zeros(n, dtype=bool)
    carry = np.zeros(n)

    for _ in range(MAX_STEPS):
        v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        v_safe = np.where(v > 0.0, v, 1e-9)

        S = np.where(has_spin, BALL_RADIUS_FT * omega_rad_s / v_safe, 0.0)
        CL = np.where(has_spin, 1.120 * S / (0.583 + 2.333 * S), 0.0)

        ax = -K * CD * v * vx + K * CL * v * (wy * vz - wz * vy) / omega_safe
        ay = -K * CD * v * vy + K * CL * v * (wz * vx - wx * vz) / omega_safe
        az = -K * CD * v * vz + K * CL * v * (wx * vy - wy * vx) / omega_safe - G_FT_S2

        x_prev, y_prev, z_prev = x, y, z

        x = x_prev + vx * DT + 0.5 * ax * DT ** 2
        y = y_prev + vy * DT + 0.5 * ay * DT ** 2
        z = z_prev + vz * DT + 0.5 * az * DT ** 2

        vx = vx + ax * DT
        vy = vy + ay * DT
        vz = vz + az * DT

        newly_landed = (~landed) & (z_prev >= 0.0) & (z < 0.0)
        if np.any(newly_landed):
            denom = z_prev[newly_landed] - z[newly_landed]
            frac = z_prev[newly_landed] / denom
            x_land = x_prev[newly_landed] + frac * (x[newly_landed] - x_prev[newly_landed])
            y_land = y_prev[newly_landed] + frac * (y[newly_landed] - y_prev[newly_landed])
            carry[newly_landed] = np.sqrt(x_land ** 2 + y_land ** 2)
            landed[newly_landed] = True

        if landed.all():
            break

    if not landed.all():
        unl = ~landed
        carry[unl] = np.sqrt(x[unl] ** 2 + y[unl] ** 2)

    if scalar_out:
        return float(carry[0])
    return carry.reshape(shape)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _all_scalar(*args):
    return all(np.ndim(a) == 0 for a in args)
