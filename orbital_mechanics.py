"""
SMART-DEORBIT SYSTEM — Orbital Mechanics Engine
=================================================
Implements gravity (with J2), atmospheric drag, and RK8 numerical
propagation for orbital decay simulation.
"""

import numpy as np
from scipy.integrate import solve_ivp
from config import (
    MU_EARTH, R_EARTH, J2, ATMOSPHERE_TABLE,
    SOLAR_ACTIVITY_FACTOR, REENTRY_ALTITUDE, SECONDS_PER_YEAR
)


# ─────────────────────────────────────────────
# Atmosphere Model
# ─────────────────────────────────────────────
def atmospheric_density(altitude_m: float) -> float:
    """
    Compute atmospheric density at a given altitude using a
    piecewise exponential model.
    
    Parameters
    ----------
    altitude_m : float
        Altitude above Earth's surface [m].
    
    Returns
    -------
    rho : float
        Atmospheric density [kg/m³].
    """
    alt_km = altitude_m / 1000.0
    
    if alt_km < 0:
        return ATMOSPHERE_TABLE[0][1]  # surface density
    
    if alt_km > ATMOSPHERE_TABLE[-1][0]:
        # Extrapolate from last band
        _, rho0, H = ATMOSPHERE_TABLE[-1]
        return rho0 * np.exp(-(altitude_m - ATMOSPHERE_TABLE[-1][0] * 1000) / H) * SOLAR_ACTIVITY_FACTOR
    
    # Find the appropriate band
    for i in range(len(ATMOSPHERE_TABLE) - 1, -1, -1):
        if alt_km >= ATMOSPHERE_TABLE[i][0]:
            base_alt_km, rho0, H = ATMOSPHERE_TABLE[i]
            delta_h = altitude_m - base_alt_km * 1000.0
            return rho0 * np.exp(-delta_h / H) * SOLAR_ACTIVITY_FACTOR
    
    return ATMOSPHERE_TABLE[0][1] * SOLAR_ACTIVITY_FACTOR


# ─────────────────────────────────────────────
# Acceleration Components
# ─────────────────────────────────────────────
def accel_gravity_j2(r_vec: np.ndarray) -> np.ndarray:
    """
    Gravitational acceleration including J2 perturbation.
    
    Parameters
    ----------
    r_vec : np.ndarray, shape (3,)
        Position vector in ECI [m].
    
    Returns
    -------
    a : np.ndarray, shape (3,)
        Acceleration [m/s²].
    """
    r = np.linalg.norm(r_vec)
    x, y, z = r_vec
    
    r2 = r * r
    r5 = r2 * r2 * r
    
    # J2 factor
    j2_factor = 1.5 * J2 * (R_EARTH ** 2) / r2
    z2_over_r2 = (z * z) / r2
    
    # Central gravity + J2
    common = -MU_EARTH / (r * r2)
    
    ax = common * x * (1.0 + j2_factor * (1.0 - 5.0 * z2_over_r2))
    ay = common * y * (1.0 + j2_factor * (1.0 - 5.0 * z2_over_r2))
    az = common * z * (1.0 + j2_factor * (3.0 - 5.0 * z2_over_r2))
    
    return np.array([ax, ay, az])


def accel_drag(r_vec: np.ndarray, v_vec: np.ndarray,
               Cd: float, A: float, mass: float) -> np.ndarray:
    """
    Atmospheric drag acceleration.
    
    a_drag = -0.5 * rho * (Cd * A / m) * |v_rel| * v_rel
    
    Parameters
    ----------
    r_vec : np.ndarray, shape (3,)
        Position vector in ECI [m].
    v_vec : np.ndarray, shape (3,)
        Velocity vector in ECI [m/s].
    Cd : float
        Drag coefficient.
    A : float
        Cross-sectional area [m²].
    mass : float
        Spacecraft mass [kg].
    
    Returns
    -------
    a : np.ndarray, shape (3,)
        Drag acceleration [m/s²].
    """
    r = np.linalg.norm(r_vec)
    altitude = r - R_EARTH
    
    if altitude < 0:
        altitude = 0.0
    
    rho = atmospheric_density(altitude)
    
    # Relative velocity (ignore Earth rotation for simplicity in this demo)
    v_rel = v_vec
    v_rel_mag = np.linalg.norm(v_rel)
    
    if v_rel_mag < 1e-10:
        return np.zeros(3)
    
    B = Cd * A / mass  # ballistic coefficient
    a_drag = -0.5 * rho * B * v_rel_mag * v_rel
    
    return a_drag


# ─────────────────────────────────────────────
# Equations of Motion
# ─────────────────────────────────────────────
def equations_of_motion(t, state, Cd, A, mass):
    """
    Full equations of motion: gravity (J2) + drag.
    
    Parameters
    ----------
    t : float
        Time [s].
    state : array-like, shape (6,)
        [x, y, z, vx, vy, vz] in ECI.
    Cd : float
        Drag coefficient.
    A : float
        Cross-sectional area [m²].
    mass : float
        Spacecraft mass [kg].
    
    Returns
    -------
    dstate_dt : np.ndarray, shape (6,)
        Time derivative of the state vector.
    """
    r_vec = state[:3]
    v_vec = state[3:]
    
    a_grav = accel_gravity_j2(r_vec)
    a_drag_vec = accel_drag(r_vec, v_vec, Cd, A, mass)
    
    a_total = a_grav + a_drag_vec
    
    return np.concatenate([v_vec, a_total])


# ─────────────────────────────────────────────
# Re-entry Event Detection
# ─────────────────────────────────────────────
def _reentry_event(t, state, Cd, A, mass):
    """Event function: triggers when altitude drops below REENTRY_ALTITUDE."""
    r = np.linalg.norm(state[:3])
    return r - R_EARTH - REENTRY_ALTITUDE

_reentry_event.terminal = True
_reentry_event.direction = -1


# ─────────────────────────────────────────────
# RK8 (DOP853) Propagator
# ─────────────────────────────────────────────
def propagate_orbit(state0: np.ndarray, t_span_years: float,
                    Cd: float, A: float, mass: float,
                    dt_output_days: float = 1.0,
                    max_step_days: float = 0.5) -> dict:
    """
    Propagate an orbit using DOP853 (8th-order Runge-Kutta).
    
    Parameters
    ----------
    state0 : np.ndarray, shape (6,)
        Initial state [x, y, z, vx, vy, vz] in ECI [m, m/s].
    t_span_years : float
        Total propagation time [years].
    Cd : float
        Drag coefficient.
    A : float
        Cross-sectional area [m²].
    mass : float
        Spacecraft mass [kg].
    dt_output_days : float
        Output time step [days].
    max_step_days : float
        Maximum integration step [days].
    
    Returns
    -------
    result : dict
        - 't_years': array of times [years]
        - 't_seconds': array of times [s]
        - 'states': array of shape (N, 6) — [r, v] at each time
        - 'altitudes_km': altitude above Earth [km]
        - 'semi_major_axes_km': semi-major axis [km]
        - 'reentry': bool — whether re-entry was detected
        - 'lifetime_years': orbital lifetime [years]
        - 'compute_time_s': wall-clock computation time [s]
    """
    import time as time_module
    
    t_end = t_span_years * SECONDS_PER_YEAR
    dt_output = dt_output_days * 86400.0
    max_step = max_step_days * 86400.0
    
    t_eval = np.arange(0, t_end + dt_output, dt_output)
    if t_eval[-1] > t_end:
        t_eval = t_eval[:-1]

    
    start_time = time_module.perf_counter()
    
    sol = solve_ivp(
        equations_of_motion,
        t_span=[0, t_end],
        y0=state0,
        method='DOP853',
        t_eval=t_eval,
        args=(Cd, A, mass),
        events=_reentry_event,
        rtol=1e-10,
        atol=1e-12,
        max_step=max_step,
    )
    
    compute_time = time_module.perf_counter() - start_time
    
    # Extract results
    t_seconds = sol.t
    states = sol.y.T  # shape (N, 6)
    
    # Compute derived quantities
    r_magnitudes = np.linalg.norm(states[:, :3], axis=1)
    altitudes_km = (r_magnitudes - R_EARTH) / 1000.0
    
    # Semi-major axis from vis-viva: a = 1 / (2/r - v²/μ)
    v_magnitudes = np.linalg.norm(states[:, 3:], axis=1)
    semi_major_axes = 1.0 / (2.0 / r_magnitudes - v_magnitudes**2 / MU_EARTH)
    semi_major_axes_km = semi_major_axes / 1000.0
    
    # Determine re-entry
    reentry = len(sol.t_events[0]) > 0
    if reentry:
        lifetime_years = sol.t_events[0][0] / SECONDS_PER_YEAR
    else:
        lifetime_years = t_span_years  # didn't re-enter within simulation window
    
    return {
        't_years': t_seconds / SECONDS_PER_YEAR,
        't_seconds': t_seconds,
        'states': states,
        'altitudes_km': altitudes_km,
        'semi_major_axes_km': semi_major_axes_km,
        'reentry': reentry,
        'lifetime_years': lifetime_years,
        'compute_time_s': compute_time,
    }


def compute_lifetime(state0: np.ndarray, Cd: float, A: float, mass: float,
                     max_years: float = 30.0) -> float:
    """
    Compute orbital lifetime by propagating until re-entry or max_years.
    
    Returns
    -------
    lifetime : float
        Orbital lifetime [years]. Returns max_years if no re-entry detected.
    """
    result = propagate_orbit(state0, max_years, Cd, A, mass,
                             dt_output_days=5.0, max_step_days=1.0)
    return result['lifetime_years']


def apply_delta_v(state: np.ndarray, delta_v_ms: float) -> np.ndarray:
    """
    Apply a retrograde ΔV (perigee-lowering burn) to the state.
    The burn is applied opposite to the velocity direction.
    
    Parameters
    ----------
    state : np.ndarray, shape (6,)
        Current state vector [r, v].
    delta_v_ms : float
        Magnitude of ΔV [m/s], applied retrograde.
    
    Returns
    -------
    new_state : np.ndarray, shape (6,)
        State after the burn.
    """
    new_state = state.copy()
    v_vec = state[3:]
    v_hat = v_vec / np.linalg.norm(v_vec)
    new_state[3:] = v_vec - delta_v_ms * v_hat  # retrograde
    return new_state


# ─────────────────────────────────────────────
# Keplerian Elements
# ─────────────────────────────────────────────
def state_to_keplerian(state: np.ndarray) -> dict:
    """
    Convert ECI state vector to Keplerian orbital elements.
    
    Returns
    -------
    elements : dict
        - 'a': semi-major axis [km]
        - 'e': eccentricity
        - 'i': inclination [deg]
        - 'altitude_km': current altitude [km]
        - 'period_min': orbital period [minutes]
    """
    r_vec = state[:3]
    v_vec = state[3:]
    
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    
    # Semi-major axis (vis-viva)
    a = 1.0 / (2.0 / r - v**2 / MU_EARTH)
    
    # Angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    # Eccentricity vector
    e_vec = np.cross(v_vec, h_vec) / MU_EARTH - r_vec / r
    e = np.linalg.norm(e_vec)
    
    # Inclination
    i = np.degrees(np.arccos(h_vec[2] / h))
    
    # Orbital period
    period_s = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
    
    return {
        'a': a / 1000.0,
        'e': e,
        'i': i,
        'altitude_km': (r - R_EARTH) / 1000.0,
        'period_min': period_s / 60.0,
    }


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from config import circular_orbit_state, PSLV_CONFIG, SSLV_CONFIG
    
    print("=" * 60)
    print("SMART-DEORBIT — Orbital Mechanics Quick Test")
    print("=" * 60)
    
    # Test PSLV PS4 at 750 km
    state0 = circular_orbit_state(PSLV_CONFIG["orbit_altitude"])
    elements = state_to_keplerian(state0)
    print(f"\nPSLV PS4 Initial Orbit:")
    print(f"  Altitude: {elements['altitude_km']:.1f} km")
    print(f"  Semi-major axis: {elements['a']:.1f} km")
    print(f"  Period: {elements['period_min']:.1f} min")
    
    # Propagate PSLV with sail for 15 years
    print(f"\nPropagating PSLV PS4 with 10 m² sail (max 15 years)...")
    result = propagate_orbit(
        state0, 15.0,
        Cd=PSLV_CONFIG["Cd"],
        A=PSLV_CONFIG["drag_sail_area"],
        mass=PSLV_CONFIG["dry_mass"],
        dt_output_days=5.0
    )
    print(f"  Re-entry: {result['reentry']}")
    print(f"  Lifetime: {result['lifetime_years']:.2f} years")
    print(f"  Compute time: {result['compute_time_s']:.2f} s")
    
    # Test SSLV VTM at 500 km
    state0_sslv = circular_orbit_state(SSLV_CONFIG["orbit_altitude"])
    print(f"\nPropagating SSLV VTM with 5 m² sail (max 15 years)...")
    result_sslv = propagate_orbit(
        state0_sslv, 15.0,
        Cd=SSLV_CONFIG["Cd"],
        A=SSLV_CONFIG["drag_sail_area"],
        mass=SSLV_CONFIG["dry_mass"],
        dt_output_days=5.0
    )
    print(f"  Re-entry: {result_sslv['reentry']}")
    print(f"  Lifetime: {result_sslv['lifetime_years']:.2f} years")
    print(f"  Compute time: {result_sslv['compute_time_s']:.2f} s")
    
    print("\n" + "=" * 60)
    print("Quick test complete!")
