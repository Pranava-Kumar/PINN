"""
SMART-DEORBIT SYSTEM — Orbital Mechanics Engine
================================================
High-fidelity orbital propagator with comprehensive perturbation models
for accurate prediction of orbital decay and de-orbit trajectories.

Implemented Perturbations:
- Central body gravity (Keplerian)
- Earth oblateness (J2, J3, J4 zonal harmonics)
- Atmospheric drag (exponential + US Standard 1976 + NRLMSISE-00 simplified)
- Third-body effects (optional: Sun, Moon)
- Solar radiation pressure (optional)

Numerical Integrators:
- DOP853 (8th-order Runge-Kutta)
- RK45 (5th-order Runge-Kutta)
- LSODA (adaptive, stiff-aware)

References:
- Vallado, D. A. "Fundamentals of Astrodynamics and Applications"
- Montenbruck, O. & Gill, E. "Satellite Orbits: Models, Methods, Applications"
- NASA SP-8006 "U.S. Standard Atmosphere, 1976"
"""

import numpy as np
from scipy.integrate import solve_ivp, LSODA
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Union
from enum import Enum
import time as time_module
import warnings

from config import (
    PhysicalConstants, AtmosphereModel, AtmosphereModelType,
    SpacecraftConfig, DeorbitRequirements, circular_orbit_state,
    get_mission_config
)


# =============================================================================
# SECTION 1: ENUMS AND DATA CLASSES
# =============================================================================

class IntegratorMethod(Enum):
    """Available numerical integration methods."""
    DOP853 = "DOP853"      # 8th-order Runge-Kutta (default, high accuracy)
    RK45 = "RK45"          # 5th-order Runge-Kutta (faster, lower accuracy)
    RK23 = "RK23"          # 3rd-order Runge-Kutta (fastest, lowest accuracy)
    LSODA = "LSODA"        # Adaptive (handles stiff problems)
    DOPRI5 = "DOPRI5"      # 5th-order Dormand-Prince


class PerturbationModel(Enum):
    """Perturbation model fidelity levels."""
    MINIMAL = "minimal"           # Central body only
    BASIC = "basic"               # J2 + drag
    STANDARD = "standard"         # J2, J3, J4 + drag
    HIGH_FIDELITY = "high_fidelity"  # All + third body + SRP


@dataclass
class PropagationResult:
    """
    Container for orbit propagation results.
    
    Attributes
    ----------
    success : bool
        Whether propagation completed successfully.
    t_seconds : np.ndarray
        Time array [seconds].
    t_years : np.ndarray
        Time array [years].
    states : np.ndarray
        State history, shape (N, 6).
    altitudes_km : np.ndarray
        Altitude history [km].
    velocities_km_s : np.ndarray
        Velocity magnitude history [km/s].
    semi_major_axes_km : np.ndarray
        Semi-major axis history [km].
    eccentricities : np.ndarray
        Eccentricity history.
    inclinations_deg : np.ndarray
        Inclination history [deg].
    raan_deg : np.ndarray
        RAAN history [deg].
    arg_perigee_deg : np.ndarray
        Argument of perigee history [deg].
    true_anomaly_deg : np.ndarray
        True anomaly history [deg].
    reentry_detected : bool
        Whether re-entry was detected.
    lifetime_years : float
        Estimated orbital lifetime [years].
    compute_time_s : float
        Wall-clock computation time [seconds].
    n_function_evaluations : int
        Number of ODE function evaluations.
    message : str
        Solver termination message.
    """
    success: bool
    t_seconds: np.ndarray
    states: np.ndarray
    altitudes_km: np.ndarray
    velocities_km_s: np.ndarray
    semi_major_axes_km: np.ndarray
    eccentricities: np.ndarray
    inclinations_deg: np.ndarray
    reentry_detected: bool
    lifetime_years: float
    compute_time_s: float
    n_function_evaluations: int
    message: str
    
    # Optional fields
    t_years: Optional[np.ndarray] = None
    raan_deg: Optional[np.ndarray] = None
    arg_perigee_deg: Optional[np.ndarray] = None
    true_anomaly_deg: Optional[np.ndarray] = None
    drag_acceleration_history: Optional[np.ndarray] = None
    gravity_acceleration_history: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'lifetime_years': self.lifetime_years,
            'reentry_detected': self.reentry_detected,
            'compute_time_s': self.compute_time_s,
            'n_function_evaluations': self.n_function_evaluations,
            'message': self.message,
            'final_altitude_km': float(self.altitudes_km[-1]) if len(self.altitudes_km) > 0 else None,
            'min_altitude_km': float(np.min(self.altitudes_km)) if len(self.altitudes_km) > 0 else None,
        }


@dataclass
class KeplerianElements:
    """
    Classical Keplerian orbital elements.
    
    Attributes
    ----------
    semi_major_axis_km : float
        Semi-major axis [km].
    eccentricity : float
        Eccentricity.
    inclination_deg : float
        Inclination [deg].
    raan_deg : float
        Right Ascension of Ascending Node [deg].
    arg_perigee_deg : float
        Argument of perigee [deg].
    true_anomaly_deg : float
        True anomaly [deg].
    mean_motion_rev_per_day : float
        Mean motion [revolutions/day].
    period_min : float
        Orbital period [minutes].
    """
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_perigee_deg: float
    true_anomaly_deg: float
    mean_motion_rev_per_day: float
    period_min: float
    
    @property
    def apogee_km(self) -> float:
        """Apogee altitude [km]."""
        r_apogee = self.semi_major_axis_km * (1 + self.eccentricity)
        return r_apogee - PhysicalConstants.R_EARTH_MEAN / 1000
    
    @property
    def perigee_km(self) -> float:
        """Perigee altitude [km]."""
        r_perigee = self.semi_major_axis_km * (1 - self.eccentricity)
        return r_perigee - PhysicalConstants.R_EARTH_MEAN / 1000
    
    @property
    def semi_latus_rectum_km(self) -> float:
        """Semi-latus rectum [km]."""
        return self.semi_major_axis_km * (1 - self.eccentricity**2)
    
    def to_dict(self) -> Dict:
        return {
            'a_km': self.semi_major_axis_km,
            'e': self.eccentricity,
            'i_deg': self.inclination_deg,
            'raan_deg': self.raan_deg,
            'omega_deg': self.arg_perigee_deg,
            'nu_deg': self.true_anomaly_deg,
            'n_rev_day': self.mean_motion_rev_per_day,
            'period_min': self.period_min,
            'apogee_km': self.apogee_km,
            'perigee_km': self.perigee_km,
        }


# =============================================================================
# SECTION 2: GRAVITY MODELS
# =============================================================================

def gravity_central(r_vec: np.ndarray) -> np.ndarray:
    """
    Central body gravitational acceleration (Keplerian).
    
    Parameters
    ----------
    r_vec : np.ndarray, shape (3,)
        Position vector in ECI [m].
        
    Returns
    -------
    a : np.ndarray, shape (3,)
        Gravitational acceleration [m/s²].
    """
    r = np.linalg.norm(r_vec)
    if r < 1e-10:
        return np.zeros(3)
    
    mu = PhysicalConstants.MU_EARTH
    return -mu / r**3 * r_vec


def gravity_j2(r_vec: np.ndarray) -> np.ndarray:
    """
    Gravitational acceleration including J2 (oblateness) perturbation.
    """
    mu = PhysicalConstants.MU_EARTH
    R = PhysicalConstants.R_EARTH_MEAN
    J2 = PhysicalConstants.J2
    
    r = np.linalg.norm(r_vec)
    r = max(r, R)  # Prevent singularity
    
    x, y, z = r_vec
    r2 = r * r
    z2_r2 = (z * z) / r2
    
    # J2 perturbation factor
    j2_factor = 1.5 * J2 * (R ** 2) / r2
    
    # Common term
    common = -mu / (r * r2)
    
    # Acceleration components
    ax = common * x * (1.0 + j2_factor * (1.0 - 5.0 * z2_r2))
    ay = common * y * (1.0 + j2_factor * (1.0 - 5.0 * z2_r2))
    az = common * z * (1.0 + j2_factor * (3.0 - 5.0 * z2_r2))
    
    return np.array([ax, ay, az], dtype=np.float64)


def gravity_zonal_harmonics(r_vec: np.ndarray,
                            max_degree: int = 4) -> np.ndarray:
    """
    Gravitational acceleration from zonal harmonics (J2, J3, J4).
    
    Uses the general formulation for zonal harmonics up to specified degree.
    
    Parameters
    ----------
    r_vec : np.ndarray, shape (3,)
        Position vector in ECI [m].
    max_degree : int
        Maximum degree of zonal harmonics to include.
        
    Returns
    -------
    a : np.ndarray, shape (3,)
        Gravitational acceleration [m/s²].
    """
    mu = PhysicalConstants.MU_EARTH
    R = PhysicalConstants.R_EARTH_MEAN
    
    r = np.linalg.norm(r_vec)
    if r < R:
        r = R
    
    x, y, z = r_vec
    r2 = r * r
    
    # Start with central + J2
    a = gravity_j2(r_vec)
    
    if max_degree < 3:
        return a
    
    # J3 contribution (Montenbruck & Gill formulation)
    if max_degree >= 3 and hasattr(PhysicalConstants, 'J3'):
        J3 = PhysicalConstants.J3
        z_r = z / r
        z2_r2 = z_r * z_r
        
        # J3 perturbation: odd zonal harmonic (asymmetric about equator)
        j3_coeff = 0.5 * J3 * (R / r) ** 3 * (mu / r2)
        
        ax_j3 = -j3_coeff * (x / r) * (5 * z_r * (7 * z2_r2 - 3))
        ay_j3 = -j3_coeff * (y / r) * (5 * z_r * (7 * z2_r2 - 3))
        az_j3 = -j3_coeff * (3 - 30 * z2_r2 + 35 * z2_r2 * z2_r2)
        
        a += np.array([ax_j3, ay_j3, az_j3])
    
    if max_degree < 4:
        return a
    
    # J4 contribution (Montenbruck & Gill formulation)
    if max_degree >= 4 and hasattr(PhysicalConstants, 'J4'):
        J4 = PhysicalConstants.J4
        z_r = z / r
        z2_r2 = z_r * z_r
        z4_r4 = z2_r2 * z2_r2
        
        # J4 perturbation: even zonal harmonic
        j4_coeff = (5.0 / 8.0) * J4 * (R / r) ** 4 * (mu / r2)
        
        ax_j4 = -j4_coeff * (x / r) * (3 - 42 * z2_r2 + 63 * z4_r4)
        ay_j4 = -j4_coeff * (y / r) * (3 - 42 * z2_r2 + 63 * z4_r4)
        az_j4 = -j4_coeff * (z / r) * (15 - 70 * z2_r2 + 63 * z4_r4)
        
        a += np.array([ax_j4, ay_j4, az_j4])
    
    return a


# =============================================================================
# SECTION 3: ATMOSPHERIC DRAG MODEL
# =============================================================================

def atmospheric_drag(r_vec: np.ndarray, 
                     v_vec: np.ndarray,
                     Cd: float,
                     A: float,
                     mass: float,
                     atmosphere_model: AtmosphereModelType = AtmosphereModelType.EXPONENTIAL,
                     include_wind: bool = False,
                     wind_velocity: np.ndarray = None) -> np.ndarray:
    """
    Atmospheric drag acceleration.
    
    The drag acceleration is given by:
        a_drag = -0.5 * rho * (Cd * A / m) * |v_rel| * v_rel
    
    where v_rel is the velocity relative to the rotating atmosphere.
    
    Parameters
    ----------
    r_vec : np.ndarray, shape (3,)
        Position vector in ECI [m].
    v_vec : np.ndarray, shape (3,)
        Velocity vector in ECI [m/s].
    Cd : float
        Drag coefficient (typically 2.0-2.5).
    A : float
        Cross-sectional area [m²].
    mass : float
        Spacecraft mass [kg].
    atmosphere_model : AtmosphereModelType
        Choice of atmospheric density model.
    include_wind : bool
        Whether to include atmospheric wind effects.
    wind_velocity : np.ndarray, optional
        Wind velocity in ECI [m/s].
        
    Returns
    -------
    a_drag : np.ndarray, shape (3,)
        Drag acceleration [m/s²].
    """
    r = np.linalg.norm(r_vec)
    altitude = r - PhysicalConstants.R_EARTH_MEAN
    
    # No drag above 1000 km
    if altitude > 1e6:
        return np.zeros(3)
    
    # Get atmospheric density
    rho = AtmosphereModel.get_density(
        altitude, 
        model=atmosphere_model,
        solar_activity=1.0  # Can be made configurable
    )
    
    # Relative velocity (accounting for Earth's rotation)
    if include_wind and wind_velocity is not None:
        v_rel = v_vec - wind_velocity
    else:
        # Simplified: ignore atmospheric co-rotation for now
        v_rel = v_vec
    
    v_rel_mag = np.linalg.norm(v_rel)
    
    if v_rel_mag < 1e-10:
        return np.zeros(3)
    
    # Ballistic coefficient
    B = Cd * A / mass
    
    # Drag acceleration (opposite to velocity)
    a_drag = -0.5 * rho * B * v_rel_mag * v_rel
    
    return a_drag


def compute_ballistic_coefficient(Cd: float, A: float, mass: float) -> float:
    """
    Compute ballistic coefficient B = Cd * A / m.
    
    Parameters
    ----------
    Cd : float
        Drag coefficient.
    A : float
        Cross-sectional area [m²].
    mass : float
        Mass [kg].
        
    Returns
    -------
    B : float
        Ballistic coefficient [m²/kg].
    """
    return Cd * A / mass


# =============================================================================
# SECTION 4: THIRD-BODY PERTURBATIONS
# =============================================================================

def third_body_acceleration(r_sc: np.ndarray,
                            r_body: np.ndarray,
                            mu_body: float) -> np.ndarray:
    """
    Acceleration due to third-body gravity (Sun or Moon).
    
    Uses the indirect formulation:
        a = mu_3 * [(r_3 - r) / |r_3 - r|^3 - r_3 / |r_3|^3]
    
    Parameters
    ----------
    r_sc : np.ndarray, shape (3,)
        Spacecraft position in ECI [m].
    r_body : np.ndarray, shape (3,)
        Third body position in ECI [m].
    mu_body : float
        Gravitational parameter of third body [m³/s²].
        
    Returns
    -------
    a : np.ndarray, shape (3,)
        Third-body acceleration [m/s²].
    """
    r = np.linalg.norm(r_sc)
    r_3 = np.linalg.norm(r_body)
    
    # Relative position from spacecraft to third body
    r_rel = r_body - r_sc
    d = np.linalg.norm(r_rel)
    
    if d < 1e-10:
        return np.zeros(3)
    
    # Third-body acceleration
    a = mu_body * (r_rel / d**3 - r_body / r_3**3)
    
    return a


def solar_radiation_pressure(r_vec: np.ndarray,
                             r_sun: np.ndarray,
                             Cr: float,
                             A: float,
                             mass: float) -> np.ndarray:
    """
    Solar radiation pressure (SRP) acceleration.
    
    Parameters
    ----------
    r_vec : np.ndarray, shape (3,)
        Spacecraft position in ECI [m].
    r_sun : np.ndarray, shape (3,)
        Sun position in ECI [m].
    Cr : float
        Radiation pressure coefficient (~1.0-2.0).
    A : float
        Spacecraft cross-sectional area [m²].
    mass : float
        Spacecraft mass [kg].
        
    Returns
    -------
    a_srp : np.ndarray, shape (3,)
        SRP acceleration [m/s²].
    """
    # Solar constants
    P0 = 4.56e-6  # Solar radiation pressure at 1 AU [N/m²]
    AU = 1.496e11  # Astronomical Unit [m]
    
    # Sun direction
    r_sun_mag = np.linalg.norm(r_sun)
    if r_sun_mag < 1e-10:
        return np.zeros(3)
    
    sun_hat = r_sun / r_sun_mag
    
    # Check if in Earth's shadow (simplified cylindrical shadow)
    # Project spacecraft position onto sun direction
    proj = np.dot(r_vec, sun_hat)
    
    if proj > 0:
        # Spacecraft is on the sunward side, no shadow
        shadow_factor = 1.0
    else:
        # Check if in umbra
        r_perp = np.linalg.norm(r_vec - proj * sun_hat)
        R_earth = PhysicalConstants.R_EARTH_MEAN
        
        if r_perp < R_earth:
            # In umbra (full shadow)
            shadow_factor = 0.0
        else:
            # In penumbra (partial shadow) - simplified
            shadow_factor = 0.5
    
    # SRP acceleration
    a_srp = -P0 * Cr * (A / mass) * (AU / r_sun_mag)**2 * sun_hat * shadow_factor
    
    return a_srp


# =============================================================================
# SECTION 5: EQUATIONS OF MOTION
# =============================================================================

def equations_of_motion(t: float,
                        state: np.ndarray,
                        Cd: float,
                        A: float,
                        mass: float,
                        perturbation_model: PerturbationModel = PerturbationModel.STANDARD,
                        atmosphere_model: AtmosphereModelType = AtmosphereModelType.EXPONENTIAL,
                        include_third_body: bool = False,
                        include_srp: bool = False,
                        ephemeris_func: Callable = None) -> np.ndarray:
    """
    Complete equations of motion with configurable perturbation models.
    
    Parameters
    ----------
    t : float
        Time [s].
    state : np.ndarray, shape (6,)
        State vector [x, y, z, vx, vy, vz] in ECI.
    Cd : float
        Drag coefficient.
    A : float
        Cross-sectional area [m²].
    mass : float
        Spacecraft mass [kg].
    perturbation_model : PerturbationModel
        Fidelity level for perturbation modeling.
    atmosphere_model : AtmosphereModelType
        Atmospheric density model choice.
    include_third_body : bool
        Include Sun and Moon perturbations.
    include_srp : bool
        Include solar radiation pressure.
    ephemeris_func : Callable, optional
        Function to get Sun/Moon positions.
        
    Returns
    -------
    dstate_dt : np.ndarray, shape (6,)
        Time derivative of state [vx, vy, vz, ax, ay, az].
    """
    r_vec = state[:3]
    v_vec = state[3:]
    
    # Initialize acceleration
    a_total = np.zeros(3)
    
    # Gravity model based on perturbation level
    if perturbation_model == PerturbationModel.MINIMAL:
        a_gravity = gravity_central(r_vec)
    elif perturbation_model == PerturbationModel.BASIC:
        a_gravity = gravity_j2(r_vec)
    else:  # STANDARD or HIGH_FIDELITY
        max_degree = 4 if perturbation_model == PerturbationModel.STANDARD else 4
        a_gravity = gravity_zonal_harmonics(r_vec, max_degree)
    
    a_total += a_gravity
    
    # Atmospheric drag (always include for LEO)
    a_drag = atmospheric_drag(
        r_vec, v_vec, Cd, A, mass,
        atmosphere_model=atmosphere_model
    )
    a_total += a_drag
    
    # Third-body perturbations
    if include_third_body and ephemeris_func is not None:
        sun_pos, moon_pos = ephemeris_func(t)
        mu_sun = 1.32712440018e20  # [m³/s²]
        mu_moon = 4.902800066e12   # [m³/s²]
        
        a_sun = third_body_acceleration(r_vec, sun_pos, mu_sun)
        a_moon = third_body_acceleration(r_vec, moon_pos, mu_moon)
        
        a_total += a_sun + a_moon
    
    # Solar radiation pressure
    if include_srp and ephemeris_func is not None:
        sun_pos, _ = ephemeris_func(t)
        Cr = 1.5  # Typical value
        a_srp = solar_radiation_pressure(r_vec, sun_pos, Cr, A, mass)
        a_total += a_srp
    
    return np.concatenate([v_vec, a_total])


# =============================================================================
# SECTION 6: EVENT DETECTION
# =============================================================================

def _reentry_event(t: float, 
                   state: np.ndarray,
                   reentry_altitude_m: float = 120e3) -> float:
    """
    Event function for re-entry detection.
    
    Triggers when altitude drops below re-entry threshold.
    
    Parameters
    ----------
    t : float
        Time [s].
    state : np.ndarray, shape (6,)
        Current state.
    reentry_altitude_m : float
        Re-entry altitude threshold [m].
        
    Returns
    -------
    value : float
        Altitude - reentry_altitude (zero crossing triggers event).
    """
    r = np.linalg.norm(state[:3])
    altitude = r - PhysicalConstants.R_EARTH_MEAN
    return altitude - reentry_altitude_m

# Configure event properties
_reentry_event.terminal = True  # Stop integration on re-entry
_reentry_event.direction = -1   # Only detect decreasing altitude


def _perigee_event(t: float, state: np.ndarray) -> float:
    """Event function for perigee detection (for orbit analysis)."""
    r = np.linalg.norm(state[:3])
    dr_dt = np.dot(state[:3], state[3:]) / r
    return dr_dt

_perigee_event.terminal = False
_perigee_event.direction = 1  # Increasing radius (leaving perigee)


# =============================================================================
# SECTION 7: ORBIT PROPAGATION
# =============================================================================

def propagate_orbit(state0: np.ndarray,
                    t_span_years: float,
                    Cd: float,
                    A: float,
                    mass: float,
                    integrator: IntegratorMethod = IntegratorMethod.DOP853,
                    perturbation_model: PerturbationModel = PerturbationModel.BASIC,
                    atmosphere_model: AtmosphereModelType = AtmosphereModelType.EXPONENTIAL,
                    dt_output_days: float = 1.0,
                    max_step_days: float = 1.0,
                    rtol: float = 1e-10,
                    atol: float = 1e-12,
                    detect_reentry: bool = True,
                    compute_keplerian: bool = True,
                    verbose: bool = False) -> PropagationResult:
    """
    Propagate orbit using high-order numerical integration.
    
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
    integrator : IntegratorMethod
        Numerical integration method.
    perturbation_model : PerturbationModel
        Perturbation model fidelity.
    atmosphere_model : AtmosphereModelType
        Atmospheric density model.
    dt_output_days : float
        Output time step [days].
    max_step_days : float
        Maximum integration step [days].
    rtol : float
        Relative tolerance for integrator.
    atol : float
        Absolute tolerance for integrator.
    detect_reentry : bool
        Whether to detect re-entry event.
    compute_keplerian : bool
        Whether to compute Keplerian elements.
    verbose : bool
        Print progress information.
        
    Returns
    -------
    result : PropagationResult
        Propagation results container.
    """
    # Time conversion
    t_end = t_span_years * PhysicalConstants.SECONDS_PER_YEAR
    dt_output = dt_output_days * PhysicalConstants.SECONDS_PER_DAY
    max_step = max_step_days * PhysicalConstants.SECONDS_PER_DAY
    
    # Output time points
    t_eval = np.arange(0, t_end + dt_output, dt_output)
    if t_eval[-1] > t_end:
        t_eval = t_eval[:-1]
    
    if verbose:
        print(f"Propagating orbit for {t_span_years:.1f} years...")
        print(f"  Integrator: {integrator.value}")
        print(f"  Perturbations: {perturbation_model.value}")
        print(f"  Output points: {len(t_eval)}")
    
    # Setup events
    events = []
    if detect_reentry:
        # Create closure with reentry altitude
        reentry_alt = DeorbitRequirements().reentry_altitude_km * 1000
        def reentry_func(t, s):
            return _reentry_event(t, s, reentry_alt)
        reentry_func.terminal = True
        reentry_func.direction = -1
        events.append(reentry_func)
    
    if compute_keplerian:
        events.append(_perigee_event)
    
    # Start timing
    start_time = time_module.perf_counter()
    
    # Propagate
    try:
        sol = solve_ivp(
            fun=lambda t, s: equations_of_motion(
                t, s, Cd, A, mass,
                perturbation_model=perturbation_model,
                atmosphere_model=atmosphere_model
            ),
            t_span=[0, t_end],
            y0=state0,
            method=integrator.value,
            t_eval=t_eval,
            events=events if events else None,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        success = sol.success
        message = sol.message
    except Exception as e:
        warnings.warn(f"Propagation failed: {e}")
        success = False
        message = str(e)
        sol = None
    
    # Compute timing
    compute_time = time_module.perf_counter() - start_time
    
    if sol is None:
        # Return empty result on failure
        return PropagationResult(
            success=False,
            t_seconds=np.array([]),
            states=np.empty((0, 6)),
            altitudes_km=np.array([]),
            velocities_km_s=np.array([]),
            semi_major_axes_km=np.array([]),
            eccentricities=np.array([]),
            inclinations_deg=np.array([]),
            reentry_detected=False,
            lifetime_years=0.0,
            compute_time_s=compute_time,
            n_function_evaluations=0,
            message=message,
        )
    
    # Extract solution
    t_seconds = sol.t
    states = sol.y.T  # Shape (N, 6)
    
    # Compute derived quantities
    r_magnitudes = np.linalg.norm(states[:, :3], axis=1)
    v_magnitudes = np.linalg.norm(states[:, 3:], axis=1)
    
    altitudes_km = (r_magnitudes - PhysicalConstants.R_EARTH_MEAN) / 1000.0
    velocities_km_s = v_magnitudes / 1000.0
    
    # Orbital elements
    mu = PhysicalConstants.MU_EARTH
    semi_major_axes_km = np.zeros_like(r_magnitudes)
    eccentricities = np.zeros_like(r_magnitudes)
    
    for i in range(len(states)):
        r = r_magnitudes[i]
        v = v_magnitudes[i]
        
        # Semi-major axis from vis-viva
        energy = v**2 / 2 - mu / r
        if abs(energy) > 1e-10:
            semi_major_axes_km[i] = -mu / (2 * energy) / 1000
        else:
            semi_major_axes_km[i] = np.inf  # Parabolic
        
        # Eccentricity from angular momentum
        h_vec = np.cross(states[i, :3], states[i, 3:])
        h = np.linalg.norm(h_vec)
        e_vec = np.cross(states[i, 3:], h_vec) / mu - states[i, :3] / r
        eccentricities[i] = np.linalg.norm(e_vec)
    
    # Inclination
    h_vecs = np.cross(states[:, :3], states[:, 3:])
    h_magnitudes = np.linalg.norm(h_vecs, axis=1)
    inclinations_deg = np.degrees(np.arccos(np.clip(h_vecs[:, 2] / h_magnitudes, -1, 1)))
    
    # Re-entry detection
    reentry_detected = sol.t_events is not None and len(sol.t_events[0]) > 0 if detect_reentry else False
    
    if reentry_detected:
        lifetime_years = sol.t_events[0][0] / PhysicalConstants.SECONDS_PER_YEAR
    else:
        lifetime_years = t_span_years
    
    # Function evaluations (estimate)
    n_feval = sol.nfev if hasattr(sol, 'nfev') else len(t_seconds)
    
    return PropagationResult(
        success=success,
        t_seconds=t_seconds,
        states=states,
        altitudes_km=altitudes_km,
        velocities_km_s=velocities_km_s,
        semi_major_axes_km=semi_major_axes_km,
        eccentricities=eccentricities,
        inclinations_deg=inclinations_deg,
        reentry_detected=reentry_detected,
        lifetime_years=lifetime_years,
        compute_time_s=compute_time,
        n_function_evaluations=n_feval,
        message=message,
        t_years=t_seconds / PhysicalConstants.SECONDS_PER_YEAR,
    )


# =============================================================================
# SECTION 8: MANEUVERS AND TRANSFERS
# =============================================================================

def apply_delta_v(state: np.ndarray,
                  delta_v_magnitude: float,
                  direction: str = "retrograde",
                  custom_direction: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply an impulsive ΔV maneuver to the spacecraft state.
    
    Parameters
    ----------
    state : np.ndarray, shape (6,)
        Current state [r, v].
    delta_v_magnitude : float
        ΔV magnitude [m/s].
    direction : str
        Burn direction: 'retrograde', 'prograde', 'radial_in', 
        'radial_out', 'normal', 'anti_normal', or 'custom'.
    custom_direction : np.ndarray, optional
        Custom unit vector for burn direction.
        
    Returns
    -------
    new_state : np.ndarray, shape (6,)
        State after maneuver.
    """
    new_state = state.copy()
    
    r_vec = state[:3]
    v_vec = state[3:]
    
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)
    
    if direction == "retrograde":
        dv_vec = -delta_v_magnitude * v_vec / v_mag
    elif direction == "prograde":
        dv_vec = delta_v_magnitude * v_vec / v_mag
    elif direction == "radial_in":
        dv_vec = -delta_v_magnitude * r_vec / r_mag
    elif direction == "radial_out":
        dv_vec = delta_v_magnitude * r_vec / r_mag
    elif direction == "normal":
        h_vec = np.cross(r_vec, v_vec)
        h_mag = np.linalg.norm(h_vec)
        dv_vec = delta_v_magnitude * h_vec / h_mag
    elif direction == "anti_normal":
        h_vec = np.cross(r_vec, v_vec)
        h_mag = np.linalg.norm(h_vec)
        dv_vec = -delta_v_magnitude * h_vec / h_mag
    elif direction == "custom" and custom_direction is not None:
        dv_vec = delta_v_magnitude * custom_direction / np.linalg.norm(custom_direction)
    else:
        raise ValueError(f"Unknown burn direction: {direction}")
    
    new_state[3:] = v_vec + dv_vec
    return new_state


def compute_hohmann_transfer_dv(r1: float, r2: float) -> Tuple[float, float, float]:
    """
    Compute ΔV requirements for a Hohmann transfer between circular orbits.
    
    Parameters
    ----------
    r1 : float
        Initial orbit radius [m].
    r2 : float
        Final orbit radius [m].
        
    Returns
    -------
    dv1 : float
        First burn ΔV [m/s].
    dv2 : float
        Second burn ΔV [m/s].
    dv_total : float
        Total ΔV [m/s].
    """
    mu = PhysicalConstants.MU_EARTH
    
    # Circular velocities
    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)
    
    # Transfer orbit velocities
    a_transfer = (r1 + r2) / 2
    v_transfer_1 = np.sqrt(mu * (2/r1 - 1/a_transfer))
    v_transfer_2 = np.sqrt(mu * (2/r2 - 1/a_transfer))
    
    # ΔV magnitudes
    if r2 > r1:  # Ascending transfer
        dv1 = abs(v_transfer_1 - v1)
        dv2 = abs(v2 - v_transfer_2)
    else:  # Descending transfer (de-orbit)
        dv1 = abs(v1 - v_transfer_1)
        dv2 = abs(v_transfer_2 - v2)
    
    return dv1, dv2, dv1 + dv2


# =============================================================================
# SECTION 9: KEPLERIAN ELEMENTS CONVERSION
# =============================================================================

def state_to_keplerian(state: np.ndarray) -> KeplerianElements:
    """
    Convert ECI state vector to Keplerian orbital elements.
    
    Parameters
    ----------
    state : np.ndarray, shape (6,)
        State vector [x, y, z, vx, vy, vz] in ECI.
        
    Returns
    -------
    elements : KeplerianElements
        Classical orbital elements.
    """
    mu = PhysicalConstants.MU_EARTH
    
    r_vec = state[:3]
    v_vec = state[3:]
    
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    
    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    # Node vector
    k_vec = np.array([0, 0, 1.0])
    n_vec = np.cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec)
    
    # Eccentricity vector
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = np.linalg.norm(e_vec)
    
    # Semi-major axis
    energy = v**2 / 2 - mu / r
    if abs(energy) < 1e-10:
        a = np.inf  # Parabolic
    else:
        a = -mu / (2 * energy)
    
    # Inclination
    i = np.degrees(np.arccos(np.clip(h_vec[2] / h, -1, 1)))
    
    # RAAN
    if n > 1e-10:
        raan = np.degrees(np.arccos(np.clip(n_vec[0] / n, -1, 1)))
        if n_vec[1] < 0:
            raan = 360 - raan
    else:
        raan = 0.0
    
    # Argument of perigee
    if n > 1e-10 and e > 1e-10:
        omega = np.degrees(np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1)))
        if e_vec[2] < 0:
            omega = 360 - omega
    else:
        omega = 0.0
    
    # True anomaly
    if e > 1e-10:
        nu = np.degrees(np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1, 1)))
        if np.dot(r_vec, v_vec) < 0:
            nu = 360 - nu
    else:
        # Circular orbit: use argument of latitude
        if n > 1e-10:
            nu = np.degrees(np.arccos(np.clip(np.dot(n_vec, r_vec) / (n * r), -1, 1)))
            if r_vec[2] < 0:
                nu = 360 - nu
        else:
            nu = 0.0
    
    # Mean motion and period
    if np.isinf(a):
        n_rev_day = 0
        period_min = 0
    else:
        n_rad_s = np.sqrt(mu / a**3)
        n_rev_day = n_rad_s * PhysicalConstants.SECONDS_PER_DAY / (2 * np.pi)
        period_min = 2 * np.pi * np.sqrt(a**3 / mu) / 60
    
    return KeplerianElements(
        semi_major_axis_km=a / 1000,
        eccentricity=e,
        inclination_deg=i,
        raan_deg=raan,
        arg_perigee_deg=omega,
        true_anomaly_deg=nu,
        mean_motion_rev_per_day=n_rev_day,
        period_min=period_min,
    )


def keplerian_to_state(elements: KeplerianElements) -> np.ndarray:
    """
    Convert Keplerian elements to ECI state vector.
    
    Parameters
    ----------
    elements : KeplerianElements
        Orbital elements.
        
    Returns
    -------
    state : np.ndarray, shape (6,)
        State vector [x, y, z, vx, vy, vz] in ECI.
    """
    mu = PhysicalConstants.MU_EARTH
    
    a = elements.semi_major_axis_km * 1000
    e = elements.eccentricity
    i = np.radians(elements.inclination_deg)
    raan = np.radians(elements.raan_deg)
    omega = np.radians(elements.arg_perigee_deg)
    nu = np.radians(elements.true_anomaly_deg)
    
    # Semi-latus rectum
    p = a * (1 - e**2)
    
    # Position and velocity in perifocal frame
    r = p / (1 + e * np.cos(nu))
    
    r_pf = np.array([
        r * np.cos(nu),
        r * np.sin(nu),
        0
    ])
    
    v_pf = np.array([
        -np.sqrt(mu / p) * np.sin(nu),
        np.sqrt(mu / p) * (e + np.cos(nu)),
        0
    ])
    
    # Rotation matrices
    R3_raan = np.array([
        [np.cos(raan), -np.sin(raan), 0],
        [np.sin(raan), np.cos(raan), 0],
        [0, 0, 1]
    ])
    
    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    
    R3_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation
    R = R3_raan @ R1_i @ R3_omega
    
    # Transform to ECI
    r_eci = R @ r_pf
    v_eci = R @ v_pf
    
    return np.concatenate([r_eci, v_eci])


# =============================================================================
# SECTION 10: LIFETIME COMPUTATION
# =============================================================================

def compute_lifetime(state0: np.ndarray,
                     Cd: float,
                     A: float,
                     mass: float,
                     max_years: float = 30.0,
                     verbose: bool = False) -> float:
    """
    Compute orbital lifetime until re-entry.
    
    Parameters
    ----------
    state0 : np.ndarray, shape (6,)
        Initial state.
    Cd : float
        Drag coefficient.
    A : float
        Cross-sectional area [m²].
    mass : float
        Mass [kg].
    max_years : float
        Maximum simulation time [years].
    verbose : bool
        Print progress.
        
    Returns
    -------
    lifetime : float
        Orbital lifetime [years].
    """
    result = propagate_orbit(
        state0, max_years, Cd, A, mass,
        dt_output_days=5.0,
        max_step_days=2.0,
        verbose=verbose,
    )
    return result.lifetime_years


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SMART-DEORBIT SYSTEM — Orbital Mechanics Module Test")
    print("=" * 70)
    
    # Test gravity models
    print("\n--- Gravity Model Comparison ---")
    r_test = np.array([PhysicalConstants.R_EARTH_MEAN + 400e3, 0, 0])
    
    a_central = gravity_central(r_test)
    a_j2 = gravity_j2(r_test)
    a_full = gravity_zonal_harmonics(r_test, 4)
    
    print(f"At 400 km altitude:")
    print(f"  Central only: {np.linalg.norm(a_central):.4f} m/s²")
    print(f"  With J2: {np.linalg.norm(a_j2):.4f} m/s²")
    print(f"  With J2-J4: {np.linalg.norm(a_full):.4f} m/s²")
    
    # Test atmosphere model
    print("\n--- Atmospheric Density Profile ---")
    print(f"{'Altitude (km)':>15} | {'Density (kg/m³)':>18} | {'Scale Height (km)':>17}")
    print("-" * 55)
    for alt in [0, 100, 200, 300, 400, 500, 600, 800, 1000]:
        rho = AtmosphereModel.get_density(alt * 1000)
        # Estimate scale height from density gradient
        if alt < 1000:
            rho_next = AtmosphereModel.get_density((alt + 50) * 1000)
            H = -50000 / np.log(rho_next / rho) / 1000
        else:
            H = np.nan
        print(f"{alt:>15} | {rho:>18.6e} | {H:>17.1f}")
    
    # Test orbit propagation
    print("\n--- Orbit Propagation Test (PSLV PS4 @ 750 km) ---")
    state0 = circular_orbit_state(750e3, 97.5)
    elements = state_to_keplerian(state0)
    
    print(f"Initial Keplerian Elements:")
    elem_dict = elements.to_dict()
    for key, value in elem_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # Propagate with drag
    print(f"\nPropagating with 10 m² sail for 15 years...")
    result = propagate_orbit(
        state0,
        t_span_years=15.0,
        Cd=2.2,
        A=10.0,
        mass=920.0,
        dt_output_days=5.0,
        max_step_days=1.0,
        verbose=True,
    )
    
    print(f"\nPropagation Results:")
    print(f"  Success: {result.success}")
    print(f"  Re-entry detected: {result.reentry_detected}")
    print(f"  Lifetime: {result.lifetime_years:.2f} years")
    print(f"  Compute time: {result.compute_time_s:.2f} s")
    print(f"  Function evaluations: {result.n_function_evaluations}")
    print(f"  Final altitude: {result.altitudes_km[-1]:.1f} km")
    print(f"  Min altitude: {np.min(result.altitudes_km):.1f} km")
    
    # Test Delta-V maneuver
    print("\n--- Delta-V Maneuver Test ---")
    dv_test = 10.0  # m/s
    new_state = apply_delta_v(state0, dv_test, "retrograde")
    new_elements = state_to_keplerian(new_state)
    
    print(f"Applied {dv_test} m/s retrograde burn:")
    print(f"  Original perigee: {elements.perigee_km:.1f} km")
    print(f"  New perigee: {new_elements.perigee_km:.1f} km")
    print(f"  Perigee drop: {elements.perigee_km - new_elements.perigee_km:.1f} km")
    
    print("\n" + "=" * 70)
    print("Orbital mechanics module test complete!")
