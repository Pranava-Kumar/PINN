"""
SMART-DEORBIT SYSTEM — Configuration & Physical Constants
=========================================================
Central store of all physical constants, mission parameters,
and PINN hyperparameters.
"""

import numpy as np

# ─────────────────────────────────────────────
# Physical Constants
# ─────────────────────────────────────────────
MU_EARTH = 3.986004418e14       # Earth gravitational parameter [m³/s²]
R_EARTH = 6.371e6               # Earth mean radius [m]
J2 = 1.0826267e-3               # Earth J2 oblateness coefficient
OMEGA_EARTH = 7.2921159e-5      # Earth rotation rate [rad/s]

# Re-entry threshold
REENTRY_ALTITUDE = 120e3        # [m] — below this altitude, consider re-entered

# ─────────────────────────────────────────────
# Exponential Atmosphere Model
# ─────────────────────────────────────────────
# Piecewise exponential model: (base_alt_km, rho0 [kg/m³], scale_height [m])
# Derived from the U.S. Standard Atmosphere / simplified NRLMSISE-00
ATMOSPHERE_TABLE = [
    (0,    1.225,       8500),
    (100,  5.297e-7,    5900),
    (150,  2.070e-9,   26100),
    (200,  2.789e-10,  37900),
    (250,  7.248e-11,  45500),
    (300,  2.418e-11,  53600),
    (350,  9.518e-12,  53500),
    (400,  3.725e-12,  58200),
    (450,  1.585e-12,  60000),
    (500,  6.967e-13,  63800),
    (550,  3.614e-13,  62200),
    (600,  1.454e-13,  71600),
    (650,  8.770e-14,  78800),
    (700,  5.740e-14,  84400),
    (750,  3.614e-14,  88800),
    (800,  2.418e-14,  93200),
    (850,  1.614e-14,  98500),
    (900,  1.170e-14, 104000),
    (950,  8.770e-15, 110000),
    (1000, 6.570e-15, 116000),
]

# Solar activity multiplier (1.0 = average, >1 for solar max, <1 for solar min)
SOLAR_ACTIVITY_FACTOR = 1.0

# ─────────────────────────────────────────────
# Mission Parameters — PSLV PS4 Stage
# ─────────────────────────────────────────────
PSLV_CONFIG = {
    "name": "PSLV PS4",
    "dry_mass": 920.0,              # [kg]
    "orbit_altitude": 750e3,        # [m] — typical circular orbit
    "drag_sail_area": 10.0,         # [m²]
    "Cd": 2.2,                      # drag coefficient
    "max_delta_v": 50.0,            # [m/s] — max available ΔV
    "cross_section_no_sail": 2.0,   # [m²] — without sail
}

# ─────────────────────────────────────────────
# Mission Parameters — SSLV VTM Stage
# ─────────────────────────────────────────────
SSLV_CONFIG = {
    "name": "SSLV VTM",
    "dry_mass": 100.0,              # [kg]
    "orbit_altitude": 500e3,        # [m] — typical circular orbit
    "drag_sail_area": 5.0,          # [m²]
    "Cd": 2.2,                      # drag coefficient
    "max_delta_v": 15.0,            # [m/s] — limited fuel budget
    "cross_section_no_sail": 0.8,   # [m²] — without sail
}

# ─────────────────────────────────────────────
# PINN Hyperparameters
# ─────────────────────────────────────────────
PINN_HPARAMS = {
    "hidden_layers": [128, 128, 128, 128],
    "activation": "tanh",
    "learning_rate": 1e-3,
    "epochs": 5000,
    "n_collocation": 2000,         # physics collocation points
    "n_data": 50,                  # supervised data points from RK8
    "lambda_physics": 1.0,         # weight of physics loss
    "lr_schedule_step": 1000,      # step LR scheduler every N epochs
    "lr_schedule_gamma": 0.5,      # multiply LR by this factor
}

# ─────────────────────────────────────────────
# Time Scales
# ─────────────────────────────────────────────
SECONDS_PER_YEAR = 365.25 * 24 * 3600
TARGET_LIFETIME_YEARS = 5.0
TARGET_LIFETIME_SECONDS = TARGET_LIFETIME_YEARS * SECONDS_PER_YEAR


def get_mission_config(mission: str) -> dict:
    """Return configuration dict for the selected mission."""
    if mission.upper().startswith("PSLV"):
        return PSLV_CONFIG.copy()
    elif mission.upper().startswith("SSLV"):
        return SSLV_CONFIG.copy()
    else:
        raise ValueError(f"Unknown mission: {mission}. Use 'PSLV' or 'SSLV'.")


def ballistic_coefficient(config: dict, use_sail: bool = True) -> float:
    """Compute the ballistic coefficient B = Cd * A / m  [m²/kg]."""
    A = config["drag_sail_area"] if use_sail else config["cross_section_no_sail"]
    return config["Cd"] * A / config["dry_mass"]


def circular_orbit_state(altitude: float, inclination_deg: float = 97.5) -> np.ndarray:
    """
    Compute initial ECI state vector [x, y, z, vx, vy, vz] for a
    circular orbit at the given altitude and inclination.
    
    Parameters
    ----------
    altitude : float
        Orbital altitude above Earth's surface [m].
    inclination_deg : float
        Orbital inclination [degrees]. Default 97.5° (typical sun-synchronous).
    
    Returns
    -------
    state : np.ndarray, shape (6,)
        [x, y, z, vx, vy, vz] in ECI frame [m, m/s].
    """
    r = R_EARTH + altitude
    v = np.sqrt(MU_EARTH / r)  # circular orbital velocity
    inc = np.radians(inclination_deg)
    
    # Start at ascending node (RAAN=0, arg_perigee=0, true_anomaly=0)
    x = r
    y = 0.0
    z = 0.0
    vx = 0.0
    vy = v * np.cos(inc)
    vz = v * np.sin(inc)
    
    return np.array([x, y, z, vx, vy, vz])
