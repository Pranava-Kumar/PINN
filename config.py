"""
SMART-DEORBIT SYSTEM — Configuration & Physical Constants
=========================================================
Central repository of all physical constants, mission parameters,
atmospheric models, and PINN hyperparameters.

References:
- IERS Conventions 2010
- US Standard Atmosphere 1976
- NRLMSISE-00 Empirical Atmosphere Model
- IADC Space Debris Mitigation Guidelines
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# SECTION 1: FUNDAMENTAL PHYSICAL CONSTANTS
# =============================================================================

class PhysicalConstants:
    """
    Fundamental physical constants from IERS Conventions 2010.
    All values in SI units unless specified.
    """
    # Earth Gravitational Parameters
    MU_EARTH = 3.986004418e14          # Gravitational parameter [m³/s²]
    GM_EARTH = 3.986004418e14          # Same as MU (standard notation)
    
    # Earth Geometry
    R_EARTH_POLAR = 6.3567523142e6     # Polar radius [m]
    R_EARTH_EQUATOR = 6.3781370e6      # Equatorial radius [m]
    R_EARTH_MEAN = 6.371e6             # Mean radius [m]
    FLATTENING = 1 / 298.257222101     # Earth flattening factor
    
    # Earth Rotation
    OMEGA_EARTH = 7.2921159e-5         # Rotation rate [rad/s]
    DAY_SIDEREAL = 86164.0905          # Sidereal day [s]
    DAY_SOLAR = 86400.0                # Solar day [s]
    
    # Earth Gravity Field Coefficients (EGM96)
    J2 = 1.0826267e-3                  # Second zonal harmonic
    J3 = -2.53266e-6                   # Third zonal harmonic
    J4 = -1.6196e-6                    # Fourth zonal harmonic
    C20 = -J2                          # Alternative notation
    C30 = -J3
    C40 = -J4
    
    # Atmospheric Constants
    R_GAS = 8.314462618                # Universal gas constant [J/(mol·K)]
    R_SPECIFIC_AIR = 287.058           # Specific gas constant for air [J/(kg·K)]
    G0 = 9.80665                       # Standard gravity [m/s²]
    
    # Time Constants
    SECONDS_PER_MINUTE = 60.0
    SECONDS_PER_HOUR = 3600.0
    SECONDS_PER_DAY = 86400.0
    SECONDS_PER_YEAR = 365.25 * 86400.0  # Julian year [s]
    DAYS_PER_YEAR = 365.25


# =============================================================================
# SECTION 2: ATMOSPHERIC MODELS
# =============================================================================

class AtmosphereModelType(Enum):
    EXPONENTIAL = "exponential"
    US_STANDARD_1976 = "us_standard_1976"
    NRLMSISE00_SIMPLIFIED = "nrlmsise00_simplified"


@dataclass
class AtmosphericLayer:
    """Defines a layer in the piecewise atmospheric model."""
    base_altitude_km: float      # Base altitude [km]
    density_base: float          # Density at base [kg/m³]
    scale_height: float          # Scale height [m]
    temperature: Optional[float] = None  # Temperature [K] (optional)


class AtmosphereModel:
    """
    Multi-fidelity atmospheric density models.
    
    References:
    - US Standard Atmosphere 1976
    - NRLMSISE-00 (Picone et al., 2002)
    """
    
    # Piecewise exponential model parameters
    # Format: (base_alt_km, rho0 [kg/m³], scale_height [m])
    EXPONENTIAL_LAYERS = [
        AtmosphericLayer(0,     1.225,       8500,    288.15),
        AtmosphericLayer(100,  5.297e-7,    5900,    210.0),
        AtmosphericLayer(150,  2.070e-9,   26100,    370.0),
        AtmosphericLayer(200,  2.789e-10,  37900,    500.0),
        AtmosphericLayer(250,  7.248e-11,  45500,    700.0),
        AtmosphericLayer(300,  2.418e-11,  53600,    900.0),
        AtmosphericLayer(350,  9.518e-12,  53500,    950.0),
        AtmosphericLayer(400,  3.725e-12,  58200,    1000.0),
        AtmosphericLayer(450,  1.585e-12,  60000,    1050.0),
        AtmosphericLayer(500,  6.967e-13,  63800,    1100.0),
        AtmosphericLayer(550,  3.614e-13,  62200,    1150.0),
        AtmosphericLayer(600,  1.454e-13,  71600,    1200.0),
        AtmosphericLayer(650,  8.770e-14,  78800,    1250.0),
        AtmosphericLayer(700,  5.740e-14,  84400,    1300.0),
        AtmosphericLayer(750,  3.614e-14,  88800,    1350.0),
        AtmosphericLayer(800,  2.418e-14,  93200,    1400.0),
        AtmosphericLayer(850,  1.614e-14,  98500,    1450.0),
        AtmosphericLayer(900,  1.170e-14, 104000,    1500.0),
        AtmosphericLayer(950,  8.770e-15, 110000,    1550.0),
        AtmosphericLayer(1000, 6.570e-15, 116000,    1600.0),
    ]
    
    # Solar activity modifiers (F10.7 cm radio flux proxy)
    SOLAR_ACTIVITY_MIN = 0.7     # Solar minimum multiplier
    SOLAR_ACTIVITY_AVG = 1.0     # Average solar activity
    SOLAR_ACTIVITY_MAX = 1.5     # Solar maximum multiplier
    
    @classmethod
    def get_density(cls, altitude_m: float, 
                    model: AtmosphereModelType = AtmosphereModelType.EXPONENTIAL,
                    solar_activity: float = SOLAR_ACTIVITY_AVG,
                    latitude_deg: float = 45.0,
                    local_hour: float = 12.0) -> float:
        """
        Compute atmospheric density at given altitude.
        
        Parameters
        ----------
        altitude_m : float
            Geometric altitude above Earth's surface [m].
        model : AtmosphereModelType
            Choice of atmospheric model fidelity.
        solar_activity : float
            Solar activity multiplier (0.7-1.5 typical).
        latitude_deg : float
            Geographic latitude for diurnal variation [deg].
        local_hour : float
            Local solar hour (0-24) for diurnal variation.
            
        Returns
        -------
        rho : float
            Atmospheric density [kg/m³].
        """
        alt_km = altitude_m / 1000.0
        
        if alt_km < 0:
            return cls.EXPONENTIAL_LAYERS[0].density_base * solar_activity
        
        if model == AtmosphereModelType.EXPONENTIAL:
            return cls._exponential_density(alt_km, solar_activity)
        elif model == AtmosphereModelType.US_STANDARD_1976:
            return cls._us_standard_1976(alt_km, solar_activity)
        else:
            return cls._nrlmsise_simplified(alt_km, solar_activity, 
                                            latitude_deg, local_hour)
    
    @classmethod
    def _exponential_density(cls, alt_km: float, solar_factor: float) -> float:
        """Piecewise exponential atmospheric model."""
        layers = cls.EXPONENTIAL_LAYERS
        
        if alt_km > layers[-1].base_altitude_km:
            # Extrapolate above highest layer
            last = layers[-1]
            delta_h = (alt_km - last.base_altitude_km) * 1000
            return last.density_base * np.exp(-delta_h / last.scale_height) * solar_factor
        
        # Find appropriate layer
        for i in range(len(layers) - 1, -1, -1):
            if alt_km >= layers[i].base_altitude_km:
                layer = layers[i]
                delta_h = (alt_km - layer.base_altitude_km) * 1000
                return layer.density_base * np.exp(-delta_h / layer.scale_height) * solar_factor
        
        return layers[0].density_base * solar_factor
    
    @classmethod
    def _us_standard_1976(cls, alt_km: float, solar_factor: float) -> float:
        """
        Simplified US Standard Atmosphere 1976 implementation.
        Uses polynomial fits for quick computation.
        """
        if alt_km < 0:
            return 1.225 * solar_factor
        elif alt_km <= 86:
            # Troposphere + Stratosphere (barometric formula)
            T0 = 288.15
            L = 0.0065  # Lapse rate [K/m]
            h = alt_km * 1000
            T = T0 - L * h
            if T <= 0:
                T = 216.65
            p0 = 101325.0
            g0 = PhysicalConstants.G0
            R = PhysicalConstants.R_SPECIFIC_AIR
            p = p0 * (T / T0) ** (g0 / (R * L))
            rho = p / (R * T)
            return rho * solar_factor
        else:
            # Use exponential model above 86 km
            return cls._exponential_density(alt_km, solar_factor)
    
    @classmethod
    def _nrlmsise_simplified(cls, alt_km: float, solar_factor: float,
                             latitude_deg: float, local_hour: float) -> float:
        """
        Simplified NRLMSISE-00 with diurnal and latitudinal variations.
        
        Adds:
        - Diurnal density variation (day/night)
        - Latitudinal variation (equator vs poles)
        """
        base_density = cls._exponential_density(alt_km, solar_factor)
        
        # Diurnal variation (peak at 14:00 local time)
        diurnal_phase = 2 * np.pi * (local_hour - 14) / 24
        diurnal_factor = 1.0 + 0.15 * np.cos(diurnal_phase)
        
        # Latitudinal variation (higher density at equator)
        lat_rad = np.radians(latitude_deg)
        lat_factor = 1.0 + 0.1 * np.cos(2 * lat_rad)
        
        return base_density * diurnal_factor * lat_factor


# =============================================================================
# SECTION 3: MISSION CONFIGURATIONS
# =============================================================================

class MissionType(Enum):
    PSLV_PS4 = "PSLV PS4"
    SSLV_VTM = "SSLV VTM"
    CUSTOM = "Custom"


@dataclass
class SpacecraftConfig:
    """
    Spacecraft/upper stage configuration for de-orbit analysis.
    
    Attributes
    ----------
    name : str
        Mission/stage name.
    dry_mass : float
        Spacecraft dry mass [kg].
    wet_mass : float
        Spacecraft mass including propellant [kg].
    cross_section_no_sail : float
        Cross-sectional area without sail [m²].
    drag_sail_area : float
        Deployable drag sail area [m²].
    Cd : float
        Drag coefficient (typically 2.0-2.5 for satellites).
    max_delta_v : float
        Maximum available ΔV from propulsion [m/s].
    specific_impulse : float
        Engine specific impulse [s].
    orbit_altitude : float
        Initial circular orbit altitude [m].
    orbit_inclination : float
        Orbital inclination [deg].
    orbit_eccentricity : float
        Orbital eccentricity (0 for circular).
    raan : float
        Right Ascension of Ascending Node [deg].
    arg_perigee : float
        Argument of perigee [deg].
    true_anomaly : float
        True anomaly at epoch [deg].
    mission_type : MissionType
        Classification of mission.
    """
    name: str
    dry_mass: float
    wet_mass: float
    cross_section_no_sail: float
    drag_sail_area: float
    Cd: float
    max_delta_v: float
    specific_impulse: float
    orbit_altitude: float
    orbit_inclination: float = 97.5
    orbit_eccentricity: float = 0.0
    raan: float = 0.0
    arg_perigee: float = 0.0
    true_anomaly: float = 0.0
    mission_type: MissionType = MissionType.CUSTOM
    
    @property
    def ballistic_coefficient_sail(self) -> float:
        """Ballistic coefficient with sail deployed [m²/kg]."""
        return self.Cd * self.drag_sail_area / self.dry_mass
    
    @property
    def ballistic_coefficient_no_sail(self) -> float:
        """Ballistic coefficient without sail [m²/kg]."""
        return self.Cd * self.cross_section_no_sail / self.dry_mass
    
    @property
    def delta_v_capability(self) -> float:
        """Maximum ΔV from rocket equation [m/s]."""
        g0 = PhysicalConstants.G0
        return self.specific_impulse * g0 * np.log(self.wet_mass / self.dry_mass)


# Pre-defined mission configurations
PSLV_CONFIG = SpacecraftConfig(
    name="PSLV PS4",
    dry_mass=920.0,
    wet_mass=1150.0,
    cross_section_no_sail=2.0,
    drag_sail_area=10.0,
    Cd=2.2,
    max_delta_v=50.0,
    specific_impulse=285.0,
    orbit_altitude=750e3,
    orbit_inclination=97.5,
    mission_type=MissionType.PSLV_PS4,
)

SSLV_CONFIG = SpacecraftConfig(
    name="SSLV VTM",
    dry_mass=100.0,
    wet_mass=140.0,
    cross_section_no_sail=0.8,
    drag_sail_area=5.0,
    Cd=2.2,
    max_delta_v=15.0,
    specific_impulse=260.0,
    orbit_altitude=500e3,
    orbit_inclination=97.0,
    mission_type=MissionType.SSLV_VTM,
)

# Additional realistic mission scenarios
CARTOSAT_CONFIG = SpacecraftConfig(
    name="CARTOSAT-2",
    dry_mass=680.0,
    wet_mass=750.0,
    cross_section_no_sail=1.5,
    drag_sail_area=8.0,
    Cd=2.2,
    max_delta_v=30.0,
    specific_impulse=280.0,
    orbit_altitude=630e3,
    orbit_inclination=97.9,
    mission_type=MissionType.CUSTOM,
)

RESOURCESAT_CONFIG = SpacecraftConfig(
    name="RESOURCESAT-2",
    dry_mass=1235.0,
    wet_mass=1400.0,
    cross_section_no_sail=3.0,
    drag_sail_area=15.0,
    Cd=2.2,
    max_delta_v=60.0,
    specific_impulse=285.0,
    orbit_altitude=817e3,
    orbit_inclination=98.7,
    mission_type=MissionType.CUSTOM,
)


# =============================================================================
# SECTION 4: DE-ORBIT REQUIREMENTS & CONSTRAINTS
# =============================================================================

@dataclass
class DeorbitRequirements:
    """
    Regulatory and mission-specific de-orbit requirements.
    
    References:
    - IADC Space Debris Mitigation Guidelines
    - ISO 24113:2019 Space Debris Mitigation Requirements
    - NASA STD-8719.14
    """
    # Standard compliance requirements
    max_lifetime_years: float = 25.0        # IADC 25-year rule
    target_lifetime_years: float = 5.0      # Aggressive de-orbit target
    reentry_altitude_km: float = 120.0      # Re-entry interface altitude
    
    # Safety constraints
    min_perigee_km: float = 150.0           # Minimum safe perigee
    max_apogee_km: float = 2000.0           # Maximum allowed apogee
    
    # Operational constraints
    max_burn_duration_s: float = 300.0      # Maximum burn time
    min_coast_orbits: int = 1               # Minimum orbits before burn
    
    # Risk thresholds
    casualty_risk_threshold: float = 1e-4   # Max casualty risk (1 in 10,000)
    fragmentation_risk_alt_km: float = 500  # Altitude where fragmentation risk peaks


# =============================================================================
# SECTION 5: PINN ARCHITECTURE CONFIGURATIONS
# =============================================================================

@dataclass
class PINNConfig:
    """
    Physics-Informed Neural Network configuration.
    
    Attributes
    ----------
    hidden_layers : List[int]
        Number of neurons in each hidden layer.
    activation : str
        Activation function ('tanh', 'relu', 'swish', 'gelu').
    learning_rate : float
        Initial learning rate.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    n_collocation : int
        Number of collocation points for physics loss.
    n_data : int
        Number of supervised data points.
    lambda_physics : float
        Weight for physics loss term.
    lambda_data : float
        Weight for data loss term.
    lambda_initial : float
        Weight for initial condition loss.
    lr_schedule_step : int
        Step size for learning rate scheduler.
    lr_schedule_gamma : float
        Gamma for learning rate scheduler.
    weight_decay : float
        L2 regularization strength.
    dropout_rate : float
        Dropout rate (0 for no dropout).
    use_adaptive_weights : bool
        Whether to use adaptive loss weighting.
    collocation_strategy : str
        Strategy for collocation point selection.
    """
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128, 128, 128])
    activation: str = "tanh"
    learning_rate: float = 1e-3
    epochs: int = 5000
    batch_size: int = 64
    n_collocation: int = 2000
    n_data: int = 50
    lambda_physics: float = 1.0
    lambda_data: float = 1.0
    lambda_initial: float = 10.0
    lr_schedule_step: int = 1000
    lr_schedule_gamma: float = 0.5
    weight_decay: float = 1e-6
    dropout_rate: float = 0.0
    use_adaptive_weights: bool = True
    collocation_strategy: str = "random"  # 'random', 'sobol', 'halton', 'uniform'
    
    # Architecture variants
    use_residual_connections: bool = False
    use_attention: bool = False
    fourier_features: bool = False
    fourier_num_freqs: int = 10


# Pre-defined PINN configurations
PINN_STANDARD = PINNConfig(
    hidden_layers=[128, 128, 128, 128],
    learning_rate=1e-3,
    epochs=5000,
    n_collocation=2000,
    n_data=50,
    lambda_physics=1.0,
)

PINN_DEEP = PINNConfig(
    hidden_layers=[256, 256, 256, 256, 256],
    learning_rate=5e-4,
    epochs=8000,
    n_collocation=5000,
    n_data=100,
    lambda_physics=1.0,
)

PINN_FAST = PINNConfig(
    hidden_layers=[64, 64, 64],
    learning_rate=2e-3,
    epochs=2000,
    n_collocation=500,
    n_data=30,
    lambda_physics=0.5,
)

PINN_HIGH_ACCURACY = PINNConfig(
    hidden_layers=[256, 256, 256, 256, 256, 256],
    learning_rate=5e-4,
    epochs=10000,
    n_collocation=10000,
    n_data=200,
    lambda_physics=10.0,
    lambda_initial=100.0,
    use_adaptive_weights=True,
    fourier_features=True,
    fourier_num_freqs=16,
)


# =============================================================================
# SECTION 6: OPTIMIZATION CONFIGURATIONS
# =============================================================================

@dataclass
class OptimizerConfig:
    """Configuration for ΔV optimization algorithms."""
    # Search parameters
    dv_min: float = 0.0
    dv_max: float = 100.0
    dv_step_coarse: float = 2.0
    dv_step_fine: float = 0.1
    
    # Optimization method
    method: str = "hybrid"  # 'grid', 'brent', 'golden', 'hybrid', 'nelder-mead'
    max_iterations: int = 100
    tolerance: float = 0.01
    
    # Lifetime computation
    max_sim_years: float = 30.0
    dt_output_days: float = 1.0
    
    # Multi-objective weights
    weight_fuel: float = 0.7
    weight_time: float = 0.3
    
    # Constraints
    max_burn_duration_s: float = 300.0
    min_coast_time_days: float = 0.0


# =============================================================================
# SECTION 7: UTILITY FUNCTIONS
# =============================================================================

def get_mission_config(mission_name: str) -> SpacecraftConfig:
    """
    Get spacecraft configuration by name.
    
    Parameters
    ----------
    mission_name : str
        Name of mission ('PSLV', 'SSLV', 'CARTOSAT', 'RESOURCESAT').
        
    Returns
    -------
    config : SpacecraftConfig
        Spacecraft configuration.
    """
    configs = {
        'PSLV': PSLV_CONFIG,
        'SSLV': SSLV_CONFIG,
        'CARTOSAT': CARTOSAT_CONFIG,
        'RESOURCESAT': RESOURCESAT_CONFIG,
    }
    
    key = mission_name.upper().split()[0]
    if key in configs:
        return configs[key]
    raise ValueError(f"Unknown mission: {mission_name}. "
                     f"Available: {list(configs.keys())}")


def get_pinn_config(config_name: str = "standard") -> PINNConfig:
    """
    Get PINN configuration by name.
    
    Parameters
    ----------
    config_name : str
        Configuration name ('standard', 'deep', 'fast', 'high_accuracy').
        
    Returns
    -------
    config : PINNConfig
        PINN configuration.
    """
    configs = {
        'standard': PINN_STANDARD,
        'deep': PINN_DEEP,
        'fast': PINN_FAST,
        'high_accuracy': PINN_HIGH_ACCURACY,
    }
    
    if config_name.lower() in configs:
        return configs[config_name.lower()]
    raise ValueError(f"Unknown PINN config: {config_name}. "
                     f"Available: {list(configs.keys())}")


def circular_orbit_state(altitude: float, 
                       inclination_deg: float = 97.5,
                       raan_deg: float = 0.0,
                       arg_perigee_deg: float = 0.0,
                       true_anomaly_deg: float = 0.0) -> np.ndarray:
    """
    Compute initial ECI state vector for a circular orbit.
    
    Parameters
    ----------
    altitude : float
        Orbital altitude above Earth's surface [m].
    inclination_deg : float
        Orbital inclination [deg].
    raan_deg : float
        Right Ascension of Ascending Node [deg].
    arg_perigee_deg : float
        Argument of perigee [deg] (not used for circular).
    true_anomaly_deg : float
        True anomaly at epoch [deg].
        
    Returns
    -------
    state : np.ndarray, shape (6,)
        [x, y, z, vx, vy, vz] in ECI frame [m, m/s].
    """
    mu = PhysicalConstants.MU_EARTH
    R = PhysicalConstants.R_EARTH_MEAN
    
    r = R + altitude
    v_circular = np.sqrt(mu / r)
    
    # Convert angles to radians
    i = np.radians(inclination_deg)
    omega = np.radians(raan_deg)
    w = np.radians(arg_perigee_deg)
    theta = np.radians(true_anomaly_deg)
    
    # Position in orbital frame (perifocal)
    r_pf = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
    
    # Velocity in orbital frame
    v_pf = np.array([-v_circular * np.sin(theta), 
                     v_circular * np.cos(theta), 
                     0.0])
    
    # Rotation matrices (perifocal to ECI: R = R3(Ω) · R1(i) · R3(ω))
    R3_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])
    
    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    
    R3_w = np.array([
        [np.cos(w), -np.sin(w), 0],
        [np.sin(w), np.cos(w), 0],
        [0, 0, 1]
    ])
    
    # Transform to ECI
    R = R3_omega @ R1_i @ R3_w
    r_eci = R @ r_pf
    v_eci = R @ v_pf
    
    return np.concatenate([r_eci, v_eci])


def compute_reentry_risk(altitude_km: float, 
                         mass_kg: float,
                         cross_section_m2: float) -> Dict[str, float]:
    """
    Estimate re-entry casualty risk based on spacecraft parameters.
    
    Parameters
    ----------
    altitude_km : float
        Current altitude [km].
    mass_kg : float
        Spacecraft mass [kg].
    cross_section_m2 : float
        Cross-sectional area [m²].
        
    Returns
    -------
    risk : dict
        Dictionary containing risk metrics.
    """
    # Simplified casualty risk model (NASA CPR model approximation)
    # Reference: NASA STD-8719.14
    
    # Survivable mass fraction (typically 10-50% survives re-entry)
    survivable_fraction = 0.2
    
    # Casualty area (assume person presents 0.36 m² cross-section)
    casualty_area = 0.36
    
    # Earth surface area (excluding polar regions where debris rarely falls)
    earth_surface_area = 5.1e14  # m²
    
    # Population density weighted factor (higher risk over populated areas)
    population_factor = 0.8  # 80% of Earth's surface has some population
    
    # Compute risk
    surviving_mass = mass_kg * survivable_fraction
    risk = (surviving_mass * casualty_area * population_factor) / earth_surface_area
    
    return {
        'casualty_risk': risk,
        'surviving_mass_kg': surviving_mass,
        'risk_threshold': 1e-4,
        'compliant': risk < 1e-4,
    }


# =============================================================================
# SECTION 8: VALIDATION & TESTING CONSTANTS
# =============================================================================

class ValidationCases:
    """Pre-defined validation test cases for verification."""
    
    # ISS orbit (for validation)
    ISS = {
        'altitude_km': 408,
        'inclination_deg': 51.6,
        'expected_period_min': 92.68,
        'expected_velocity_kms': 7.66,
    }
    
    # Sun-synchronous orbit
    SSO = {
        'altitude_km': 600,
        'inclination_deg': 97.8,
        'expected_period_min': 96.7,
        'expected_velocity_kms': 7.56,
    }
    
    # Geostationary transfer orbit (for testing)
    GTO = {
        'perigee_km': 250,
        'apogee_km': 35786,
        'inclination_deg': 28.5,
        'expected_period_hr': 10.5,
    }
    
    # Graveyard orbit (GEO disposal)
    GEO_GRAVEYARD = {
        'altitude_km': 36100,
        'inclination_deg': 0.0,
        'expected_period_hr': 24.1,
    }


# =============================================================================
# SECTION 9: OUTPUT & REPORTING CONFIGURATION
# =============================================================================

@dataclass
class OutputConfig:
    """Configuration for output files and reports."""
    # Directory structure
    output_dir: str = "outputs"
    models_dir: str = "models"
    plots_dir: str = "plots"
    reports_dir: str = "reports"
    
    # File formats
    save_model_weights: bool = True
    save_training_history: bool = True
    plot_format: str = "png"  # 'png', 'pdf', 'svg'
    data_format: str = "csv"  # 'csv', 'json', 'npz'
    
    # Plot settings
    dpi: int = 300
    figsize_training: Tuple[int, int] = (10, 6)
    figsize_trajectory: Tuple[int, int] = (12, 8)
    figsize_optimization: Tuple[int, int] = (10, 6)
    
    # Report settings
    include_uncertainty: bool = True
    decimal_precision: int = 4
    generate_pdf_report: bool = True


# Default output configuration
OUTPUT_CONFIG = OutputConfig()


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SMART-DEORBIT SYSTEM — Configuration Module Test")
    print("=" * 70)
    
    # Test mission configurations
    print("\n--- Mission Configurations ---")
    for mission in [PSLV_CONFIG, SSLV_CONFIG]:
        print(f"\n{mission.name}:")
        print(f"  Altitude: {mission.orbit_altitude / 1000:.0f} km")
        print(f"  Mass: {mission.dry_mass:.0f} kg")
        print(f"  Sail Area: {mission.drag_sail_area:.1f} m²")
        print(f"  Ballistic Coef (sail): {mission.ballistic_coefficient_sail:.4f} m²/kg")
        print(f"  ΔV Capability: {mission.delta_v_capability:.1f} m/s")
    
    # Test atmosphere model
    print("\n--- Atmospheric Density Profile ---")
    print(f"{'Altitude (km)':>15} | {'Density (kg/m³)':>18}")
    print("-" * 37)
    for alt in [0, 100, 200, 400, 600, 800, 1000]:
        rho = AtmosphereModel.get_density(alt * 1000)
        print(f"{alt:>15} | {rho:>18.6e}")
    
    # Test circular orbit state
    print("\n--- Circular Orbit State (PSLV @ 750 km) ---")
    state = circular_orbit_state(750e3, 97.5)
    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:])
    alt = (r - PhysicalConstants.R_EARTH_MEAN) / 1000
    period = 2 * np.pi * np.sqrt(r**3 / PhysicalConstants.MU_EARTH) / 60
    
    print(f"  Position magnitude: {r / 1000:.1f} km")
    print(f"  Velocity magnitude: {v / 1000:.2f} km/s")
    print(f"  Altitude: {alt:.1f} km")
    print(f"  Orbital Period: {period:.1f} min")
    
    # Test PINN configs
    print("\n--- PINN Configurations ---")
    for name in ['standard', 'deep', 'fast', 'high_accuracy']:
        cfg = get_pinn_config(name)
        print(f"  {name}: {len(cfg.hidden_layers)} layers, "
              f"{sum(cfg.hidden_layers)} neurons, "
              f"{cfg.epochs} epochs")
    
    print("\n" + "=" * 70)
    print("Configuration module test complete!")
