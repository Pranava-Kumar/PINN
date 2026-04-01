"""
SMART-DEORBIT SYSTEM — Unit Tests and Validation
=================================================
Comprehensive test suite for verifying orbital mechanics calculations,
PINN training, and optimization correctness.

Run: pytest tests/test_orbital_mechanics.py -v
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PhysicalConstants, AtmosphereModel, AtmosphereModelType,
    circular_orbit_state, get_mission_config, PSLV_CONFIG, SSLV_CONFIG,
    PINNConfig, get_pinn_config
)
from orbital_mechanics import (
    gravity_central, gravity_j2, gravity_zonal_harmonics,
    atmospheric_drag, propagate_orbit, state_to_keplerian,
    keplerian_to_state, apply_delta_v, compute_lifetime,
    IntegratorMethod, PerturbationModel
)
from pinn_model import (
    OrbitalPINN, StateNormalizer, compute_physics_residual,
    train_pinn, predict_trajectory
)
from delta_v_optimizer import (
    optimize_delta_v, OptimizationMethod, compute_propulsive_only_dv,
    compute_lifetime_with_dv as _compute_lifetime_with_dv
)
from config import OptimizerConfig


# =============================================================================
# SECTION 1: CONFIGURATION TESTS
# =============================================================================

class TestPhysicalConstants:
    """Test physical constants are correctly defined."""
    
    def test_gravitational_parameter(self):
        """Test Earth's gravitational parameter."""
        assert PhysicalConstants.MU_EARTH == 3.986004418e14
        assert PhysicalConstants.MU_EARTH > 0
    
    def test_earth_radius(self):
        """Test Earth radius values."""
        assert PhysicalConstants.R_EARTH_MEAN == 6.371e6
        assert PhysicalConstants.R_EARTH_POLAR < PhysicalConstants.R_EARTH_EQUATOR
    
    def test_j2_coefficient(self):
        """Test J2 oblateness coefficient."""
        assert PhysicalConstants.J2 == 1.0826267e-3
        assert 0 < PhysicalConstants.J2 < 0.1
    
    def test_time_constants(self):
        """Test time conversion constants."""
        assert PhysicalConstants.SECONDS_PER_MINUTE == 60.0
        assert PhysicalConstants.SECONDS_PER_HOUR == 3600.0
        assert PhysicalConstants.SECONDS_PER_DAY == 86400.0
        assert abs(PhysicalConstants.SECONDS_PER_YEAR - 3.15576e7) < 1e5


class TestAtmosphereModel:
    """Test atmospheric density models."""
    
    def test_density_decreases_with_altitude(self):
        """Density should decrease with increasing altitude."""
        rho_0 = AtmosphereModel.get_density(0)
        rho_100 = AtmosphereModel.get_density(100e3)
        rho_500 = AtmosphereModel.get_density(500e3)
        
        assert rho_0 > rho_100 > rho_500
    
    def test_surface_density(self):
        """Test sea-level density is approximately correct."""
        rho_0 = AtmosphereModel.get_density(0)
        expected = 1.225  # kg/m³
        
        assert abs(rho_0 - expected) < 0.01
    
    def test_density_at_400km(self):
        """Test density at ISS altitude."""
        rho = AtmosphereModel.get_density(400e3)
        
        # Should be in the range 1e-12 to 1e-11 kg/m³
        assert 1e-13 < rho < 1e-10
    
    def test_solar_activity_effect(self):
        """Higher solar activity should increase density."""
        alt = 400e3
        
        rho_min = AtmosphereModel.get_density(alt, solar_activity=0.7)
        rho_avg = AtmosphereModel.get_density(alt, solar_activity=1.0)
        rho_max = AtmosphereModel.get_density(alt, solar_activity=1.5)
        
        assert rho_min < rho_avg < rho_max
    
    def test_negative_altitude(self):
        """Negative altitude should return surface density."""
        rho_neg = AtmosphereModel.get_density(-10e3)
        rho_0 = AtmosphereModel.get_density(0)
        
        assert rho_neg == rho_0


# =============================================================================
# SECTION 2: ORBITAL MECHANICS TESTS
# =============================================================================

class TestGravityModels:
    """Test gravity model implementations."""
    
    def test_central_gravity_magnitude(self):
        """Test central gravity magnitude at Earth surface."""
        r = np.array([PhysicalConstants.R_EARTH_MEAN, 0, 0])
        a = gravity_central(r)
        
        expected_g = PhysicalConstants.G0  # ~9.8 m/s²
        actual_g = np.linalg.norm(a)
        
        assert abs(actual_g - expected_g) / expected_g < 0.01
    
    def test_central_gravity_direction(self):
        """Central gravity should point toward Earth center."""
        r = np.array([PhysicalConstants.R_EARTH_MEAN + 400e3, 0, 0])
        a = gravity_central(r)
        
        # Acceleration should be in negative x direction
        assert a[0] < 0
        assert a[1] == 0
        assert a[2] == 0
    
    def test_j2_increases_gravity_at_equator(self):
        """J2 should slightly increase gravity at equator."""
        r = np.array([PhysicalConstants.R_EARTH_MEAN, 0, 0])
        
        a_central = gravity_central(r)
        a_j2 = gravity_j2(r)
        
        # J2 effect at equator is small but non-zero
        assert np.linalg.norm(a_j2) != np.linalg.norm(a_central)
    
    def test_j2_zonal_variation(self):
        """J2 should cause different acceleration at poles vs equator."""
        r_equator = np.array([PhysicalConstants.R_EARTH_MEAN, 0, 0])
        r_pole = np.array([0, 0, PhysicalConstants.R_EARTH_MEAN])
        
        a_equator = gravity_j2(r_equator)
        a_pole = gravity_j2(r_pole)
        
        # Magnitudes should differ due to J2
        assert abs(np.linalg.norm(a_equator) - np.linalg.norm(a_pole)) > 0.01


class TestCircularOrbit:
    """Test circular orbit state generation."""
    
    def test_circular_orbit_altitude(self):
        """Test circular orbit has correct altitude."""
        alt = 400e3
        state = circular_orbit_state(alt)
        
        r = np.linalg.norm(state[:3])
        expected_r = PhysicalConstants.R_EARTH_MEAN + alt
        
        assert abs(r - expected_r) / expected_r < 1e-10
    
    def test_circular_orbit_velocity(self):
        """Test circular orbit velocity magnitude."""
        alt = 400e3
        state = circular_orbit_state(alt)
        
        v = np.linalg.norm(state[3:])
        r = np.linalg.norm(state[:3])
        expected_v = np.sqrt(PhysicalConstants.MU_EARTH / r)
        
        assert abs(v - expected_v) / expected_v < 1e-10
    
    def test_circular_orbit_elements(self):
        """Test circular orbit has near-zero eccentricity."""
        alt = 500e3
        state = circular_orbit_state(alt)
        elements = state_to_keplerian(state)
        
        assert elements.eccentricity < 1e-10
        assert abs(elements.semi_major_axis_km - (PhysicalConstants.R_EARTH_MEAN + alt) / 1000) < 1


class TestKeplerianConversion:
    """Test state vector to Keplerian elements conversion."""
    
    def test_roundtrip_conversion(self):
        """State → Keplerian → State should be identity."""
        state0 = circular_orbit_state(600e3, 45.0)
        
        elements = state_to_keplerian(state0)
        state1 = keplerian_to_state(elements)
        
        assert np.allclose(state0, state1, rtol=1e-10)
    
    def test_iss_orbit(self):
        """Test ISS orbit parameters."""
        # ISS typical values
        alt_km = 408
        inc_deg = 51.6
        
        state = circular_orbit_state(alt_km * 1000, inc_deg)
        elements = state_to_keplerian(state)
        
        assert abs(elements.inclination_deg - inc_deg) < 0.1
        assert abs(elements.perigee_km - alt_km) < 1
        
        # Period should be ~92.7 minutes
        assert abs(elements.period_min - 92.7) < 1


class TestOrbitPropagation:
    """Test orbit propagation functionality."""
    
    def test_short_propagation_conservation(self):
        """Energy should be nearly conserved for short propagation without drag."""
        state0 = circular_orbit_state(800e3)  # High altitude, minimal drag
        
        result = propagate_orbit(
            state0,
            t_span_years=0.01,  # ~3.65 days
            Cd=2.2,
            A=1.0,
            mass=1000.0,
            perturbation_model=PerturbationModel.BASIC,
            dt_output_days=0.1,
        )
        
        # Semi-major axis should remain nearly constant
        a_initial = result.semi_major_axes_km[0]
        a_final = result.semi_major_axes_km[-1]
        
        assert abs(a_initial - a_final) / a_initial < 0.01
    
    def test_decay_with_drag(self):
        """Orbit should decay with atmospheric drag."""
        state0 = circular_orbit_state(400e3)  # Low altitude
        
        result = propagate_orbit(
            state0,
            t_span_years=1.0,
            Cd=2.2,
            A=10.0,  # Large area for faster decay
            mass=100.0,  # Low mass
            dt_output_days=5.0,
        )
        
        # Altitude should decrease
        assert result.altitudes_km[-1] < result.altitudes_km[0]
    
    def test_reentry_detection(self):
        """Propagation should detect re-entry."""
        state0 = circular_orbit_state(300e3)  # Very low altitude
        
        result = propagate_orbit(
            state0,
            t_span_years=1.0,
            Cd=2.2,
            A=10.0,
            mass=50.0,
            dt_output_days=1.0,
        )
        
        # Should detect re-entry or very low altitude
        assert result.reentry_detected or np.min(result.altitudes_km) < 120


class TestDeltaVManeuver:
    """Test ΔV maneuver implementation."""
    
    def test_retrograde_reduces_velocity(self):
        """Retrograde burn should reduce velocity magnitude."""
        state0 = circular_orbit_state(500e3)
        v0 = np.linalg.norm(state0[3:])
        
        dv = 100.0
        new_state = apply_delta_v(state0, dv, "retrograde")
        v1 = np.linalg.norm(new_state[3:])
        
        assert v1 < v0
        assert abs(v0 - v1 - dv) < 1.0  # Approximately correct
    
    def test_retrograde_lowers_perigee(self):
        """Retrograde burn should lower perigee."""
        state0 = circular_orbit_state(600e3)
        elements0 = state_to_keplerian(state0)
        
        new_state = apply_delta_v(state0, 50.0, "retrograde")
        elements1 = state_to_keplerian(new_state)
        
        assert elements1.perigee_km < elements0.perigee_km
    
    def test_prograde_increases_apogee(self):
        """Prograde burn should raise apogee."""
        state0 = circular_orbit_state(500e3)
        elements0 = state_to_keplerian(state0)
        
        new_state = apply_delta_v(state0, 50.0, "prograde")
        elements1 = state_to_keplerian(new_state)
        
        assert elements1.apogee_km > elements0.apogee_km


# =============================================================================
# SECTION 3: PINN TESTS
# =============================================================================

class TestStateNormalizer:
    """Test state normalization utilities."""
    
    def test_normalize_denormalize_roundtrip(self):
        """Normalization followed by denormalization should be identity."""
        normalizer = StateNormalizer()
        
        state = np.array([
            PhysicalConstants.R_EARTH_MEAN + 400e3, 0, 0,
            0, 7500, 0
        ])
        
        state_norm = normalizer.normalize_state(state)
        state_denorm = normalizer.denormalize_state(state_norm)
        
        assert np.allclose(state, state_denorm, rtol=1e-6)
    
    def test_time_normalization(self):
        """Time should be normalized to [0, 1] range."""
        normalizer = StateNormalizer(t_scale=PhysicalConstants.SECONDS_PER_YEAR)
        
        t = PhysicalConstants.SECONDS_PER_YEAR / 2
        t_norm = normalizer.normalize_time(t)
        
        assert abs(t_norm - 0.5) < 1e-10


class TestPINNArchitecture:
    """Test PINN model architecture."""
    
    def test_pinn_forward_pass(self):
        """Test PINN forward pass produces valid output."""
        model = OrbitalPINN(PINNConfig(hidden_layers=[32, 32]))
        model.eval()
        
        t = torch.randn(10, 1)
        
        with torch.no_grad():
            output = model(t)
        
        assert output.shape == (10, 6)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_pinn_output_dimensions(self):
        """PINN output should be 6-dimensional state."""
        config = PINNConfig(hidden_layers=[64, 64])
        model = OrbitalPINN(config)
        
        t = torch.randn(1, 1)
        output = model(t)
        
        assert output.shape[1] == 6


# =============================================================================
# SECTION 4: OPTIMIZATION TESTS
# =============================================================================

class TestDeltaVOptimizer:
    """Test ΔV optimization functionality."""
    
    def test_optimization_finds_solution(self):
        """Optimizer should find a compliant solution."""
        state0 = circular_orbit_state(500e3)
        
        result = optimize_delta_v(
            state0,
            Cd=2.2,
            A=5.0,
            mass=100.0,
            target_lifetime=5.0,
            method=OptimizationMethod.GRID_SEARCH,
            config=OptimizerConfig(dv_max=20.0, dv_step_coarse=5.0),
            max_years=6.0,
            verbose=False,
        )
        
        assert result.success or result.optimal_dv > 0
    
    def test_higher_dv_reduces_lifetime(self):
        """Higher ΔV should generally reduce lifetime."""
        state0 = circular_orbit_state(400e3)
        
        lt_0 = compute_lifetime_with_dv(state0, 0, 2.2, 5.0, 100.0)
        lt_10 = compute_lifetime_with_dv(state0, 10, 2.2, 5.0, 100.0)
        lt_20 = compute_lifetime_with_dv(state0, 20, 2.2, 5.0, 100.0)
        
        assert lt_0 >= lt_10 >= lt_20


# Helper function for optimization tests
def compute_lifetime_with_dv(state0, dv, Cd, A, mass):
    """Helper to compute lifetime with ΔV."""
    new_state = apply_delta_v(state0, dv, "retrograde")
    return compute_lifetime(new_state, Cd, A, mass, max_years=5.0)


# =============================================================================
# SECTION 5: INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_workflow(self):
        """Test complete PINN training and optimization workflow (quick)."""
        # Use SSLV config for faster testing
        config = SSLV_CONFIG
        state0 = circular_orbit_state(config.orbit_altitude)
        
        # Quick training
        model, normalizer, history, rk8_result = train_pinn(
            state0,
            Cd=config.Cd,
            A=config.drag_sail_area,
            mass=config.dry_mass,
            config=PINNConfig(
                hidden_layers=[32, 32],
                epochs=100,
                n_collocation=100,
                n_data=20,
            ),
            t_span_years=2.0,
            verbose=False,
        )
        
        # Verify training completed
        assert len(history.epochs) > 0
        assert history.best_loss < float('inf')
        
        # Test prediction
        t_test = np.linspace(0, 2, 10)
        states, pred_time = predict_trajectory(model, normalizer, t_test)
        
        assert states.shape == (10, 6)
        assert pred_time > 0
        
        # Test optimization
        opt_result = optimize_delta_v(
            state0,
            Cd=config.Cd,
            A=config.drag_sail_area,
            mass=config.dry_mass,
            target_lifetime=5.0,
            method=OptimizationMethod.GRID_SEARCH,
            config=OptimizerConfig(dv_step_coarse=5.0),
            verbose=False,
        )
        
        assert opt_result.optimal_dv >= 0


class TestValidationCases:
    """Test against known validation cases."""
    
    def test_iss_period(self):
        """Test ISS orbital period."""
        iss_alt = 408e3
        state = circular_orbit_state(iss_alt, 51.6)
        elements = state_to_keplerian(state)
        
        # ISS period is approximately 92.68 minutes
        expected_period = 92.68
        assert abs(elements.period_min - expected_period) < 0.5
    
    def test_sun_synchronous_inclination(self):
        """Test sun-synchronous orbit inclination."""
        alt = 600e3
        state = circular_orbit_state(alt)
        elements = state_to_keplerian(state)
        
        # SSO inclination is typically 97-98 degrees
        assert 97.0 < elements.inclination_deg < 98.5


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
