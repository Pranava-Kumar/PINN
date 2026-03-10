"""
SMART-DEORBIT SYSTEM — ΔV Optimizer
=====================================
Uses the trained PINN as a fast surrogate to find the minimum
retrograde ΔV that ensures re-entry within the target lifetime.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from config import (
    SECONDS_PER_YEAR, TARGET_LIFETIME_YEARS, R_EARTH,
    REENTRY_ALTITUDE, circular_orbit_state
)
from orbital_mechanics import apply_delta_v, propagate_orbit, compute_lifetime
from pinn_model import (
    train_pinn, predict_trajectory, predict_lifetime_pinn, StateNormalizer
)


def compute_lifetime_with_dv_rk8(state0, delta_v, Cd, A, mass,
                                  max_years=30.0):
    """
    Compute orbital lifetime after applying a retrograde ΔV, using RK8.
    
    Parameters
    ----------
    state0 : np.ndarray, shape (6,)
        Initial state before burn.
    delta_v : float
        ΔV magnitude [m/s], applied retrograde.
    Cd, A, mass : float
        Drag parameters.
    max_years : float
        Maximum simulation time.
    
    Returns
    -------
    lifetime : float
        Orbital lifetime [years].
    """
    new_state = apply_delta_v(state0, delta_v)
    return compute_lifetime(new_state, Cd, A, mass, max_years)


def compute_lifetime_with_dv_pinn(model, normalizer, state0, delta_v,
                                   Cd, A, mass, t_span_years=10.0):
    """
    Compute orbital lifetime after applying a retrograde ΔV, using PINN.
    
    For each candidate ΔV, we retrain a lightweight PINN or use the
    existing one with adjusted initial conditions. For the demo, we use
    the RK8 propagator for different ΔV values (the PINN is shown for
    the trajectory visualization).
    
    In a full system, the PINN would be the fast surrogate.
    """
    new_state = apply_delta_v(state0, delta_v)
    result = propagate_orbit(
        new_state, t_span_years, Cd, A, mass,
        dt_output_days=5.0, max_step_days=1.0
    )
    return result['lifetime_years']


def optimize_delta_v(state0, Cd, A, mass,
                     target_lifetime=TARGET_LIFETIME_YEARS,
                     dv_range=(0, 50), dv_step=0.5,
                     max_sim_years=30.0,
                     callback=None):
    """
    Find the minimum retrograde ΔV for compliance with the target lifetime.
    
    Uses a grid search followed by refinement.
    
    Parameters
    ----------
    state0 : np.ndarray, shape (6,)
        Initial state.
    Cd, A, mass : float
        Drag parameters.
    target_lifetime : float
        Maximum allowed lifetime [years].
    dv_range : tuple
        (min_dv, max_dv) search range [m/s].
    dv_step : float
        Grid step size [m/s].
    max_sim_years : float
        Max simulation window.
    callback : callable or None
        Called with (dv, lifetime) for each evaluation.
    
    Returns
    -------
    result : dict
        - 'optimal_dv': float — minimum ΔV [m/s]
        - 'lifetime_at_optimal': float — predicted lifetime [years]
        - 'dv_values': array — all tested ΔV values
        - 'lifetimes': array — corresponding lifetimes
        - 'no_burn_lifetime': float — lifetime with ΔV=0
        - 'propulsive_only_dv': float — ΔV needed without sail
    """
    dv_values = np.arange(dv_range[0], dv_range[1] + dv_step, dv_step)
    lifetimes = []
    
    print(f"Scanning ΔV from {dv_range[0]} to {dv_range[1]} m/s "
          f"(step {dv_step} m/s)...")
    
    for dv in dv_values:
        lt = compute_lifetime_with_dv_rk8(state0, dv, Cd, A, mass, max_sim_years)
        lifetimes.append(lt)
        
        if callback:
            callback(dv, lt)
        
        if dv % 5 == 0:
            print(f"  ΔV = {dv:5.1f} m/s → Lifetime = {lt:.2f} years")
    
    lifetimes = np.array(lifetimes)
    
    # Find minimum ΔV for compliance
    compliant = np.where(lifetimes <= target_lifetime)[0]
    
    if len(compliant) == 0:
        print(f"  WARNING: No ΔV in range achieves {target_lifetime}-year target!")
        optimal_dv = dv_range[1]
        optimal_lt = lifetimes[-1]
    else:
        # First compliant point
        idx = compliant[0]
        
        # Refine between previous and current
        if idx > 0:
            dv_lo = dv_values[idx - 1]
            dv_hi = dv_values[idx]
            
            # Binary search refinement
            for _ in range(10):
                dv_mid = (dv_lo + dv_hi) / 2
                lt_mid = compute_lifetime_with_dv_rk8(
                    state0, dv_mid, Cd, A, mass, max_sim_years
                )
                if lt_mid <= target_lifetime:
                    dv_hi = dv_mid
                else:
                    dv_lo = dv_mid
            
            optimal_dv = dv_hi
            optimal_lt = compute_lifetime_with_dv_rk8(
                state0, optimal_dv, Cd, A, mass, max_sim_years
            )
        else:
            optimal_dv = dv_values[idx]
            optimal_lt = lifetimes[idx]
    
    # No-burn lifetime (ΔV=0 with sail)
    no_burn_lt = lifetimes[0] if len(lifetimes) > 0 else max_sim_years
    
    print(f"\n  ✓ Optimal ΔV: {optimal_dv:.2f} m/s")
    print(f"  ✓ Predicted lifetime: {optimal_lt:.2f} years")
    print(f"  ✗ No-burn lifetime (sail only): {no_burn_lt:.2f} years")
    
    return {
        'optimal_dv': optimal_dv,
        'lifetime_at_optimal': optimal_lt,
        'dv_values': dv_values,
        'lifetimes': lifetimes,
        'no_burn_lifetime': no_burn_lt,
        'target_lifetime': target_lifetime,
    }


def compute_propulsive_only_dv(state0, mass, max_dv=200,
                                cross_section=2.0, Cd=2.2,
                                target_lifetime=TARGET_LIFETIME_YEARS,
                                max_sim_years=30.0):
    """
    Find the ΔV needed for compliance using propulsion ONLY (no sail).
    Uses the spacecraft's natural cross-section instead of sail.
    """
    print(f"\nComputing propulsive-only ΔV (no sail)...")
    
    dv_step = 2.0
    dv_values = np.arange(0, max_dv + dv_step, dv_step)
    
    for dv in dv_values:
        lt = compute_lifetime_with_dv_rk8(
            state0, dv, Cd, cross_section, mass, max_sim_years
        )
        if lt <= target_lifetime:
            print(f"  Propulsive-only ΔV: {dv:.1f} m/s")
            return dv
    
    print(f"  Propulsive-only ΔV: >{max_dv} m/s (insufficient range)")
    return max_dv


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from config import PSLV_CONFIG, SSLV_CONFIG
    
    print("=" * 60)
    print("SMART-DEORBIT — ΔV Optimization Test")
    print("=" * 60)
    
    # ── PSLV PS4 ──
    print("\n── PSLV PS4 @ 750 km ──")
    cfg = PSLV_CONFIG
    state0 = circular_orbit_state(cfg["orbit_altitude"])
    
    result = optimize_delta_v(
        state0,
        Cd=cfg["Cd"],
        A=cfg["drag_sail_area"],
        mass=cfg["dry_mass"],
        dv_range=(0, 30),
        dv_step=1.0,
    )
    
    prop_dv = compute_propulsive_only_dv(
        state0, cfg["dry_mass"],
        cross_section=cfg["cross_section_no_sail"],
        Cd=cfg["Cd"]
    )
    
    print(f"\n  COMPARISON:")
    print(f"    Hybrid (burn + sail): ΔV = {result['optimal_dv']:.2f} m/s")
    print(f"    Propulsive only:      ΔV = {prop_dv:.1f} m/s")
    print(f"    Fuel savings:         {(1 - result['optimal_dv']/prop_dv)*100:.0f}%")
    
    # ── SSLV VTM ──
    print("\n\n── SSLV VTM @ 500 km ──")
    cfg2 = SSLV_CONFIG
    state0_sslv = circular_orbit_state(cfg2["orbit_altitude"])
    
    result2 = optimize_delta_v(
        state0_sslv,
        Cd=cfg2["Cd"],
        A=cfg2["drag_sail_area"],
        mass=cfg2["dry_mass"],
        dv_range=(0, 10),
        dv_step=0.5,
    )
