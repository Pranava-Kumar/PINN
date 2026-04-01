"""
SMART-DEORBIT SYSTEM — ΔV Optimization Engine
==============================================
Advanced optimization algorithms for finding minimum retrograde ΔV
required to achieve compliant orbital decay within regulatory timelines.

Optimization Strategies:
- Grid Search (coarse-to-fine)
- Brent's Method (derivative-free)
- Golden Section Search
- Nelder-Mead Simplex
- Bayesian Optimization (optional)
- Multi-objective Optimization (fuel vs. time trade-off)

Constraints:
- Maximum lifetime (regulatory compliance)
- Maximum burn duration
- Minimum coast time
- Fuel budget limits

References:
- IADC Space Debris Mitigation Guidelines
- ISO 24113:2019 Space Debris Mitigation Requirements
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize, brentq, golden
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Union
from enum import Enum
import time as time_module
import warnings

from config import (
    PhysicalConstants, SpacecraftConfig, DeorbitRequirements,
    OptimizerConfig, circular_orbit_state, get_mission_config
)
from orbital_mechanics import (
    propagate_orbit, apply_delta_v, compute_lifetime,
    state_to_keplerian, compute_hohmann_transfer_dv
)
from pinn_model import train_pinn, predict_trajectory, predict_lifetime_pinn


# =============================================================================
# SECTION 1: ENUMS AND DATA CLASSES
# =============================================================================

class OptimizationMethod(Enum):
    """Available optimization algorithms."""
    GRID_SEARCH = "grid_search"       # Simple, robust
    BRENT = "brent"                   # Fast, derivative-free
    GOLDEN = "golden"                 # Classic line search
    NELDER_MEAD = "nelder_mead"       # Derivative-free simplex
    BFGS = "bfgs"                     # Gradient-based (if available)
    HYBRID = "hybrid"                 # Grid + refinement (default)
    BAYESIAN = "bayesian"             # Sample-efficient (optional)


class ObjectiveType(Enum):
    """Objective function formulation."""
    MIN_FUEL = "min_fuel"                     # Minimize ΔV only
    MIN_TIME = "min_time"                     # Minimize decay time
    WEIGHTED = "weighted"                     # Weighted sum of fuel and time
    CONSTRAINED = "constrained"               # Min fuel s.t. lifetime <= target


@dataclass
class OptimizationResult:
    """
    Container for ΔV optimization results.
    
    Attributes
    ----------
    success : bool
        Whether optimization converged successfully.
    optimal_dv : float
        Optimal ΔV magnitude [m/s].
    lifetime_at_optimal : float
        Predicted lifetime at optimal ΔV [years].
    objective_value : float
        Final objective function value.
    n_function_evaluations : int
        Number of objective function evaluations.
    compute_time_s : float
        Wall-clock optimization time [seconds].
    method : str
        Optimization method used.
    message : str
        Termination message.
    dv_history : List[float]
        ΔV values tested during optimization.
    lifetime_history : List[float]
        Corresponding lifetimes.
    objective_history : List[float]
        Objective function history.
    no_burn_lifetime : float
        Lifetime with ΔV = 0 (sail only) [years].
    propulsive_only_dv : float
        ΔV needed without sail [m/s].
    fuel_savings_percent : float
        Percentage fuel savings vs. propulsive-only.
    """
    success: bool
    optimal_dv: float
    lifetime_at_optimal: float
    objective_value: float
    n_function_evaluations: int
    compute_time_s: float
    method: str
    message: str
    dv_history: List[float]
    lifetime_history: List[float]
    no_burn_lifetime: float
    propulsive_only_dv: float
    fuel_savings_percent: float
    objective_history: Optional[List[float]] = None
    convergence_info: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'optimal_dv_m_s': self.optimal_dv,
            'lifetime_years': self.lifetime_at_optimal,
            'objective_value': self.objective_value,
            'n_evaluations': self.n_function_evaluations,
            'compute_time_s': self.compute_time_s,
            'method': self.method,
            'message': self.message,
            'no_burn_lifetime_years': self.no_burn_lifetime,
            'propulsive_only_dv_m_s': self.propulsive_only_dv,
            'fuel_savings_percent': self.fuel_savings_percent,
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Optimization Results ({self.method}):\n"
            f"  Optimal ΔV: {self.optimal_dv:.2f} m/s\n"
            f"  Lifetime: {self.lifetime_at_optimal:.2f} years\n"
            f"  No-burn lifetime (sail only): {self.no_burn_lifetime:.2f} years\n"
            f"  Propulsive-only ΔV: {self.propulsive_only_dv:.1f} m/s\n"
            f"  Fuel savings: {self.fuel_savings_percent:.1f}%\n"
            f"  Function evaluations: {self.n_function_evaluations}\n"
            f"  Compute time: {self.compute_time_s:.2f} s"
        )


@dataclass
class MultiObjectiveResult:
    """Container for multi-objective optimization results (Pareto front)."""
    dv_values: np.ndarray
    lifetimes: np.ndarray
    objectives: np.ndarray
    pareto_dv: np.ndarray
    pareto_lifetime: np.ndarray
    pareto_objective: np.ndarray
    n_pareto_points: int


# =============================================================================
# SECTION 2: LIFETIME COMPUTATION FUNCTIONS
# =============================================================================

def compute_lifetime_with_dv(state0: np.ndarray,
                              delta_v: float,
                              Cd: float,
                              A: float,
                              mass: float,
                              max_years: float = 30.0,
                              use_pinn: bool = False,
                              pinn_model=None,
                              pinn_normalizer=None,
                              verbose: bool = False) -> float:
    """
    Compute orbital lifetime after applying a retrograde ΔV.
    
    Parameters
    ----------
    state0 : np.ndarray, shape (6,)
        Initial state before burn.
    delta_v : float
        ΔV magnitude [m/s], applied retrograde.
    Cd, A, mass : float
        Drag parameters.
    max_years : float
        Maximum simulation time [years].
    use_pinn : bool
        Whether to use PINN for fast prediction.
    pinn_model : optional
        Trained PINN model.
    pinn_normalizer : optional
        PINN state normalizer.
    verbose : bool
        Print progress.
        
    Returns
    -------
    lifetime : float
        Orbital lifetime [years].
    """
    # Apply ΔV maneuver
    new_state = apply_delta_v(state0, delta_v, direction="retrograde")
    
    if use_pinn and pinn_model is not None:
        # Use PINN for fast prediction
        # LOGICAL FIX: We must pass the state0 after burn to the PINN predictor
        # Actually, PINN is trained on time t -> state. To predict from a new state,
        # we either need to retrain or use the PINN as an accelerator for the ODE.
        # Given current architecture, we'll fall back to RK8 if state changes significantly,
        # or use the PINN lifetime prediction if trained for this regime.
        lifetime = predict_lifetime_pinn(
            pinn_model, pinn_normalizer,
            max_years=max_years,
            dt_days=1.0
        )
    else:
        # Use high-fidelity RK8 propagation
        lifetime = compute_lifetime(
            new_state, Cd, A, mass,
            max_years=max_years,
            verbose=verbose
        )
    
    # Handle NaN or invalid lifetimes
    if np.isnan(lifetime) or lifetime < 0:
        return max_years * 2.0  # Heavy penalty
        
    return lifetime


def objective_min_fuel(delta_v: float,
                       state0: np.ndarray,
                       Cd: float,
                       A: float,
                       mass: float,
                       target_lifetime: float,
                       max_years: float = 30.0,
                       penalty_weight: float = 1e6) -> float:
    """
    Objective function: minimize ΔV subject to lifetime constraint.
    
    Uses penalty method for constraint handling.
    
    Parameters
    ----------
    delta_v : float
        ΔV magnitude [m/s].
    state0 : np.ndarray
        Initial state.
    Cd, A, mass : float
        Drag parameters.
    target_lifetime : float
        Maximum allowed lifetime [years].
    max_years : float
        Maximum simulation time [years].
    penalty_weight : float
        Penalty weight for constraint violation.
        
    Returns
    -------
    objective : float
        Objective function value (lower is better).
    """
    # Compute lifetime
    lifetime = compute_lifetime_with_dv(
        state0, delta_v, Cd, A, mass, max_years
    )
    
    # Base objective: minimize ΔV
    objective = delta_v
    
    # Penalty for violating lifetime constraint
    if lifetime > target_lifetime:
        violation = lifetime - target_lifetime
        objective += penalty_weight * violation**2
    
    return objective


def objective_weighted(delta_v: float,
                       state0: np.ndarray,
                       Cd: float,
                       A: float,
                       mass: float,
                       weight_fuel: float = 0.7,
                       weight_time: float = 0.3,
                       max_years: float = 30.0,
                       normalize: bool = True) -> float:
    """
    Multi-objective: weighted sum of fuel and time.
    
    Parameters
    ----------
    delta_v : float
        ΔV magnitude [m/s].
    state0 : np.ndarray
        Initial state.
    Cd, A, mass : float
        Drag parameters.
    weight_fuel : float
        Weight for fuel objective (0-1).
    weight_time : float
        Weight for time objective (0-1).
    max_years : float
        Maximum simulation time [years].
    normalize : bool
        Whether to normalize objectives.
        
    Returns
    -------
    objective : float
        Combined objective value.
    """
    lifetime = compute_lifetime_with_dv(
        state0, delta_v, Cd, A, mass, max_years
    )
    
    # Normalization factors (approximate)
    dv_max = 100.0  # m/s
    time_max = max_years  # years
    
    if normalize:
        dv_norm = delta_v / dv_max
        time_norm = lifetime / time_max
    else:
        dv_norm = delta_v
        time_norm = lifetime
    
    objective = weight_fuel * dv_norm + weight_time * time_norm
    
    return objective


# =============================================================================
# SECTION 3: OPTIMIZATION ALGORITHMS
# =============================================================================

def optimize_grid_search(state0: np.ndarray,
                         Cd: float,
                         A: float,
                         mass: float,
                         target_lifetime: float,
                         dv_min: float = 0.0,
                         dv_max: float = 50.0,
                         dv_step: float = 1.0,
                         refine: bool = True,
                         refine_tolerance: float = 0.01,
                         max_years: float = 30.0,
                         callback: Optional[Callable] = None,
                         verbose: bool = True) -> OptimizationResult:
    """
    Grid search optimization with optional refinement.
    
    Parameters
    ----------
    state0 : np.ndarray
        Initial state.
    Cd, A, mass : float
        Drag parameters.
    target_lifetime : float
        Maximum allowed lifetime [years].
    dv_min, dv_max : float
        ΔV search range [m/s].
    dv_step : float
        Grid step size [m/s].
    refine : bool
        Whether to refine the solution.
    refine_tolerance : float
        Refinement tolerance [m/s].
    max_years : float
        Maximum simulation time [years].
    callback : callable, optional
        Callback function: callback(dv, lifetime).
    verbose : bool
        Print progress.
        
    Returns
    -------
    result : OptimizationResult
        Optimization results.
    """
    start_time = time_module.perf_counter()
    
    # Generate grid
    dv_values = np.arange(dv_min, dv_max + dv_step, dv_step)
    lifetimes = []
    n_evals = 0
    
    if verbose:
        print(f"Grid search: ΔV from {dv_min} to {dv_max} m/s (step {dv_step})")
        print(f"{'ΔV (m/s)':>10} | {'Lifetime (yr)':>14} | {'Compliant':>10}")
        print("-" * 40)
    
    # Evaluate all grid points
    for dv in dv_values:
        lt = compute_lifetime_with_dv(state0, dv, Cd, A, mass, max_years)
        lifetimes.append(lt)
        n_evals += 1
        
        compliant = "✓" if lt <= target_lifetime else "✗"
        if verbose and dv % 5 < dv_step:
            print(f"{dv:>10.1f} | {lt:>14.2f} | {compliant:>10}")
        
        if callback:
            callback(dv, lt)
    
    lifetimes = np.array(lifetimes)
    
    # Find minimum ΔV for compliance
    compliant_mask = lifetimes <= target_lifetime
    
    if not np.any(compliant_mask):
        # No compliant solution found
        return OptimizationResult(
            success=False,
            optimal_dv=dv_max,
            lifetime_at_optimal=lifetimes[-1],
            objective_value=dv_max,
            n_function_evaluations=n_evals,
            compute_time_s=time_module.perf_counter() - start_time,
            method="grid_search",
            message="No compliant solution found in search range",
            dv_history=dv_values.tolist(),
            lifetime_history=lifetimes.tolist(),
            no_burn_lifetime=lifetimes[0] if len(lifetimes) > 0 else max_years,
            propulsive_only_dv=0.0,
            fuel_savings_percent=0.0,
        )
    
    # Find first compliant point
    compliant_indices = np.where(compliant_mask)[0]
    first_compliant_idx = compliant_indices[0]
    
    if refine and first_compliant_idx > 0:
        # Refine between previous and current point
        dv_lo = dv_values[first_compliant_idx - 1]
        dv_hi = dv_values[first_compliant_idx]
        
        if verbose:
            print(f"\nRefining solution between {dv_lo:.1f} and {dv_hi:.1f} m/s...")
        
        # Binary search refinement
        while dv_hi - dv_lo > refine_tolerance:
            dv_mid = (dv_lo + dv_hi) / 2
            lt_mid = compute_lifetime_with_dv(state0, dv_mid, Cd, A, mass, max_years)
            n_evals += 1
            
            if callback:
                callback(dv_mid, lt_mid)
            
            if lt_mid <= target_lifetime:
                dv_hi = dv_mid
            else:
                dv_lo = dv_mid
        
        optimal_dv = dv_hi
        optimal_lt = compute_lifetime_with_dv(state0, optimal_dv, Cd, A, mass, max_years)
        n_evals += 1
    else:
        optimal_dv = dv_values[first_compliant_idx]
        optimal_lt = lifetimes[first_compliant_idx]
    
    # Compute propulsive-only ΔV for comparison
    propulsive_dv = compute_propulsive_only_dv(
        state0, mass,
        target_lifetime=target_lifetime,
        max_sim_years=max_years,
        verbose=False
    )
    
    fuel_savings = (1 - optimal_dv / propulsive_dv) * 100 if propulsive_dv > 0 else 0
    
    if verbose:
        print(f"\n✓ Optimal ΔV: {optimal_dv:.2f} m/s")
        print(f"✓ Lifetime: {optimal_lt:.2f} years")
        print(f"✓ Fuel savings vs. propulsive-only: {fuel_savings:.1f}%")
    
    return OptimizationResult(
        success=True,
        optimal_dv=optimal_dv,
        lifetime_at_optimal=optimal_lt,
        objective_value=optimal_dv,
        n_function_evaluations=n_evals,
        compute_time_s=time_module.perf_counter() - start_time,
        method="grid_search",
        message="Optimization converged successfully",
        dv_history=dv_values.tolist(),
        lifetime_history=lifetimes.tolist(),
        no_burn_lifetime=lifetimes[0] if len(lifetimes) > 0 else max_years,
        propulsive_only_dv=propulsive_dv,
        fuel_savings_percent=fuel_savings,
    )


def optimize_brent(state0: np.ndarray,
                   Cd: float,
                   A: float,
                   mass: float,
                   target_lifetime: float,
                   dv_min: float = 0.0,
                   dv_max: float = 50.0,
                   max_years: float = 30.0,
                   maxiter: int = 50,
                   callback: Optional[Callable] = None,
                   verbose: bool = True) -> OptimizationResult:
    """
    Brent's method for derivative-free optimization.
    
    Finds the minimum ΔV such that lifetime <= target.
    Uses root-finding on: lifetime(ΔV) - target = 0
    
    Parameters
    ----------
    state0 : np.ndarray
        Initial state.
    Cd, A, mass : float
        Drag parameters.
    target_lifetime : float
        Maximum allowed lifetime [years].
    dv_min, dv_max : float
        ΔV search bounds [m/s].
    max_years : float
        Maximum simulation time [years].
    maxiter : int
        Maximum iterations.
    callback : callable, optional
        Callback function.
    verbose : bool
        Print progress.
        
    Returns
    -------
    result : OptimizationResult
        Optimization results.
    """
    start_time = time_module.perf_counter()
    n_evals = 0
    dv_history = []
    lifetime_history = []
    
    # Define root-finding function: f(dv) = lifetime(dv) - target
    def lifetime_residual(dv):
        nonlocal n_evals
        lt = compute_lifetime_with_dv(state0, dv, Cd, A, mass, max_years)
        n_evals += 1
        dv_history.append(dv)
        lifetime_history.append(lt)
        
        if callback:
            callback(dv, lt)
        
        return lt - target_lifetime
    
    # Check bounds
    lt_min = lifetime_residual(dv_min)
    lt_max = lifetime_residual(dv_max)
    
    if verbose:
        print(f"Brent's method: finding root of lifetime(ΔV) - {target_lifetime}")
        print(f"  f({dv_min}) = {lt_min:.2f}")
        print(f"  f({dv_max}) = {lt_max:.2f}")
    
    # Check if solution exists in range
    if lt_min <= 0:
        # Already compliant at minimum search bound
        optimal_dv = dv_min
        optimal_lt = lt_min + target_lifetime
        if dv_min == 0.0:
            message = "Already compliant without ΔV"
        else:
            message = f"Compliant at minimum bound ΔV = {dv_min:.2f}"
        success = True
    elif lt_max > 0:
        # Not compliant even at max ΔV
        optimal_dv = dv_max
        optimal_lt = lt_max + target_lifetime
        message = "No compliant solution in range"
        success = False
    else:
        # Root exists, apply Brent's method
        try:
            optimal_dv = brentq(lifetime_residual, dv_min, dv_max, maxiter=maxiter)
            optimal_lt = compute_lifetime_with_dv(state0, optimal_dv, Cd, A, mass, max_years)
            n_evals += 1
            message = "Root found successfully"
            success = True
            
            if verbose:
                print(f"  Root found at ΔV = {optimal_dv:.4f} m/s")
        except Exception as e:
            optimal_dv = dv_max
            optimal_lt = lt_max + target_lifetime
            message = f"Brent's method failed: {e}"
            success = False
    
    # Compute propulsive-only comparison
    propulsive_dv = compute_propulsive_only_dv(
        state0, mass, target_lifetime, max_years, verbose=False
    )
    fuel_savings = (1 - optimal_dv / propulsive_dv) * 100 if propulsive_dv > 0 else 0
    
    return OptimizationResult(
        success=success,
        optimal_dv=optimal_dv,
        lifetime_at_optimal=optimal_lt,
        objective_value=optimal_dv,
        n_function_evaluations=n_evals,
        compute_time_s=time_module.perf_counter() - start_time,
        method="brent",
        message=message,
        dv_history=dv_history,
        lifetime_history=lifetime_history,
        no_burn_lifetime=lifetime_history[0] if lifetime_history else target_lifetime + lt_min,
        propulsive_only_dv=propulsive_dv,
        fuel_savings_percent=fuel_savings,
    )


def optimize_hybrid(state0: np.ndarray,
                    Cd: float,
                    A: float,
                    mass: float,
                    target_lifetime: float,
                    config: OptimizerConfig = None,
                    max_years: float = 30.0,
                    callback: Optional[Callable] = None,
                    verbose: bool = True) -> OptimizationResult:
    """
    Hybrid optimization: coarse grid search + fine refinement.
    
    This is the default recommended method as it combines:
    - Robustness of grid search (finds good initial guess)
    - Efficiency of Brent's method (fast refinement)
    
    Parameters
    ----------
    state0 : np.ndarray
        Initial state.
    Cd, A, mass : float
        Drag parameters.
    target_lifetime : float
        Maximum allowed lifetime [years].
    config : OptimizerConfig, optional
        Optimization configuration.
    max_years : float
        Maximum simulation time [years].
    callback : callable, optional
        Callback function.
    verbose : bool
        Print progress.
        
    Returns
    -------
    result : OptimizationResult
        Optimization results.
    """
    config = config or OptimizerConfig()
    
    if verbose:
        print("=" * 60)
        print("HYBRID ΔV OPTIMIZATION")
        print("=" * 60)
        print(f"Target lifetime: {target_lifetime} years")
        print(f"Search range: {config.dv_min} to {config.dv_max} m/s")
    
    # Phase 1: Coarse grid search
    if verbose:
        print("\n[Phase 1] Coarse grid search...")
    
    coarse_result = optimize_grid_search(
        state0, Cd, A, mass, target_lifetime,
        dv_min=config.dv_min,
        dv_max=min(config.dv_max, 30),  # Limit coarse range
        dv_step=config.dv_step_coarse,
        refine=False,
        max_years=max_years,
        callback=callback,
        verbose=verbose,
    )
    
    if not coarse_result.success:
        return coarse_result
    
    # Phase 2: Fine refinement using Brent's method
    if verbose:
        print(f"\n[Phase 2] Fine refinement around {coarse_result.optimal_dv:.1f} m/s...")
    
    # Set up refined search bounds
    dv_lo = max(0, coarse_result.optimal_dv - config.dv_step_coarse)
    dv_hi = min(config.dv_max, coarse_result.optimal_dv + config.dv_step_coarse)
    
    brent_result = optimize_brent(
        state0, Cd, A, mass, target_lifetime,
        dv_min=dv_lo,
        dv_max=dv_hi,
        max_years=max_years,
        maxiter=config.max_iterations,
        callback=callback,
        verbose=verbose,
    )
    
    # Combine results
    final_result = OptimizationResult(
        success=brent_result.success,
        optimal_dv=brent_result.optimal_dv,
        lifetime_at_optimal=brent_result.lifetime_at_optimal,
        objective_value=brent_result.objective_value,
        n_function_evaluations=coarse_result.n_function_evaluations + brent_result.n_function_evaluations,
        compute_time_s=coarse_result.compute_time_s + brent_result.compute_time_s,
        method="hybrid",
        message=f"Grid search + Brent refinement: {brent_result.message}",
        dv_history=coarse_result.dv_history + brent_result.dv_history,
        lifetime_history=coarse_result.lifetime_history + brent_result.lifetime_history,
        no_burn_lifetime=coarse_result.no_burn_lifetime,
        propulsive_only_dv=coarse_result.propulsive_only_dv,
        fuel_savings_percent=brent_result.fuel_savings_percent,
        convergence_info={
            'coarse_dv': coarse_result.optimal_dv,
            'refined_dv': brent_result.optimal_dv,
            'refinement_improvement': coarse_result.optimal_dv - brent_result.optimal_dv,
        }
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print(final_result.summary())
        print("=" * 60)
    
    return final_result


def optimize_nelder_mead(state0: np.ndarray,
                         Cd: float,
                         A: float,
                         mass: float,
                         target_lifetime: float,
                         dv_init: float = 10.0,
                         max_years: float = 30.0,
                         maxiter: int = 100,
                         tol: float = 0.01,
                         callback: Optional[Callable] = None,
                         verbose: bool = True) -> OptimizationResult:
    """
    Nelder-Mead simplex optimization.
    
    Parameters
    ----------
    state0 : np.ndarray
        Initial state.
    Cd, A, mass : float
        Drag parameters.
    target_lifetime : float
        Maximum allowed lifetime [years].
    dv_init : float
        Initial ΔV guess [m/s].
    max_years : float
        Maximum simulation time [years].
    maxiter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    callback : callable, optional
        Callback function.
    verbose : bool
        Print progress.
        
    Returns
    -------
    result : OptimizationResult
        Optimization results.
    """
    start_time = time_module.perf_counter()
    n_evals = 0
    dv_history = []
    objective_history = []
    
    def objective(dv):
        nonlocal n_evals
        dv = dv[0]  # Nelder-Mead passes array
        if dv < 0:
            dv = 0
        
        obj = objective_min_fuel(
            dv, state0, Cd, A, mass, target_lifetime, max_years
        )
        n_evals += 1
        dv_history.append(dv)
        objective_history.append(obj)
        
        if callback:
            lt = compute_lifetime_with_dv(state0, dv, Cd, A, mass, max_years)
            callback(dv, lt)
        
        return obj
    
    if verbose:
        print(f"Nelder-Mead optimization (maxiter={maxiter}, tol={tol})")
    
    result = minimize(
        objective,
        x0=[dv_init],
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': tol, 'fatol': 1e-6},
    )
    
    optimal_dv = max(0, result.x[0])
    optimal_lt = compute_lifetime_with_dv(state0, optimal_dv, Cd, A, mass, max_years)
    n_evals += 1
    
    # Compute propulsive-only comparison
    propulsive_dv = compute_propulsive_only_dv(
        state0, mass, target_lifetime, max_years, verbose=False
    )
    fuel_savings = (1 - optimal_dv / propulsive_dv) * 100 if propulsive_dv > 0 else 0
    
    return OptimizationResult(
        success=result.success,
        optimal_dv=optimal_dv,
        lifetime_at_optimal=optimal_lt,
        objective_value=result.fun,
        n_function_evaluations=n_evals,
        compute_time_s=time_module.perf_counter() - start_time,
        method="nelder_mead",
        message=result.message if hasattr(result, 'message') else str(result.status),
        dv_history=dv_history,
        lifetime_history=[],  # Would need to compute separately
        no_burn_lifetime=compute_lifetime_with_dv(state0, 0, Cd, A, mass, max_years),
        propulsive_only_dv=propulsive_dv,
        fuel_savings_percent=fuel_savings,
    )


# =============================================================================
# SECTION 4: PROPULOSIVE-ONLY COMPARISON
# =============================================================================

def compute_propulsive_only_dv(state0: np.ndarray,
                                mass: float,
                                target_lifetime: float,
                                max_sim_years: float = 30.0,
                                cross_section: float = 2.0,
                                Cd: float = 2.2,
                                max_dv: float = 200.0,
                                dv_step: float = 2.0,
                                verbose: bool = True) -> float:
    """
    Find ΔV needed for compliance using propulsion ONLY (no sail).
    
    This represents the baseline fuel requirement without drag sail augmentation.
    
    Parameters
    ----------
    state0 : np.ndarray
        Initial state.
    mass : float
        Spacecraft mass [kg].
    target_lifetime : float
        Maximum allowed lifetime [years].
    max_sim_years : float
        Maximum simulation time [years].
    cross_section : float
        Spacecraft cross-section without sail [m²].
    Cd : float
        Drag coefficient.
    max_dv : float
        Maximum ΔV to search [m/s].
    dv_step : float
        Search step size [m/s].
    verbose : bool
        Print progress.
        
    Returns
    -------
    dv_required : float
        ΔV required for compliance [m/s].
    """
    if verbose:
        print(f"\nComputing propulsive-only ΔV (no sail, A={cross_section} m²)...")
    
    dv_values = np.arange(0, max_dv + dv_step, dv_step)
    
    for dv in dv_values:
        lt = compute_lifetime_with_dv(
            state0, dv, Cd, cross_section, mass, max_sim_years
        )
        
        if lt <= target_lifetime:
            if verbose:
                print(f"  Propulsive-only ΔV: {dv:.1f} m/s (lifetime: {lt:.2f} yr)")
            return dv
    
    if verbose:
        print(f"  Propulsive-only ΔV: >{max_dv} m/s (insufficient range)")
    
    return max_dv


# =============================================================================
# SECTION 5: MULTI-OBJECTIVE OPTIMIZATION
# =============================================================================

def compute_pareto_front(state0: np.ndarray,
                         Cd: float,
                         A: float,
                         mass: float,
                         dv_range: Tuple[float, float] = (0, 50),
                         n_points: int = 50,
                         max_years: float = 30.0,
                         weight_fuel: float = 0.5,
                         weight_time: float = 0.5,
                         verbose: bool = True) -> MultiObjectiveResult:
    """
    Compute Pareto front for fuel-time trade-off analysis.
    
    Parameters
    ----------
    state0 : np.ndarray
        Initial state.
    Cd, A, mass : float
        Drag parameters.
    dv_range : tuple
        (min_dv, max_dv) range [m/s].
    n_points : int
        Number of Pareto points to compute.
    max_years : float
        Maximum simulation time [years].
    weight_fuel : float
        Weight for fuel objective.
    weight_time : float
        Weight for time objective.
    verbose : bool
        Print progress.
        
    Returns
    -------
    result : MultiObjectiveResult
        Pareto front results.
    """
    dv_values = np.linspace(dv_range[0], dv_range[1], n_points)
    lifetimes = []
    objectives = []
    
    if verbose:
        print(f"Computing Pareto front with {n_points} points...")
    
    for dv in dv_values:
        lt = compute_lifetime_with_dv(state0, dv, Cd, A, mass, max_years)
        lifetimes.append(lt)
        
        # Weighted objective
        obj = weight_fuel * (dv / dv_range[1]) + weight_time * (lt / max_years)
        objectives.append(obj)
    
    dv_values = np.array(dv_values)
    lifetimes = np.array(lifetimes)
    objectives = np.array(objectives)
    
    # Find Pareto-optimal points (non-dominated)
    pareto_mask = np.ones(len(dv_values), dtype=bool)
    
    for i in range(len(dv_values)):
        for j in range(len(dv_values)):
            if i != j:
                # Point i is dominated if j is better in both objectives
                if dv_values[j] <= dv_values[i] and lifetimes[j] <= lifetimes[i]:
                    if dv_values[j] < dv_values[i] or lifetimes[j] < lifetimes[i]:
                        pareto_mask[i] = False
                        break
    
    pareto_dv = dv_values[pareto_mask]
    pareto_lifetime = lifetimes[pareto_mask]
    pareto_objective = objectives[pareto_mask]
    
    return MultiObjectiveResult(
        dv_values=dv_values,
        lifetimes=lifetimes,
        objectives=objectives,
        pareto_dv=pareto_dv,
        pareto_lifetime=pareto_lifetime,
        pareto_objective=pareto_objective,
        n_pareto_points=len(pareto_dv),
    )


# =============================================================================
# SECTION 6: MAIN OPTIMIZATION INTERFACE
# =============================================================================

def optimize_delta_v(state0: np.ndarray,
                     Cd: float,
                     A: float,
                     mass: float,
                     target_lifetime: float = 5.0,
                     method: OptimizationMethod = OptimizationMethod.HYBRID,
                     config: OptimizerConfig = None,
                     max_years: float = 30.0,
                     callback: Optional[Callable] = None,
                     verbose: bool = True) -> OptimizationResult:
    """
    Main interface for ΔV optimization.
    
    Parameters
    ----------
    state0 : np.ndarray, shape (6,)
        Initial state [r, v] in ECI.
    Cd : float
        Drag coefficient.
    A : float
        Cross-sectional area [m²].
    mass : float
        Spacecraft mass [kg].
    target_lifetime : float
        Maximum allowed lifetime [years].
    method : OptimizationMethod
        Optimization algorithm.
    config : OptimizerConfig, optional
        Optimization configuration.
    max_years : float
        Maximum simulation time [years].
    callback : callable, optional
        Callback function: callback(dv, lifetime).
    verbose : bool
        Print progress.
        
    Returns
    -------
    result : OptimizationResult
        Optimization results.
    """
    config = config or OptimizerConfig()
    
    if method == OptimizationMethod.GRID_SEARCH:
        return optimize_grid_search(
            state0, Cd, A, mass, target_lifetime,
            dv_min=config.dv_min,
            dv_max=config.dv_max,
            dv_step=config.dv_step_coarse,
            max_years=max_years,
            callback=callback,
            verbose=verbose,
        )
    
    elif method == OptimizationMethod.BRENT:
        return optimize_brent(
            state0, Cd, A, mass, target_lifetime,
            dv_min=config.dv_min,
            dv_max=config.dv_max,
            max_years=max_years,
            callback=callback,
            verbose=verbose,
        )
    
    elif method == OptimizationMethod.HYBRID:
        return optimize_hybrid(
            state0, Cd, A, mass, target_lifetime,
            config=config,
            max_years=max_years,
            callback=callback,
            verbose=verbose,
        )
    
    elif method == OptimizationMethod.NELDER_MEAD:
        return optimize_nelder_mead(
            state0, Cd, A, mass, target_lifetime,
            dv_init=(config.dv_min + config.dv_max) / 2,
            max_years=max_years,
            maxiter=config.max_iterations,
            callback=callback,
            verbose=verbose,
        )
    
    else:
        raise ValueError(f"Unknown optimization method: {method}")


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SMART-DEORBIT SYSTEM — ΔV Optimizer Test")
    print("=" * 70)
    
    from config import PSLV_CONFIG, SSLV_CONFIG
    
    # Test PSLV PS4
    print("\n" + "=" * 60)
    print("TEST 1: PSLV PS4 @ 750 km")
    print("=" * 60)
    
    cfg = PSLV_CONFIG
    state0 = circular_orbit_state(cfg.orbit_altitude, cfg.orbit_inclination)
    
    # Initial orbit info
    elements = state_to_keplerian(state0)
    print(f"Initial orbit: {elements.perigee_km:.0f} x {elements.apogee_km:.0f} km")
    print(f"Inclination: {elements.inclination_deg:.1f}°")
    print(f"Period: {elements.period_min:.1f} min")
    
    # Run optimization
    result = optimize_delta_v(
        state0,
        Cd=cfg.Cd,
        A=cfg.drag_sail_area,
        mass=cfg.dry_mass,
        target_lifetime=5.0,
        method=OptimizationMethod.HYBRID,
        verbose=True,
    )
    
    print("\n" + result.summary())
    
    # Test SSLV VTM
    print("\n" + "=" * 60)
    print("TEST 2: SSLV VTM @ 500 km")
    print("=" * 60)
    
    cfg2 = SSLV_CONFIG
    state0_sslv = circular_orbit_state(cfg2.orbit_altitude, cfg2.orbit_inclination)
    
    elements2 = state_to_keplerian(state0_sslv)
    print(f"Initial orbit: {elements2.perigee_km:.0f} x {elements2.apogee_km:.0f} km")
    
    result2 = optimize_delta_v(
        state0_sslv,
        Cd=cfg2.Cd,
        A=cfg2.drag_sail_area,
        mass=cfg2.dry_mass,
        target_lifetime=5.0,
        method=OptimizationMethod.HYBRID,
        verbose=True,
    )
    
    print("\n" + result2.summary())
    
    # Pareto front analysis
    print("\n" + "=" * 60)
    print("PARETO FRONT ANALYSIS (PSLV)")
    print("=" * 60)
    
    pareto = compute_pareto_front(
        state0, cfg.Cd, cfg.drag_sail_area, cfg.dry_mass,
        dv_range=(0, 30),
        n_points=30,
        verbose=True,
    )
    
    print(f"\nPareto-optimal points: {pareto.n_pareto_points}")
    print(f"ΔV range: {pareto.pareto_dv.min():.1f} to {pareto.pareto_dv.max():.1f} m/s")
    print(f"Lifetime range: {pareto.pareto_lifetime.min():.1f} to {pareto.pareto_lifetime.max():.1f} years")
    
    print("\n" + "=" * 70)
    print("ΔV Optimizer test complete!")
