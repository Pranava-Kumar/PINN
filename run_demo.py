"""
SMART-DEORBIT SYSTEM — Comprehensive Demo Runner
=================================================
Main demonstration script that runs the complete SMART-DEORBIT analysis
pipeline and generates publication-ready results.

Usage:
    python run_demo.py [--mission PSLV|SSLV] [--output-dir ./outputs]
    
Features:
- Complete PINN training and validation
- ΔV optimization with multiple methods
- Comparative analysis (hybrid vs. propulsive-only)
- Performance benchmarks
- Result visualization and export
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from config import (
    PhysicalConstants, get_mission_config, circular_orbit_state,
    PINNConfig, get_pinn_config, OptimizerConfig,
    PSLV_CONFIG, SSLV_CONFIG
)
from orbital_mechanics import (
    propagate_orbit, state_to_keplerian, apply_delta_v,
    PerturbationModel, IntegratorMethod
)
from pinn_model import train_pinn, predict_trajectory, predict_lifetime_pinn
from delta_v_optimizer import (
    optimize_delta_v, OptimizationMethod, compute_propulsive_only_dv,
    compute_pareto_front
)
from report_generator import ReportGenerator


def setup_output_directory(output_dir: str) -> Path:
    """Create output directory structure."""
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    (base_path / "models").mkdir(exist_ok=True)
    (base_path / "plots").mkdir(exist_ok=True)
    (base_path / "data").mkdir(exist_ok=True)
    (base_path / "reports").mkdir(exist_ok=True)
    
    return base_path


def plot_training_history(history, save_path: str):
    """Plot PINN training history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves (log scale)
    ax = axes[0, 0]
    ax.semilogy(history.epochs, history.loss_total, 'b-', linewidth=2, label='Total Loss')
    ax.semilogy(history.epochs, history.loss_data, 'g--', linewidth=2, label='Data Loss')
    ax.semilogy(history.epochs, history.loss_physics, 'r-.', linewidth=2, label='Physics Loss')
    ax.semilogy(history.epochs, history.loss_initial, 'm:', linewidth=2, label='Initial Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('PINN Training Loss History', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Learning rate schedule
    ax = axes[0, 1]
    ax.plot(history.epochs, history.learning_rate, 'k-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Loss convergence (moving average)
    ax = axes[1, 0]
    window = min(100, len(history.loss_total) // 10)
    if window > 1:
        loss_smooth = np.convolve(history.loss_total, np.ones(window)/window, mode='valid')
        epochs_smooth = history.epochs[window-1:]
        ax.plot(epochs_smooth, loss_smooth, 'b-', linewidth=2)
    else:
        ax.plot(history.epochs, history.loss_total, 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss (smoothed)', fontsize=12)
    ax.set_title('Loss Convergence', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Training progress summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    Training Summary
    ════════════════
    
    Total Epochs: {len(history.epochs)}
    Best Loss: {history.best_loss:.6e}
    Best Epoch: {history.best_epoch}
    Training Time: {history.train_time_s:.1f} s
    
    Final Losses:
      Total:  {history.loss_total[-1]:.6e}
      Data:   {history.loss_data[-1]:.6e}
      Physics: {history.loss_physics[-1]:.6e}
      Initial: {history.loss_initial[-1]:.6e}
    """
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_trajectory_comparison(rk8_result, pinn_states, t_years, save_path: str):
    """Plot RK8 vs PINN trajectory comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Compute derived quantities
    pinn_r = np.linalg.norm(pinn_states[:, :3], axis=1)
    pinn_v = np.linalg.norm(pinn_states[:, 3:], axis=1)
    pinn_alt = (pinn_r - PhysicalConstants.R_EARTH_MEAN) / 1000
    
    rk8_r = np.linalg.norm(rk8_result.states[:, :3], axis=1)
    rk8_v = np.linalg.norm(rk8_result.states[:, 3:], axis=1)
    
    # Altitude comparison
    ax = axes[0, 0]
    ax.plot(t_years, rk8_result.altitudes_km, 'k-', linewidth=2, label='RK8 (Ground Truth)', alpha=0.7)
    ax.plot(t_years, pinn_alt, 'r--', linewidth=2, label='PINN Prediction')
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Altitude (km)', fontsize=12)
    ax.set_title('Altitude Decay Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Velocity comparison
    ax = axes[0, 1]
    ax.plot(t_years, rk8_v / 1000, 'k-', linewidth=2, label='RK8', alpha=0.7)
    ax.plot(t_years, pinn_v / 1000, 'b--', linewidth=2, label='PINN')
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Velocity (km/s)', fontsize=12)
    ax.set_title('Velocity Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Semi-major axis comparison
    ax = axes[0, 2]
    ax.plot(t_years, rk8_result.semi_major_axes_km, 'k-', linewidth=2, label='RK8', alpha=0.7)
    
    # Compute PINN semi-major axis
    pinn_sma = np.zeros_like(pinn_alt)
    for i in range(len(pinn_states)):
        r = pinn_r[i]
        v = pinn_v[i]
        energy = v**2 / 2 - PhysicalConstants.MU_EARTH / r
        if abs(energy) > 1e-10:
            pinn_sma[i] = -PhysicalConstants.MU_EARTH / (2 * energy) / 1000
        else:
            pinn_sma[i] = np.nan
    
    ax.plot(t_years, pinn_sma, 'g--', linewidth=2, label='PINN')
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Semi-major Axis (km)', fontsize=12)
    ax.set_title('Semi-major Axis Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error analysis
    min_len = min(len(rk8_result.altitudes_km), len(pinn_alt))
    alt_error = np.abs(pinn_alt[:min_len] - rk8_result.altitudes_km[:min_len])
    vel_error = np.abs(pinn_v[:min_len] / 1000 - rk8_v[:min_len] / 1000)
    
    # Altitude error
    ax = axes[1, 0]
    ax.plot(t_years[:min_len], alt_error, 'r-', linewidth=2)
    ax.fill_between(t_years[:min_len], 0, alt_error, alpha=0.3)
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Altitude Error (km)', fontsize=12)
    ax.set_title(f'Altitude Prediction Error (Mean: {np.mean(alt_error):.3f} km)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Velocity error
    ax = axes[1, 1]
    ax.plot(t_years[:min_len], vel_error, 'b-', linewidth=2)
    ax.fill_between(t_years[:min_len], 0, vel_error, alpha=0.3)
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Velocity Error (km/s)', fontsize=12)
    ax.set_title(f'Velocity Prediction Error (Mean: {np.mean(vel_error):.4f} km/s)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot error distribution
    ax = axes[1, 2]
    if np.any(np.isfinite(alt_error)):
        ax.hist(alt_error[np.isfinite(alt_error)], bins=30, color='red', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Absolute Error (km)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Prediction Error Distribution', fontsize=14)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No finite error data available", ha='center')
    ax.axvline(np.mean(alt_error), color='black', linestyle='--', linewidth=2, label=f'Mean: {np.mean(alt_error):.3f}')
    ax.axvline(np.std(alt_error), color='blue', linestyle=':', linewidth=2, label=f'±1σ: {np.std(alt_error):.3f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_optimization_results(result, save_path: str):
    """Plot ΔV optimization results."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ΔV vs Lifetime curve
    ax = axes[0]
    ax.plot(result.dv_history, result.lifetime_history, 'bo-', markersize=4, linewidth=1.5, label='Simulated Points')
    ax.axhline(y=5.0, color='green', linestyle='--', linewidth=2, label='5-Year Limit')
    ax.axvline(x=result.optimal_dv, color='red', linestyle=':', linewidth=2, label=f'Optimal ΔV: {result.optimal_dv:.2f} m/s')
    ax.axvline(x=result.propulsive_only_dv, color='orange', linestyle='-.', linewidth=2, label=f'Propulsive-only: {result.propulsive_only_dv:.1f} m/s')
    
    # Highlight optimal point
    ax.plot(result.optimal_dv, result.lifetime_at_optimal, 'r*', markersize=20, label='Optimal Solution')
    
    ax.set_xlabel('Retrograde ΔV (m/s)', fontsize=12)
    ax.set_ylabel('Orbital Lifetime (years)', fontsize=12)
    ax.set_title('ΔV Optimization: Fuel vs. Lifetime Trade-off', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Bar chart comparison
    ax = axes[1]
    categories = ['No Burn\n(Sail Only)', 'Optimal\n(Hybrid)', 'Propulsive\nOnly']
    lifetimes = [result.no_burn_lifetime, result.lifetime_at_optimal, 5.0]  # Propulsive achieves exactly 5 years
    dvs = [0, result.optimal_dv, result.propulsive_only_dv]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dvs, width, label='ΔV (m/s)', color='coral', edgecolor='black')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, lifetimes, width, label='Lifetime (yr)', color='steelblue', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel('ΔV Required (m/s)', fontsize=12, color='coral')
    ax2.set_ylabel('Orbital Lifetime (years)', fontsize=12, color='steelblue')
    ax.set_title('Strategy Comparison: Fuel Requirements', fontsize=14)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, color='white' if height > 10 else 'black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pareto_front(pareto_result, save_path: str):
    """Plot Pareto front analysis."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # All evaluated points
    ax.scatter(pareto_result.dv_values, pareto_result.lifetimes, 
               c=pareto_result.objectives, cmap='viridis', 
               alpha=0.5, s=50, label='All Points', edgecolors='gray')
    
    # Pareto-optimal points
    ax.scatter(pareto_result.pareto_dv, pareto_result.pareto_lifetime,
               c='red', s=100, marker='*', label='Pareto-Optimal', 
               edgecolors='black', linewidths=2)
    
    # Connect Pareto points
    if len(pareto_result.pareto_dv) > 1:
        sort_idx = np.argsort(pareto_result.pareto_dv)
        ax.plot(pareto_result.pareto_dv[sort_idx], pareto_result.pareto_lifetime[sort_idx],
                'r--', linewidth=2, alpha=0.5)
    
    # 5-year limit line
    ax.axhline(y=5.0, color='green', linestyle='--', linewidth=2, label='5-Year Compliance Limit')
    
    ax.set_xlabel('ΔV (m/s)', fontsize=12)
    ax.set_ylabel('Orbital Lifetime (years)', fontsize=12)
    ax.set_title('Multi-Objective Pareto Front: Fuel vs. Time', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Weighted Objective (lower is better)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_report(results: dict, save_path: Path):
    """Generate comprehensive text report."""
    report = f"""
================================================================================
                    SMART-DEORBIT SYSTEM — ANALYSIS REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
MISSION CONFIGURATION
================================================================================

Mission Name: {results['mission_name']}
Initial Altitude: {results['initial_altitude_km']:.1f} km
Initial Inclination: {results['initial_inclination_deg']:.1f}°
Spacecraft Mass: {results['mass_kg']:.1f} kg
Drag Sail Area: {results['sail_area_m2']:.1f} m²
Drag Coefficient: {results['cd']:.2f}

================================================================================
PINN TRAINING RESULTS
================================================================================

Architecture: {results['pinn_architecture']}
Hidden Layers: {results['pinn_hidden_layers']}
Training Epochs: {results['training_epochs']}
Training Time: {results['training_time_s']:.1f} s

Final Losses:
  Total Loss: {results['final_loss_total']:.6e}
  Data Loss: {results['final_loss_data']:.6e}
  Physics Loss: {results['final_loss_physics']:.6e}

Best Model:
  Best Loss: {results['best_loss']:.6e}
  Best Epoch: {results['best_epoch']}

================================================================================
TRAJECTORY PREDICTION ACCURACY
================================================================================

PINN Inference Time: {results['pinn_inference_time_ms']:.2f} ms
RK8 Propagation Time: {results['rk8_compute_time_ms']:.2f} ms
Speedup Factor: {results['speedup_factor']:.0f}x

Altitude Prediction Error:
  Mean: {results['altitude_error_mean_km']:.3f} km
  Max: {results['altitude_error_max_km']:.3f} km
  Std Dev: {results['altitude_error_std_km']:.3f} km

Velocity Prediction Error:
  Mean: {results['velocity_error_mean_ms']:.2f} m/s
  Max: {results['velocity_error_max_ms']:.2f} m/s

================================================================================
ΔV OPTIMIZATION RESULTS
================================================================================

Optimization Method: {results['optimization_method']}
Target Lifetime: {results['target_lifetime_years']:.1f} years

Results:
  Optimal ΔV (Hybrid): {results['optimal_dv_m_s']:.2f} m/s
  Lifetime at Optimal: {results['lifetime_at_optimal_years']:.2f} years
  No-Burn Lifetime (Sail Only): {results['no_burn_lifetime_years']:.2f} years
  Propulsive-Only ΔV: {results['propulsive_only_dv_m_s']:.1f} m/s

Performance:
  Fuel Savings vs. Propulsive-Only: {results['fuel_savings_percent']:.1f}%
  ΔV Reduction Factor: {results['dv_reduction_factor']:.1f}x
  Function Evaluations: {results['optimization_n_evals']}
  Optimization Time: {results['optimization_time_s']:.2f} s

================================================================================
COMPLIANCE ANALYSIS
================================================================================

IADC 25-Year Rule: {'COMPLIANT' if results['lifetime_at_optimal_years'] <= 25 else 'NON-COMPLIANT'}
5-Year Target: {'ACHIEVED' if results['lifetime_at_optimal_years'] <= 5 else 'NOT ACHIEVED'}

Recommended Strategy:
  1. Perform retrograde burn of {results['optimal_dv_m_s']:.2f} m/s immediately after mission completion
  2. Deploy {results['sail_area_m2']:.1f} m² drag sail
  3. Expected re-entry within {results['lifetime_at_optimal_years']:.2f} years

================================================================================
CONCLUSIONS
================================================================================

The SMART-DEORBIT system successfully demonstrates:

1. PINN-based trajectory prediction achieves {results['speedup_factor']:.0f}x speedup over
   traditional numerical integration with mean altitude error of 
   {results['altitude_error_mean_km']:.3f} km.

2. The hybrid de-orbit strategy (small burn + drag sail) reduces fuel
   requirements by {results['fuel_savings_percent']:.1f}% compared to propulsion-only de-orbit.

3. For the {results['mission_name']} mission at {results['initial_altitude_km']:.0f} km altitude,
   only {results['optimal_dv_m_s']:.2f} m/s of ΔV is required to achieve 5-year compliance,
   compared to {results['propulsive_only_dv_m_s']:.1f} m/s without the drag sail.

================================================================================
                              END OF REPORT
================================================================================
"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Saved: {save_path}")


def run_complete_demo(mission: str = "PSLV", output_dir: str = "./outputs"):
    """
    Run complete SMART-DEORBIT demonstration pipeline.
    
    Parameters
    ----------
    mission : str
        Mission name ('PSLV' or 'SSLV').
    output_dir : str
        Output directory path.
    """
    print("=" * 70)
    print("SMART-DEORBIT SYSTEM — Complete Demonstration")
    print("=" * 70)
    print(f"Mission: {mission}")
    print(f"Output Directory: {output_dir}")
    print()
    
    # Setup
    output_path = setup_output_directory(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load mission configuration
    config = get_mission_config(mission)
    print(f"Loaded configuration for {config.name}")
    print(f"  Altitude: {config.orbit_altitude / 1000:.0f} km")
    print(f"  Mass: {config.dry_mass:.0f} kg")
    print(f"  Sail Area: {config.drag_sail_area:.1f} m²")
    print()
    
    # Generate initial state
    state0 = circular_orbit_state(config.orbit_altitude, config.orbit_inclination)
    elements = state_to_keplerian(state0)
    
    print("Initial Orbital Elements:")
    print(f"  Perigee: {elements.perigee_km:.0f} km")
    print(f"  Apogee: {elements.apogee_km:.0f} km")
    print(f"  Inclination: {elements.inclination_deg:.1f}°")
    print(f"  Period: {elements.period_min:.1f} min")
    print()
    
    # Results dictionary for report
    results = {
        'mission_name': config.name,
        'initial_altitude_km': config.orbit_altitude / 1000,
        'initial_inclination_deg': config.orbit_inclination,
        'mass_kg': config.dry_mass,
        'sail_area_m2': config.drag_sail_area,
        'cd': config.Cd,
    }
    
    # =========================================================================
    # PHASE 1: PINN Training
    # =========================================================================
    print("=" * 60)
    print("PHASE 1: PINN Training")
    print("=" * 60)
    
    pinn_config = PINNConfig(
        hidden_layers=[128, 128, 128, 128],
        learning_rate=1e-3,
        epochs=5000,
        n_collocation=2000,
        n_data=50,
        lambda_physics=1.0,
        lambda_initial=10.0,
    )
    
    model, normalizer, history, rk8_result = train_pinn(
        state0,
        Cd=config.Cd,
        A=config.drag_sail_area,
        mass=config.dry_mass,
        config=pinn_config,
        t_span_years=8.0,
        architecture="standard",
        save_dir=str(output_path / "models"),
        verbose=True,
    )
    
    # Record training results
    results.update({
        'pinn_architecture': 'standard',
        'pinn_hidden_layers': str(pinn_config.hidden_layers),
        'training_epochs': len(history.epochs),
        'training_time_s': history.train_time_s,
        'final_loss_total': history.loss_total[-1],
        'final_loss_data': history.loss_data[-1],
        'final_loss_physics': history.loss_physics[-1],
        'best_loss': history.best_loss,
        'best_epoch': history.best_epoch,
        'rk8_compute_time_ms': rk8_result.compute_time_s * 1000,
    })
    
    # Plot training history
    plot_training_history(history, str(output_path / "plots" / "01_training_history.png"))
    
    print()
    
    # =========================================================================
    # PHASE 2: Trajectory Prediction & Validation
    # =========================================================================
    print("=" * 60)
    print("PHASE 2: Trajectory Prediction & Validation")
    print("=" * 60)
    
    # Predict with PINN
    t_test = rk8_result.t_years
    pinn_states, pinn_time_ms = predict_trajectory(model, normalizer, t_test)
    
    # Compute errors
    min_len = min(len(rk8_result.altitudes_km), len(pinn_states))
    pinn_alt = (np.linalg.norm(pinn_states[:, :3], axis=1) - PhysicalConstants.R_EARTH_MEAN) / 1000
    pinn_vel = np.linalg.norm(pinn_states[:, 3:], axis=1) / 1000
    rk8_vel = np.linalg.norm(rk8_result.states[:, 3:], axis=1) / 1000
    
    alt_error = np.abs(pinn_alt[:min_len] - rk8_result.altitudes_km[:min_len])
    vel_error = np.abs(pinn_vel[:min_len] - rk8_vel[:min_len])
    
    speedup = rk8_result.compute_time_s * 1000 / pinn_time_ms
    
    print(f"PINN Inference Time: {pinn_time_ms:.1f} ms")
    print(f"RK8 Compute Time: {rk8_result.compute_time_s * 1000:.1f} ms")
    print(f"Speedup: {speedup:.0f}x")
    print(f"Altitude Error - Mean: {np.mean(alt_error):.3f} km, Max: {np.max(alt_error):.3f} km")
    print(f"Velocity Error - Mean: {np.mean(vel_error):.3f} km/s")
    
    results.update({
        'pinn_inference_time_ms': pinn_time_ms,
        'speedup_factor': speedup,
        'altitude_error_mean_km': float(np.mean(alt_error)),
        'altitude_error_max_km': float(np.max(alt_error)),
        'altitude_error_std_km': float(np.std(alt_error)),
        'velocity_error_mean_ms': float(np.mean(vel_error) * 1000),
        'velocity_error_max_ms': float(np.max(vel_error) * 1000),
    })
    
    # Plot trajectory comparison
    plot_trajectory_comparison(
        rk8_result, pinn_states, t_test,
        str(output_path / "plots" / "02_trajectory_comparison.png")
    )
    
    print()
    
    # =========================================================================
    # PHASE 3: ΔV Optimization
    # =========================================================================
    print("=" * 60)
    print("PHASE 3: ΔV Optimization")
    print("=" * 60)
    
    opt_result = optimize_delta_v(
        state0,
        Cd=config.Cd,
        A=config.drag_sail_area,
        mass=config.dry_mass,
        target_lifetime=5.0,
        method=OptimizationMethod.HYBRID,
        verbose=True,
    )
    
    results.update({
        'optimization_method': 'hybrid',
        'target_lifetime_years': 5.0,
        'optimal_dv_m_s': opt_result.optimal_dv,
        'lifetime_at_optimal_years': opt_result.lifetime_at_optimal,
        'no_burn_lifetime_years': opt_result.no_burn_lifetime,
        'propulsive_only_dv_m_s': opt_result.propulsive_only_dv,
        'fuel_savings_percent': opt_result.fuel_savings_percent,
        'dv_reduction_factor': opt_result.propulsive_only_dv / opt_result.optimal_dv if opt_result.optimal_dv > 0 else np.inf,
        'optimization_n_evals': opt_result.n_function_evaluations,
        'optimization_time_s': opt_result.compute_time_s,
    })
    
    # Plot optimization results
    plot_optimization_results(
        opt_result,
        str(output_path / "plots" / "03_optimization_results.png")
    )
    
    print()
    
    # =========================================================================
    # PHASE 4: Pareto Front Analysis
    # =========================================================================
    print("=" * 60)
    print("PHASE 4: Pareto Front Analysis")
    print("=" * 60)
    
    pareto_result = compute_pareto_front(
        state0, config.Cd, config.drag_sail_area, config.dry_mass,
        dv_range=(0, 30),
        n_points=30,
        verbose=True,
    )
    
    print(f"Pareto-optimal points: {pareto_result.n_pareto_points}")
    
    # Plot Pareto front
    plot_pareto_front(
        pareto_result,
        str(output_path / "plots" / "04_pareto_front.png")
    )
    
    print()
    
    # =========================================================================
    # PHASE 5: Generate Report
    # =========================================================================
    print("=" * 60)
    print("PHASE 5: Report Generation")
    print("=" * 60)
    
    txt_report_path = output_path / "reports" / f"analysis_report_{timestamp}.txt"
    generate_report(results, txt_report_path)
    
    # Generate PDF Report
    print(f"  Generating PDF Technical Report...")
    try:
        pdf_gen = ReportGenerator(config, output_dir=output_path / "reports")
        pdf_gen.add_mission_data()
        pdf_gen.add_optimization_results(opt_result)
        
        # Add plots
        plot_files = [
            output_path / "plots" / "01_training_history.png",
            output_path / "plots" / "02_trajectory_comparison.png",
            output_path / "plots" / "03_optimization_results.png",
            output_path / "plots" / "04_pareto_front.png"
        ]
        pdf_gen.add_plots(plot_files)
        
        pdf_path = pdf_gen.save_document()
        print(f"  Saved: {pdf_path}")
    except Exception as pdf_err:
        print(f"  Warning: PDF generation failed: {pdf_err}")
    
    # Save results as JSON
    with open(output_path / "data" / f"results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {output_path / 'data' / f'results_{timestamp}.json'}")
    
    # Save model
    torch_save_path = output_path / "models" / "pinn_model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'normalizer_scales': normalizer.get_scales_dict(),
        'config': pinn_config.__dict__,
        'results': results,
    }, torch_save_path)
    print(f"  Saved: {torch_save_path}")
    
    print()
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_path.absolute()}")
    print(f"\nKey Results for {config.name}:")
    print(f"  • Optimal ΔV: {opt_result.optimal_dv:.2f} m/s")
    print(f"  • Lifetime: {opt_result.lifetime_at_optimal:.2f} years")
    print(f"  • Fuel Savings: {opt_result.fuel_savings_percent:.1f}%")
    print(f"  • PINN Speedup: {speedup:.0f}x")
    print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SMART-DEORBIT System — Complete Demonstration"
    )
    parser.add_argument(
        "--mission", "-m",
        type=str,
        default="PSLV",
        choices=["PSLV", "SSLV"],
        help="Mission scenario (PSLV or SSLV)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./outputs",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_complete_demo(args.mission, args.output_dir)
        return 0
    except Exception as e:
        print(f"\nERROR: Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
