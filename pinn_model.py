"""
SMART-DEORBIT SYSTEM — Physics-Informed Neural Network (PINN)
==============================================================
PyTorch-based PINN that embeds orbital mechanics ODEs into its
training to predict long-term orbital decay trajectories.
"""

import time as time_module
import numpy as np
import torch
import torch.nn as nn
from config import (
    MU_EARTH, R_EARTH, J2, PINN_HPARAMS,
    SECONDS_PER_YEAR, circular_orbit_state
)
from orbital_mechanics import (
    propagate_orbit, atmospheric_density
)


# ─────────────────────────────────────────────
# Normalization Helpers
# ─────────────────────────────────────────────
class StateNormalizer:
    """Normalize/denormalize state vectors for stable PINN training."""
    
    def __init__(self, r_scale=None, v_scale=None, t_scale=None):
        self.r_scale = r_scale or R_EARTH         # ~6.371e6 m
        self.v_scale = v_scale or 7500.0           # ~orbital velocity m/s
        self.t_scale = t_scale or SECONDS_PER_YEAR  # ~3.156e7 s
    
    def normalize_time(self, t):
        return t / self.t_scale
    
    def denormalize_time(self, t_norm):
        return t_norm * self.t_scale
    
    def normalize_state(self, state):
        """state: [..., 6] → normalized"""
        s = state.clone() if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)
        s[..., :3] = s[..., :3] / self.r_scale
        s[..., 3:] = s[..., 3:] / self.v_scale
        return s
    
    def denormalize_state(self, state_norm):
        """normalized → physical"""
        s = state_norm.clone()
        s[..., :3] = s[..., :3] * self.r_scale
        s[..., 3:] = s[..., 3:] * self.v_scale
        return s


# ─────────────────────────────────────────────
# PINN Architecture
# ─────────────────────────────────────────────
class OrbitalPINN(nn.Module):
    """
    Physics-Informed Neural Network for orbital trajectory prediction.
    
    Input:  normalized time t ∈ [0, T/t_scale]
    Output: normalized state [x, y, z, vx, vy, vz]
    """
    
    def __init__(self, hidden_layers=None):
        super().__init__()
        
        layers_config = hidden_layers or PINN_HPARAMS["hidden_layers"]
        
        layers = []
        in_dim = 1  # time only
        for h in layers_config:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 6))  # [x, y, z, vx, vy, vz]
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights (Xavier for tanh)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t):
        """
        Parameters
        ----------
        t : torch.Tensor, shape (N, 1)
            Normalized time values.
        
        Returns
        -------
        state : torch.Tensor, shape (N, 6)
            Predicted normalized state.
        """
        return self.net(t)


# ─────────────────────────────────────────────
# Physics Residual Computation
# ─────────────────────────────────────────────
def compute_physics_residual(model, t_colloc, normalizer, Cd, A, mass):
    """
    Compute the ODE residual: d²r/dt² - (a_gravity + a_drag) = 0
    
    Uses automatic differentiation to get dr/dt and d²r/dt².
    
    Parameters
    ----------
    model : OrbitalPINN
    t_colloc : torch.Tensor, shape (N, 1), requires_grad=True
        Collocation time points (normalized).
    normalizer : StateNormalizer
    Cd, A, mass : float
        Drag parameters.
    
    Returns
    -------
    residual : torch.Tensor, shape (N, 3)
        ODE residual for position equations.
    """
    t_colloc.requires_grad_(True)
    
    # Forward pass → normalized state
    state_norm = model(t_colloc)  # (N, 6)
    
    # Denormalize to physical units
    r_phys = state_norm[:, :3] * normalizer.r_scale   # (N, 3) in meters
    v_phys = state_norm[:, 3:] * normalizer.v_scale   # (N, 3) in m/s
    
    # Compute dr/dt via autograd (chain rule through normalization)
    dr_dt_list = []
    for i in range(3):
        grad = torch.autograd.grad(
            r_phys[:, i], t_colloc,
            grad_outputs=torch.ones_like(r_phys[:, i]),
            create_graph=True, retain_graph=True
        )[0]
        dr_dt_list.append(grad)
    dr_dt = torch.cat(dr_dt_list, dim=1) * normalizer.t_scale  # chain rule: dr/dt_real = dr/dt_norm * t_scale/1
    
    # Actually we need: dr_physical/dt_physical
    # r_phys = state_norm[:, :3] * r_scale  where state_norm = model(t_norm)
    # dr_phys/dt_phys = dr_phys/dt_norm * dt_norm/dt_phys = dr_phys/dt_norm / t_scale
    # Let's redo: just compute gradients w.r.t. normalized time, then convert
    
    # dr_phys/dt_norm already computed above in dr_dt_list (each has shape (N,1))
    # dr_phys/dt_phys = dr_phys/dt_norm / t_scale
    dr_dt_phys = torch.cat(dr_dt_list, dim=1) / normalizer.t_scale  # (N, 3) m/s
    
    # Compute d²r/dt² (second derivative)
    d2r_dt2_list = []
    for i in range(3):
        grad2 = torch.autograd.grad(
            dr_dt_list[i], t_colloc,
            grad_outputs=torch.ones_like(dr_dt_list[i]),
            create_graph=True, retain_graph=True
        )[0]
        d2r_dt2_list.append(grad2)
    d2r_dt2_phys = torch.cat(d2r_dt2_list, dim=1) / (normalizer.t_scale ** 2)  # (N, 3) m/s²
    
    # Compute expected accelerations from physics
    a_expected = torch.zeros_like(d2r_dt2_phys)
    
    for j in range(r_phys.shape[0]):
        r_j = r_phys[j].detach().cpu().numpy()
        v_j = dr_dt_phys[j].detach().cpu().numpy()
        r_mag = np.linalg.norm(r_j)
        alt = r_mag - R_EARTH
        
        # Gravity + J2
        x, y, z = r_j
        r2 = r_mag * r_mag
        j2_factor = 1.5 * J2 * (R_EARTH ** 2) / r2
        z2_r2 = (z * z) / r2
        common = -MU_EARTH / (r_mag * r2)
        
        ax = common * x * (1.0 + j2_factor * (1.0 - 5.0 * z2_r2))
        ay = common * y * (1.0 + j2_factor * (1.0 - 5.0 * z2_r2))
        az = common * z * (1.0 + j2_factor * (3.0 - 5.0 * z2_r2))
        
        a_grav = np.array([ax, ay, az])
        
        # Drag
        if alt > 0:
            rho = atmospheric_density(alt)
            B = Cd * A / mass
            v_mag = np.linalg.norm(v_j)
            a_drag = -0.5 * rho * B * v_mag * v_j
        else:
            a_drag = np.zeros(3)
        
        a_total = a_grav + a_drag
        a_expected[j] = torch.tensor(a_total, dtype=torch.float32)
    
    # Residual: d²r/dt² - a_expected = 0
    residual = d2r_dt2_phys - a_expected.to(d2r_dt2_phys.device)
    
    return residual


# ─────────────────────────────────────────────
# Training Pipeline
# ─────────────────────────────────────────────
def generate_training_data(state0, Cd, A, mass, t_span_years=5.0,
                           n_points=50):
    """
    Generate reference trajectory data using the RK8 propagator.
    
    Returns
    -------
    t_data : np.ndarray, shape (n_points,)
        Time points [seconds].
    states_data : np.ndarray, shape (n_points, 6)
        State vectors at each time point.
    full_result : dict
        Full propagation result.
    """
    result = propagate_orbit(
        state0, t_span_years, Cd, A, mass,
        dt_output_days=1.0, max_step_days=0.5
    )
    
    # Sample n_points evenly from the trajectory
    total = len(result['t_seconds'])
    indices = np.linspace(0, total - 1, n_points, dtype=int)
    
    t_data = result['t_seconds'][indices]
    states_data = result['states'][indices]
    
    return t_data, states_data, result


def train_pinn(state0, Cd, A, mass, t_span_years=5.0,
               epochs=None, callback=None):
    """
    Train a PINN for orbital trajectory prediction.
    
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
    t_span_years : float
        Prediction horizon [years].
    epochs : int or None
        Number of training epochs. None uses default from config.
    callback : callable or None
        Called every 100 epochs with (epoch, losses_dict).
    
    Returns
    -------
    model : OrbitalPINN
        Trained model.
    normalizer : StateNormalizer
        Normalizer used.
    history : dict
        Training history: 'total', 'data', 'physics' losses per epoch.
    rk8_result : dict
        Reference RK8 propagation result.
    train_time : float
        Total training time [seconds].
    """
    hparams = PINN_HPARAMS
    n_epochs = epochs or hparams["epochs"]
    n_colloc = hparams["n_collocation"]
    n_data = hparams["n_data"]
    lambda_phys = hparams["lambda_physics"]
    lr = hparams["learning_rate"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ── 1. Generate reference data ──
    print("Generating reference trajectory with RK8 propagator...")
    t_data_np, states_data_np, rk8_result = generate_training_data(
        state0, Cd, A, mass, t_span_years, n_data
    )
    rk8_compute_time = rk8_result['compute_time_s']
    print(f"  RK8 propagation took {rk8_compute_time:.2f} s")
    
    # ── 2. Setup normalizer ──
    normalizer = StateNormalizer(
        t_scale=t_span_years * SECONDS_PER_YEAR
    )
    
    # ── 3. Prepare training tensors ──
    t_data = torch.tensor(
        normalizer.normalize_time(t_data_np),
        dtype=torch.float32
    ).unsqueeze(1).to(device)
    
    states_data = torch.tensor(
        states_data_np, dtype=torch.float32
    ).to(device)
    states_data_norm = normalizer.normalize_state(states_data)
    
    # ── 4. Initialize model ──
    model = OrbitalPINN(hparams["hidden_layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=hparams["lr_schedule_step"],
        gamma=hparams["lr_schedule_gamma"]
    )
    
    # ── 5. Training loop ──
    history = {'total': [], 'data': [], 'physics': []}
    
    print(f"\nTraining PINN for {n_epochs} epochs...")
    start_time = time_module.perf_counter()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # ── Data Loss ──
        pred_data = model(t_data)
        loss_data = nn.functional.mse_loss(pred_data, states_data_norm)
        
        # ── Physics Loss ──
        # Sample collocation points
        t_colloc_np = np.random.uniform(0, 1, (n_colloc, 1)).astype(np.float32)
        t_colloc = torch.tensor(t_colloc_np, dtype=torch.float32,
                                requires_grad=True).to(device)
        
        residual = compute_physics_residual(
            model, t_colloc, normalizer, Cd, A, mass
        )
        
        # Normalize residual by typical acceleration scale
        accel_scale = MU_EARTH / (R_EARTH ** 2)  # ~9.8 m/s²
        residual_norm = residual / accel_scale
        loss_physics = torch.mean(residual_norm ** 2)
        
        # ── Total Loss ──
        loss_total = loss_data + lambda_phys * loss_physics
        
        loss_total.backward()
        optimizer.step()
        scheduler.step()
        
        # Record
        history['total'].append(loss_total.item())
        history['data'].append(loss_data.item())
        history['physics'].append(loss_physics.item())
        
        # Callback
        if callback and (epoch % 100 == 0 or epoch == n_epochs - 1):
            callback(epoch, {
                'total': loss_total.item(),
                'data': loss_data.item(),
                'physics': loss_physics.item(),
            })
        
        if epoch % 500 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:5d}/{n_epochs} | "
                  f"Total: {loss_total.item():.6e} | "
                  f"Data: {loss_data.item():.6e} | "
                  f"Physics: {loss_physics.item():.6e}")
    
    train_time = time_module.perf_counter() - start_time
    print(f"\nTraining complete in {train_time:.1f} s")
    
    return model, normalizer, history, rk8_result, train_time


def predict_trajectory(model, normalizer, t_years_array, device=None):
    """
    Use trained PINN to predict trajectory at given time points.
    
    Parameters
    ----------
    model : OrbitalPINN
    normalizer : StateNormalizer
    t_years_array : np.ndarray
        Time points [years].
    
    Returns
    -------
    states : np.ndarray, shape (N, 6)
        Predicted state vectors in physical units [m, m/s].
    compute_time_ms : float
        Prediction time [milliseconds].
    """
    if device is None:
        device = next(model.parameters()).device
    
    t_seconds = t_years_array * SECONDS_PER_YEAR
    t_norm = normalizer.normalize_time(t_seconds)
    t_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(device)
    
    start = time_module.perf_counter()
    with torch.no_grad():
        pred_norm = model(t_tensor)
    compute_time_ms = (time_module.perf_counter() - start) * 1000
    
    pred_denorm = normalizer.denormalize_state(pred_norm).cpu().numpy()
    
    return pred_denorm, compute_time_ms


def predict_lifetime_pinn(model, normalizer, max_years=30.0,
                          dt_days=1.0) -> float:
    """
    Estimate orbital lifetime from PINN predictions.
    
    Checks when altitude drops below REENTRY_ALTITUDE.
    """
    from config import REENTRY_ALTITUDE
    
    t_years = np.arange(0, max_years, dt_days / 365.25)
    states, _ = predict_trajectory(model, normalizer, t_years)
    
    r_magnitudes = np.linalg.norm(states[:, :3], axis=1)
    altitudes = r_magnitudes - R_EARTH
    
    # Find first time altitude < REENTRY_ALTITUDE
    below = np.where(altitudes < REENTRY_ALTITUDE)[0]
    if len(below) > 0:
        return t_years[below[0]]
    return max_years


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from config import SSLV_CONFIG, circular_orbit_state
    
    print("=" * 60)
    print("SMART-DEORBIT — PINN Quick Test (SSLV VTM @ 500 km)")
    print("=" * 60)
    
    cfg = SSLV_CONFIG
    state0 = circular_orbit_state(cfg["orbit_altitude"])
    
    model, normalizer, history, rk8_result, train_time = train_pinn(
        state0,
        Cd=cfg["Cd"],
        A=cfg["drag_sail_area"],
        mass=cfg["dry_mass"],
        t_span_years=8.0,
        epochs=2000,
    )
    
    # Compare predictions
    t_test = rk8_result['t_years']
    pinn_states, pinn_time_ms = predict_trajectory(model, normalizer, t_test)
    
    rk8_alt = rk8_result['altitudes_km']
    pinn_alt = (np.linalg.norm(pinn_states[:, :3], axis=1) - R_EARTH) / 1000
    
    error_km = np.abs(pinn_alt - rk8_alt[:len(pinn_alt)])
    
    print(f"\nPINN prediction time: {pinn_time_ms:.1f} ms")
    print(f"RK8 compute time: {rk8_result['compute_time_s']:.2f} s")
    print(f"Speedup: {rk8_result['compute_time_s'] * 1000 / pinn_time_ms:.0f}x")
    print(f"Max altitude error: {np.max(error_km):.2f} km")
    print(f"Mean altitude error: {np.mean(error_km):.2f} km")
