"""
SMART-DEORBIT SYSTEM — Physics-Informed Neural Networks
========================================================
Advanced PINN implementations for orbital trajectory prediction with
multiple architecture variants, adaptive loss weighting, and comprehensive
training pipelines.

Architecture Variants:
- Standard MLP (Multi-Layer Perceptron)
- Residual Network (ResNet-style)
- Fourier Feature Network (for multi-scale learning)
- Attention-enhanced Network

Physics Constraints:
- Two-body dynamics (Keplerian)
- J2 perturbation
- Atmospheric drag
- Initial conditions enforcement

Training Features:
- Adaptive loss weighting
- Learning rate scheduling
- Gradient clipping
- Early stopping
- Checkpoint saving
"""

import time as time_module
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import json

try:
    import trackio
    TRACKIO_AVAILABLE = True
except ImportError:
    TRACKIO_AVAILABLE = False

from config import (
    PhysicalConstants, AtmosphereModel, AtmosphereModelType,
    PINNConfig, PINN_STANDARD, PINN_HIGH_ACCURACY,
    circular_orbit_state, get_pinn_config
)
from orbital_mechanics import (
    propagate_orbit, gravity_j2, gravity_zonal_harmonics,
    atmospheric_drag, state_to_keplerian
)


# =============================================================================
# SECTION 1: NORMALIZATION UTILITIES
# =============================================================================

class StateNormalizer:
    """
    Normalization/denormalization for stable PINN training.
    
    Proper scaling is critical for neural network training stability.
    We use characteristic scales based on orbital mechanics:
    - Length: Earth radius (~6371 km)
    - Velocity: Circular orbital velocity at LEO (~7.5 km/s)
    - Time: Mission duration or orbital period
    - Acceleration: Surface gravity (~9.8 m/s²)
    """
    
    def __init__(self, r_scale: float = PhysicalConstants.R_EARTH_MEAN, 
                 t_scale: float = PhysicalConstants.SECONDS_PER_YEAR):
        self.r_scale = r_scale
        self.t_scale = t_scale
        self.v_scale = r_scale / t_scale
        self.a_scale = r_scale / (t_scale ** 2)
    
    def normalize_time(self, t: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return t / self.t_scale
    
    def denormalize_time(self, t_norm: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return t_norm * self.t_scale
    
    def normalize_state(self, state: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(state, torch.Tensor)
        if is_tensor:
            s = state.clone().float()
            s[..., :3] /= self.r_scale
            s[..., 3:] /= self.v_scale
        else:
            s = np.array(state, dtype=np.float32).copy()
            s[..., :3] /= self.r_scale
            s[..., 3:] /= self.v_scale
        return s
    
    def denormalize_state(self, state_norm: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(state_norm, torch.Tensor)
        if is_tensor:
            s = state_norm.clone()
            s[..., :3] *= self.r_scale
            s[..., 3:] *= self.v_scale
        else:
            s = np.array(state_norm, dtype=np.float32).copy()
            s[..., :3] *= self.r_scale
            s[..., 3:] *= self.v_scale
        return s
    
    def normalize_acceleration(self, a: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Normalize acceleration."""
        is_tensor = isinstance(a, torch.Tensor)
        if is_tensor:
            return a / self.a_scale
        else:
            return np.array(a, dtype=np.float32) / self.a_scale
    
    def get_scales_dict(self) -> Dict[str, float]:
        """Return all scales as dictionary."""
        return {
            'r_scale': self.r_scale,
            'v_scale': self.v_scale,
            't_scale': self.t_scale,
            'a_scale': self.a_scale,
        }


# =============================================================================
# SECTION 2: ACTIVATION FUNCTIONS
# =============================================================================

class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GaussianActivation(nn.Module):
    """Gaussian activation: exp(-x²)."""
    def forward(self, x):
        return torch.exp(-x * x)


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'swish': Swish(),
        'sigmoid': nn.Sigmoid(),
        'gaussian': GaussianActivation(),
        'sin': lambda: lambda x: torch.sin(x),  # Special handling needed
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. "
                        f"Available: {list(activations.keys())}")
    
    return activations[name]


# =============================================================================
# SECTION 3: PINN ARCHITECTURES
# =============================================================================

class OrbitalPINN(nn.Module):
    """
    Standard Multi-Layer Perceptron PINN for orbital trajectory prediction.
    
    Input:  Normalized time t ∈ [0, T/t_scale]
    Output: Normalized state [x, y, z, vx, vy, vz]
    
    The network learns to map time to orbital state, with physics
    constraints enforced through the loss function.
    """
    
    def __init__(self, config: PINNConfig = None):
        """
        Initialize PINN architecture.
        
        Parameters
        ----------
        config : PINNConfig, optional
            Network configuration. Uses standard config if None.
        """
        super().__init__()
        
        self.config = config or PINN_STANDARD
        
        # Build network layers
        layers = []
        in_dim = 1  # Time input
        
        # Fourier features (optional, for multi-scale learning)
        if self.config.fourier_features:
            self.fourier_encoder = FourierFeatures(
                input_dim=1,
                num_freqs=self.config.fourier_num_freqs
            )
            in_dim = 2 * self.config.fourier_num_freqs
        
        # Hidden layers
        for h_dim in self.config.hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            
            # Dropout (optional)
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
            
            # Activation
            layers.append(get_activation(self.config.activation))
            in_dim = h_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer (6-dimensional state)
        self.output_layer = nn.Linear(in_dim, 6)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for m in self.hidden_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        
        # Output layer starts very small to prevent gradient explosion
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.01)
        nn.init.constant_(self.output_layer.bias, 0.5)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict state from time.
        
        Parameters
        ----------
        t : torch.Tensor, shape (N, 1)
            Normalized time values.
            
        Returns
        -------
        state : torch.Tensor, shape (N, 6)
            Predicted normalized state.
        """
        if self.config.fourier_features and hasattr(self, 'fourier_encoder'):
            t = self.fourier_encoder(t)
        
        x = self.hidden_layers(t)
        return self.output_layer(x)
    
    def predict_with_uncertainty(self, t: torch.Tensor, 
                                  n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict state with Monte Carlo dropout uncertainty.
        
        Parameters
        ----------
        t : torch.Tensor, shape (N, 1)
            Time values.
        n_samples : int
            Number of Monte Carlo samples.
            
        Returns
        -------
        mean : torch.Tensor, shape (N, 6)
            Mean prediction.
        std : torch.Tensor, shape (N, 6)
            Predictive standard deviation.
        """
        self.train()  # Enable dropout
        predictions = []
        
        for _ in range(n_samples):
            pred = self(t)
            predictions.append(pred.unsqueeze(0))
        
        self.eval()
        
        predictions = torch.cat(predictions, dim=0)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std


class ResidualPINN(nn.Module):
    """
    Residual Network PINN with skip connections.
    
    Residual connections help with gradient flow in deep networks
    and enable training of much deeper architectures.
    """
    
    def __init__(self, config: PINNConfig = None):
        super().__init__()
        
        self.config = config or PINN_STANDARD
        
        # Input projection
        in_dim = 1
        if self.config.fourier_features:
            self.fourier_encoder = FourierFeatures(1, self.config.fourier_num_freqs)
            in_dim = 2 * self.config.fourier_num_freqs
        
        self.input_proj = nn.Linear(in_dim, self.config.hidden_layers[0])
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(self.config.hidden_layers) - 1):
            self.blocks.append(
                ResidualBlock(
                    self.config.hidden_layers[i],
                    self.config.hidden_layers[i + 1],
                    activation=self.config.activation,
                    dropout=self.config.dropout_rate
                )
            )
        
        # Output layer
        self.output_layer = nn.Linear(self.config.hidden_layers[-1], 6)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.config.fourier_features and hasattr(self, 'fourier_encoder'):
            t = self.fourier_encoder(t)
        
        x = self.input_proj(t)
        
        for block in self.blocks:
            x = block(x)
        
        return self.output_layer(x)


class ResidualBlock(nn.Module):
    """Residual block with optional dropout."""
    
    def __init__(self, in_dim: int, out_dim: int, 
                 activation: str = 'tanh', dropout: float = 0.0):
        super().__init__()
        
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Projection for skip connection if dimensions differ
        self.skip_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_proj(x)
        
        out = self.linear1(x)
        out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.activation(out)
        
        return out + residual


class FourierFeatures(nn.Module):
    """Fourier feature mapping for learning multi-scale functions."""
    
    def __init__(self, input_dim: int, num_freqs: int, 
                 sigma: float = 10.0, trainable: bool = False):
        super().__init__()
        # Initialize random frequencies
        B = torch.randn(input_dim, num_freqs) * sigma
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1), B: (1, num_freqs) -> proj: (N, num_freqs)
        proj = 2 * np.pi * torch.matmul(x, self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class AttentionPINN(nn.Module):
    """
    Attention-enhanced PINN with self-attention layers.
    
    Uses multi-head self-attention to capture long-range
    dependencies in the temporal domain.
    """
    
    def __init__(self, config: PINNConfig = None):
        super().__init__()
        
        self.config = config or PINN_STANDARD
        
        # Input encoding
        in_dim = 1
        if self.config.fourier_features:
            self.fourier_encoder = FourierFeatures(1, self.config.fourier_num_freqs)
            in_dim = 2 * self.config.fourier_num_freqs
        
        self.input_proj = nn.Linear(in_dim, self.config.hidden_layers[0])
        
        # Transformer-style attention layers
        d_model = self.config.hidden_layers[0]
        nhead = 4  # Number of attention heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        
        # Feedforward layers
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.config.hidden_layers[1] if len(self.config.hidden_layers) > 1 else d_model),
            nn.Tanh(),
            nn.Linear(self.config.hidden_layers[1] if len(self.config.hidden_layers) > 1 else d_model, d_model),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Output
        self.output_layer = nn.Linear(d_model, 6)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.config.fourier_features and hasattr(self, 'fourier_encoder'):
            t = self.fourier_encoder(t)
        
        # Project to model dimension
        x = self.input_proj(t).unsqueeze(1)  # Add sequence dimension
        
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Feedforward
        x_norm = self.norm2(x)
        ff_out = self.ffn(x_norm)
        x = x + ff_out
        
        # Output
        return self.output_layer(x.squeeze(1))


def create_pinn(config: PINNConfig = None, 
                architecture: str = "standard") -> nn.Module:
    """
    Factory function to create PINN with specified architecture.
    
    Parameters
    ----------
    config : PINNConfig, optional
        Network configuration.
    architecture : str
        Architecture type: 'standard', 'residual', 'fourier', 'attention'.
        
    Returns
    -------
    model : nn.Module
        PINN model.
    """
    architectures = {
        'standard': OrbitalPINN,
        'residual': ResidualPINN,
        'fourier': lambda c: OrbitalPINN(PINNConfig(
            **(c.__dict__ if c else {}), fourier_features=True
        )),
        'attention': AttentionPINN,
    }
    
    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Available: {list(architectures.keys())}")
    
    return architectures[architecture](config)


# =============================================================================
# SECTION 4: PHYSICS LOSS COMPUTATION
# =============================================================================

def compute_physics_residual(model: nn.Module,
                             t_colloc: torch.Tensor,
                             normalizer: StateNormalizer,
                             Cd: float,
                             A: float,
                             mass: float,
                             gravity_model: str = "j2",
                             atmosphere_model: AtmosphereModelType = AtmosphereModelType.EXPONENTIAL) -> Tuple[torch.Tensor, Dict]:
    """
    Compute physics residual (ODE residual) for PINN loss using nondimensionalized units.
    
    The nondimensionalized ODE is:
        d²r_hat/dt_hat² = (T²/R) * a_physics(R*r_hat, (R/T)*v_hat)
    """
    model.train()
    t_colloc = t_colloc.requires_grad_(True)
    
    # Forward pass (Normalized units)
    state_norm = model(t_colloc)
    r_norm = state_norm[:, :3]
    v_norm = state_norm[:, 3:]
    
    # Scaling factors
    R = normalizer.r_scale
    T = normalizer.t_scale
    V = normalizer.v_scale # Should be R/T
    
    # First derivative: dr_hat/dt_hat
    dr_dt_norm = torch.zeros_like(r_norm)
    for i in range(3):
        dr_dt_norm[:, i] = torch.autograd.grad(
            r_norm[:, i].sum(), t_colloc, 
            create_graph=True, retain_graph=True
        )[0].squeeze()
        
    # Second derivative: d²r_hat/dt_hat²
    d2r_dt2_norm = torch.zeros_like(r_norm)
    for i in range(3):
        d2r_dt2_norm[:, i] = torch.autograd.grad(
            dr_dt_norm[:, i].sum(), t_colloc,
            create_graph=True, retain_graph=True
        )[0].squeeze()
        
    # Physical states for acceleration calculation
    r_phys = r_norm * R
    v_phys = dr_dt_norm * (R / T)
    r_mag = torch.norm(r_phys, dim=1, keepdim=True)
    
    # 1. Central Gravity (Vectorized)
    a_grav = -PhysicalConstants.MU_EARTH * r_phys / (r_mag**3 + 1e-6)
    
    # 2. J2 Perturbation (Vectorized)
    if gravity_model in ["j2", "full"]:
        x = r_phys[:, 0:1]
        y = r_phys[:, 1:2]
        z = r_phys[:, 2:3]
        r2 = r_mag**2
        RE = PhysicalConstants.R_EARTH_MEAN
        J2 = PhysicalConstants.J2
        mu = PhysicalConstants.MU_EARTH
        
        j2_factor = 1.5 * J2 * mu * (RE**2) / (r_mag**5 + 1e-30)
        z2_r2 = (z**2) / (r2 + 1e-30)
        
        ax_j2 = -j2_factor * x * (1.0 - 5.0 * z2_r2)
        ay_j2 = -j2_factor * y * (1.0 - 5.0 * z2_r2)
        az_j2 = -j2_factor * z * (3.0 - 5.0 * z2_r2)
        
        a_grav = a_grav + torch.cat([ax_j2, ay_j2, az_j2], dim=1)
        
    # 3. Atmospheric Drag (Vectorized approximation)
    alt = r_mag - PhysicalConstants.R_EARTH_MEAN
    
    # Using exponential model for differentiability
    # Altitude-dependent scale height for better LEO approximation
    rho0 = 1.225
    # Scale height increases with altitude: ~7.5km at surface, ~50-90km at LEO
    # Use a smooth approximation: H = 7500 + 0.078 * alt (gives ~70km at 800km alt)
    H = 7500.0 + 0.078 * torch.clamp(alt, min=0)
    rho = rho0 * torch.exp(-torch.clamp(alt, min=0) / H)
    
    # Mask drag above 1000km
    rho = rho * (alt < 1000e3).float()
    
    v_mag_phys = torch.norm(v_phys, dim=1, keepdim=True)
    B = Cd * A / mass
    a_drag = -0.5 * rho * B * v_mag_phys * v_phys
    
    # Total physical acceleration
    a_total_phys = a_grav + a_drag
    
    # Convert physics to normalized residual
    # a_hat_expected = (T²/R) * a_total_phys
    a_norm_expected = a_total_phys * (T**2 / R)
    
    residual = d2r_dt2_norm - a_norm_expected
    
    # Also enforce velocity consistency: v_norm = dr_norm/dt_norm
    vel_residual = v_norm - dr_dt_norm
    
    total_residual = torch.cat([vel_residual, residual], dim=1)
    
    diagnostics = {
        'a_grav_norm': torch.norm(a_norm_expected, dim=1).mean().item(),
        'residual_mag': torch.norm(total_residual, dim=1).mean().item(),
    }
    
    return total_residual, diagnostics


# =============================================================================
# SECTION 5: ADAPTIVE LOSS WEIGHTING
# =============================================================================

class AdaptiveLossWeights:
    """
    Adaptive loss weighting using gradient norm balancing.
    Ensures that data and physics gradients have similar magnitudes.
    """
    def __init__(self, method: str = "gradient_norm", **kwargs):
        self.method = method
        self.lambda_data = kwargs.get('lambda_data', 1.0)
        self.lambda_physics = kwargs.get('lambda_physics', 1.0)
        self.lambda_initial = kwargs.get('lambda_initial', 1.0)
        self.w_physics = self.lambda_physics
        self.w_initial = self.lambda_initial

    def compute_weights(self, losses: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, float]:
        if self.method == "constant":
            return {'data': self.lambda_data, 'physics': self.lambda_physics, 'initial': self.lambda_initial}
        
        # Gradient-based balancing (Wang et al. 2021)
        # We want to balance w_physics * |grad(L_physics)| ≈ |grad(L_data)|
        model.zero_grad()
        losses['data'].backward(retain_graph=True)
        grad_data = self._get_max_grad(model)
        
        model.zero_grad()
        losses['physics'].backward(retain_graph=True)
        grad_physics = self._get_max_grad(model)
        
        if grad_physics > 0:
            alpha = 0.9 # Smoothing
            new_w = grad_data / (grad_physics + 1e-8)
            self.w_physics = alpha * self.w_physics + (1 - alpha) * new_w
            
        return {'data': self.lambda_data, 'physics': self.w_physics, 'initial': self.w_initial}

    def _get_max_grad(self, model):
        max_grad = 0.0
        for p in model.parameters():
            if p.grad is not None:
                max_grad = max(max_grad, p.grad.abs().max().item())
        return max_grad


# =============================================================================
# SECTION 6: TRAINING PIPELINE
# =============================================================================

@dataclass
class TrainingHistory:
    """Container for training history and metrics."""
    epochs: List[int] = field(default_factory=list)
    loss_total: List[float] = field(default_factory=list)
    loss_data: List[float] = field(default_factory=list)
    loss_physics: List[float] = field(default_factory=list)
    loss_initial: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    train_time_s: float = 0.0
    best_epoch: int = 0
    best_loss: float = float('inf')
    
    def to_dict(self) -> Dict:
        return {
            'epochs': self.epochs,
            'loss_total': self.loss_total,
            'loss_data': self.loss_data,
            'loss_physics': self.loss_physics,
            'loss_initial': self.loss_initial,
            'learning_rate': self.learning_rate,
            'train_time_s': self.train_time_s,
            'best_epoch': self.best_epoch,
            'best_loss': self.best_loss,
        }
    
    def save(self, filepath: str):
        """Save history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingHistory':
        """Load history from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


def generate_training_data(state0: np.ndarray,
                           Cd: float,
                           A: float,
                           mass: float,
                           t_span_years: float,
                           n_points: int = 50,
                           strategy: str = "uniform") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate reference trajectory data using RK8 propagator.
    
    Parameters
    ----------
    state0 : np.ndarray, shape (6,)
        Initial state.
    Cd, A, mass : float
        Drag parameters.
    t_span_years : float
        Propagation duration [years].
    n_points : int
        Number of data points.
    strategy : str
        Sampling strategy: 'uniform', 'logarithmic', 'adaptive'.
        
    Returns
    -------
    t_data : np.ndarray, shape (n_points,)
        Time points [seconds].
    states_data : np.ndarray, shape (n_points, 6)
        State vectors.
    propagation_result : dict
        Full propagation result.
    """
    # Propagate reference trajectory
    result = propagate_orbit(
        state0, t_span_years, Cd, A, mass,
        dt_output_days=1.0,
        max_step_days=0.5,
        verbose=False,
    )
    
    if not result.success or len(result.t_seconds) == 0:
        raise RuntimeError("Reference trajectory propagation failed")
    
    # Sample points based on strategy
    total_points = len(result.t_seconds)
    
    if strategy == "uniform":
        indices = np.linspace(0, total_points - 1, n_points, dtype=int)
    elif strategy == "logarithmic":
        # More points early in trajectory where changes are faster
        indices = np.unique(np.logspace(0, np.log10(total_points - 1), n_points, dtype=int))
    elif strategy == "adaptive":
        # Sample more densely where altitude changes rapidly
        alt_diff = np.abs(np.diff(result.altitudes_km))
        importance = alt_diff / np.max(alt_diff) + 0.1
        probabilities = importance / np.sum(importance)
        indices = np.sort(np.random.choice(total_points - 1, min(n_points, total_points - 1), p=probabilities, replace=False))
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    t_data = result.t_seconds[indices]
    states_data = result.states[indices]
    
    return t_data, states_data, result


def train_pinn(state0: np.ndarray,
               Cd: float,
               A: float,
               mass: float,
               config: PINNConfig = None,
               t_span_years: float = 8.0,
               architecture: str = "standard",
               device: str = None,
               callback: Optional[Callable] = None,
               save_dir: Optional[str] = None,
               verbose: bool = True) -> Tuple[nn.Module, StateNormalizer, TrainingHistory, Dict]:
    """
    Complete PINN training pipeline.
    
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
    config : PINNConfig, optional
        Training configuration.
    t_span_years : float
        Training horizon [years].
    architecture : str
        Network architecture: 'standard', 'residual', 'fourier', 'attention'.
    device : str, optional
        Training device ('cuda', 'cpu'). Auto-detected if None.
    callback : callable, optional
        Callback function called every N epochs: callback(epoch, losses_dict).
    save_dir : str, optional
        Directory to save model checkpoints.
    verbose : bool
        Print training progress.
        
    Returns
    -------
    model : nn.Module
        Trained PINN model.
    normalizer : StateNormalizer
        State normalizer.
    history : TrainingHistory
        Training history.
    rk8_result : dict
        Reference RK8 propagation result.
    """
    # Setup
    config = config or PINN_STANDARD
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    if verbose:
        print(f"Training PINN on device: {device}")
        print(f"  Architecture: {architecture}")
        print(f"  Hidden layers: {config.hidden_layers}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Collocation points: {config.n_collocation}")
    
    # Generate reference data
    if verbose:
        print("\nGenerating reference trajectory with RK8 propagator...")
    
    t_data_np, states_data_np, rk8_result = generate_training_data(
        state0, Cd, A, mass, t_span_years, config.n_data
    )
    
    if verbose:
        print(f"  Generated {len(t_data_np)} data points")
        print(f"  RK8 propagation time: {rk8_result.compute_time_s:.2f} s")
        print(f"  Lifetime: {rk8_result.lifetime_years:.2f} years")
    
    # Setup normalizer
    normalizer = StateNormalizer(t_scale=t_span_years * PhysicalConstants.SECONDS_PER_YEAR)
    
    # Prepare tensors
    t_data = torch.tensor(
        normalizer.normalize_time(t_data_np),
        dtype=torch.float32,
        device=device
    ).unsqueeze(1)
    
    states_data_norm = normalizer.normalize_state(states_data_np)
    states_data = torch.tensor(states_data_norm, dtype=torch.float32, device=device)
    
    # Initial condition
    t_initial = torch.zeros(1, 1, device=device)
    state_initial_norm = normalizer.normalize_state(state0.reshape(1, -1))
    state_initial = torch.tensor(state_initial_norm, dtype=torch.float32, device=device)
    
    # Create model
    model = create_pinn(config, architecture).to(device)
    
    # SAFE INITIALIZATION: Initialize output layer to small values
    # to preventMU_EARTH from causing immediate explosion
    if hasattr(model, 'output_layer'):
        nn.init.xavier_uniform_(model.output_layer.weight, gain=0.01)
        nn.init.constant_(model.output_layer.bias, 0.5) # Start in the middle of normalized range
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate * 0.1, # Start with smaller LR for warmup
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_schedule_step,
        gamma=config.lr_schedule_gamma
    )
    
    # Adaptive loss weights (optional)
    if config.use_adaptive_weights:
        loss_weighter = AdaptiveLossWeights(
            method="uncertainty",
            lambda_data=config.lambda_data,
            lambda_physics=config.lambda_physics,
            lambda_initial=config.lambda_initial,
        )
    else:
        loss_weighter = AdaptiveLossWeights(method="constant")
    
    # Training history
    history = TrainingHistory()
    start_time = time_module.perf_counter()
    
    # Training loop
    if verbose:
        print(f"\nTraining for {config.epochs} epochs...")
        print(f"{'Epoch':>8} | {'Total Loss':>14} | {'Data Loss':>12} | {'Physics Loss':>14} | {'LR':>10}")
        print("-" * 70)
        
    if TRACKIO_AVAILABLE:
        trackio.init(project="smart-deorbit-pinn", config=config.__dict__)
    
    best_loss = float('inf')
    patience_counter = 0
    patience = config.epochs // 10  # Early stopping patience
    
    # Pre-generate collocation points to reduce stochastic noise
    # We'll resample every few epochs if needed, but fixed points are more stable for PINNs
    t_colloc_np = np.random.uniform(0, 1, (config.n_collocation, 1)).astype(np.float32)
    t_colloc = torch.tensor(t_colloc_np, dtype=torch.float32, device=device, requires_grad=True)
    
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        
        # Periodic resampling of collocation points (e.g., every 500 epochs)
        if epoch > 0 and epoch % 500 == 0:
            t_colloc_np = np.random.uniform(0, 1, (config.n_collocation, 1)).astype(np.float32)
            t_colloc = torch.tensor(t_colloc_np, dtype=torch.float32, device=device, requires_grad=True)
        
        # Data loss
        pred_data = model(t_data)
        loss_data = F.mse_loss(pred_data, states_data)
        
        # Initial condition loss
        pred_initial = model(t_initial)
        loss_initial = F.mse_loss(pred_initial, state_initial)
        
        # Physics loss
        residual, _ = compute_physics_residual(
            model, t_colloc, normalizer, Cd, A, mass
        )
        
        # Normalize residual by characteristic acceleration
        # accel_scale = PhysicalConstants.MU_EARTH / (PhysicalConstants.R_EARTH_MEAN ** 2)
        # residual_norm = residual / accel_scale
        # Using normalized residual calculated in compute_physics_residual
        loss_physics = torch.mean(residual**2)
        
        # Combine losses
        current_losses = {
            'data': loss_data,
            'physics': loss_physics,
            'initial': loss_initial,
        }
        
        weights = loss_weighter.compute_weights(current_losses, model)
        
        loss_total = (
            weights['data'] * loss_data +
            weights['physics'] * loss_physics +
            weights['initial'] * loss_initial
        )
        
        # Backpropagation
        optimizer.zero_grad()
        
        # Check for NaN in losses
        if torch.isnan(loss_total):
            if verbose:
                print(f"\n[ERROR] NaN detected at epoch {epoch}. Stopping training.")
            if callback:
                callback(epoch, {'total': np.nan, 'data': np.nan, 'physics': np.nan, 'initial': np.nan})
            break
            
        loss_total.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
        
        optimizer.step()
        scheduler.step()
        
        # Record history
        current_lr = optimizer.param_groups[0]['lr']
        history.epochs.append(epoch)
        history.loss_total.append(loss_total.item())
        history.loss_data.append(loss_data.item())
        history.loss_physics.append(loss_physics.item())
        history.loss_initial.append(loss_initial.item())
        history.learning_rate.append(current_lr)
        
        # Track best
        if loss_total.item() < best_loss:
            best_loss = loss_total.item()
            history.best_loss = best_loss
            history.best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': config.__dict__,
                }, save_path / 'best_model.pt')
        else:
            patience_counter += 1
        
        # Callback - ALWAYS call on final epoch or early stopping
        if callback:
            is_final = (epoch == config.epochs - 1)
            is_stopped = (patience_counter >= patience and epoch > config.epochs // 4)
            if epoch % 100 == 0 or is_final or is_stopped:
                callback(epoch, {
                    'total': loss_total.item(),
                    'data': loss_data.item(),
                    'physics': loss_physics.item(),
                    'initial': loss_initial.item(),
                })
            if is_stopped:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
            
        if TRACKIO_AVAILABLE:
            trackio.log({
                "epoch": epoch,
                "loss_total": loss_total.item(),
                "loss_data": loss_data.item(),
                "loss_physics": loss_physics.item(),
                "loss_initial": loss_initial.item(),
                "learning_rate": current_lr
            })
            
            # Fire an alert if loss explodes
            if epoch > 100 and loss_total.item() > 1e3:
                trackio.alert(
                    title="Loss divergence",
                    text=f"Total loss exploded to {loss_total.item():.2e} at epoch {epoch}",
                    level=trackio.AlertLevel.ERROR
                )
        
        # Progress output
        if verbose and (epoch % 500 == 0 or epoch == config.epochs - 1):
            print(f"{epoch:8d} | {loss_total.item():14.6e} | {loss_data.item():12.6e} | "
                  f"{loss_physics.item():14.6e} | {current_lr:10.2e}")
        
        # Early stopping
        if patience_counter >= patience and epoch > config.epochs // 4:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
            
    if TRACKIO_AVAILABLE:
        trackio.finish()
    
    # Final timing
    history.train_time_s = time_module.perf_counter() - start_time
    
    if verbose:
        print(f"\nTraining complete!")
        print(f"  Total time: {history.train_time_s:.1f} s")
        print(f"  Best loss: {history.best_loss:.6e} at epoch {history.best_epoch}")
    
    # Save final model
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': config.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': history.best_loss,
            'config': config.__dict__,
        }, save_path / 'final_model.pt')
        history.save(save_path / 'training_history.json')
    
    return model, normalizer, history, rk8_result


# =============================================================================
# SECTION 7: PREDICTION AND INFERENCE
# =============================================================================

def predict_trajectory(model: nn.Module,
                       normalizer: StateNormalizer,
                       t_years_array: np.ndarray,
                       device: str = None,
                       batch_size: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Use trained PINN to predict trajectory at given time points.
    
    Parameters
    ----------
    model : nn.Module
        Trained PINN model.
    normalizer : StateNormalizer
        State normalizer.
    t_years_array : np.ndarray
        Time points [years].
    device : str, optional
        Device for inference.
    batch_size : int
        Batch size for prediction.
        
    Returns
    -------
    states : np.ndarray, shape (N, 6)
        Predicted state vectors [m, m/s].
    compute_time_ms : float
        Prediction time [milliseconds].
    """
    model.eval()
    device = device or next(model.parameters()).device
    
    t_seconds = t_years_array * PhysicalConstants.SECONDS_PER_YEAR
    t_norm = normalizer.normalize_time(t_seconds)
    
    # Batch prediction for memory efficiency
    states_norm = []
    start_time = time_module.perf_counter()
    
    with torch.no_grad():
        for i in range(0, len(t_norm), batch_size):
            t_batch = t_norm[i:i+batch_size]
            t_tensor = torch.tensor(t_batch, dtype=torch.float32, device=device).unsqueeze(1)
            pred = model(t_tensor)
            states_norm.append(pred.cpu().numpy())
    
    compute_time_ms = (time_module.perf_counter() - start_time) * 1000
    
    states_norm = np.vstack(states_norm)
    states_phys = normalizer.denormalize_state(states_norm)
    
    return states_phys, compute_time_ms


def predict_lifetime_pinn(model: nn.Module,
                          normalizer: StateNormalizer,
                          max_years: float = 30.0,
                          dt_days: float = 1.0,
                          device: str = None) -> float:
    """
    Estimate orbital lifetime from PINN predictions.
    
    Parameters
    ----------
    model : nn.Module
        Trained PINN model.
    normalizer : StateNormalizer
        State normalizer.
    max_years : float
        Maximum simulation time [years].
    dt_days : float
        Time step for sampling [days].
    device : str, optional
        Inference device.
        
    Returns
    -------
    lifetime : float
        Estimated lifetime [years].
    """
    from config import DeorbitRequirements
    
    reentry_alt = DeorbitRequirements().reentry_altitude_km * 1000
    
    t_years = np.arange(0, max_years, dt_days / 365.25)
    states, _ = predict_trajectory(model, normalizer, t_years, device)
    
    r_magnitudes = np.linalg.norm(states[:, :3], axis=1)
    altitudes = r_magnitudes - PhysicalConstants.R_EARTH_MEAN
    
    # Find first time altitude drops below reentry threshold
    below = np.where(altitudes < reentry_alt)[0]
    
    if len(below) > 0:
        return t_years[below[0]]
    return max_years


def compute_prediction_uncertainty(model: nn.Module,
                                    normalizer: StateNormalizer,
                                    t_years_array: np.ndarray,
                                    n_samples: int = 10,
                                    device: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute prediction uncertainty using Monte Carlo dropout.
    
    Parameters
    ----------
    model : nn.Module
        PINN model with dropout.
    normalizer : StateNormalizer
        State normalizer.
    t_years_array : np.ndarray
        Time points [years].
    n_samples : int
        Number of Monte Carlo samples.
    device : str, optional
        Inference device.
        
    Returns
    -------
    mean_states : np.ndarray, shape (N, 6)
        Mean predictions.
    std_states : np.ndarray, shape (N, 6)
        Predictive standard deviation.
    """
    model.eval()
    device = device or next(model.parameters()).device
    
    t_seconds = t_years_array * PhysicalConstants.SECONDS_PER_YEAR
    t_norm = normalizer.normalize_time(t_seconds)
    t_tensor = torch.tensor(t_norm, dtype=torch.float32, device=device).unsqueeze(1)
    
    mean_norm, std_norm = model.predict_with_uncertainty(t_tensor, n_samples)
    
    # Denormalize (approximate for uncertainty)
    mean_phys = normalizer.denormalize_state(mean_norm).cpu().numpy()
    std_phys = std_norm.clone()
    std_phys[..., :3] *= normalizer.r_scale
    std_phys[..., 3:] *= normalizer.v_scale
    std_phys = std_phys.cpu().numpy()
    
    return mean_phys, std_phys


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SMART-DEORBIT SYSTEM — PINN Module Test")
    print("=" * 70)
    
    # Test configuration
    from config import SSLV_CONFIG, circular_orbit_state
    
    print("\n--- Testing PINN Training (SSLV VTM @ 500 km) ---")
    
    cfg = SSLV_CONFIG
    state0 = circular_orbit_state(cfg.orbit_altitude)
    
    # Quick training test
    model, normalizer, history, rk8_result = train_pinn(
        state0,
        Cd=cfg.Cd,
        A=cfg.drag_sail_area,
        mass=cfg.dry_mass,
        config=PINNConfig(
            hidden_layers=[64, 64, 64],
            epochs=1000,
            n_collocation=500,
            n_data=30,
            learning_rate=1e-3,
        ),
        t_span_years=5.0,
        architecture="standard",
        verbose=True,
    )
    
    # Test prediction
    print("\n--- Testing PINN Prediction ---")
    t_test = np.linspace(0, 5, 100)
    states_pred, pred_time = predict_trajectory(model, normalizer, t_test)
    
    print(f"  Predicted {len(states_pred)} states in {pred_time:.1f} ms")
    print(f"  Average: {pred_time / len(states_pred):.3f} ms per state")
    
    # Compare with RK8
    rk8_states = rk8_result.states[::max(1, len(rk8_result.states)//100)][:100]
    
    pred_alt = (np.linalg.norm(states_pred[:, :3], axis=1) - PhysicalConstants.R_EARTH_MEAN) / 1000
    rk8_alt = (np.linalg.norm(rk8_states[:, :3], axis=1) - PhysicalConstants.R_EARTH_MEAN) / 1000
    
    error = np.abs(pred_alt - rk8_alt[:len(pred_alt)])
    
    print(f"\n  Altitude Error Statistics:")
    print(f"    Mean: {np.mean(error):.3f} km")
    print(f"    Max: {np.max(error):.3f} km")
    print(f"    Std: {np.std(error):.3f} km")
    
    # Speedup comparison
    speedup = rk8_result.compute_time_s * 1000 / pred_time
    print(f"\n  Speedup: {speedup:.0f}x (PINN vs RK8)")
    
    print("\n" + "=" * 70)
    print("PINN module test complete!")
