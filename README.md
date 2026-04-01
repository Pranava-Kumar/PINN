# SMART-DEORBIT System

## Physics-Informed Neural Networks for Satellite De-orbit Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

The **SMART-DEORBIT System** is an AI-accelerated solution for rapid orbital decay prediction and fuel-optimal de-orbit maneuver planning for ISRO's upper stage disposal (PSLV, SSLV).

### Problem Statement

Upper rocket stages (PSLV's PS4, SSLV's VTM) left in orbit after satellite deployment must re-enter Earth's atmosphere within **5 years** per international space debris mitigation guidelines. Traditional propulsion-only de-orbit requires significant fuel (ΔV), reducing mission capability.

### Solution

This system uses a **hybrid approach** combining:
1. **Small retrograde burn** to lower perigee
2. **Deployable drag sail** to increase atmospheric drag
3. **Physics-Informed Neural Networks (PINNs)** for AI-accelerated trajectory prediction

### Key Results

| Metric | Value |
|--------|-------|
| **PINN Speedup** | >100x vs. RK8 numerical integration |
| **Altitude Prediction Error** | <1 km (mean) |
| **Fuel Savings (PSLV @ 750 km)** | ~83% (7.5 m/s vs. 45 m/s) |
| **Fuel Savings (SSLV @ 500 km)** | ~90% (<1 m/s vs. 10 m/s) |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/smart-deorbit.git
cd smart-deorbit

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

**Option 1: Interactive Dashboard (Recommended)**
```bash
streamlit run app.py
```

**Option 2: Command-Line Demo**
```bash
python run_demo.py --mission PSLV --output-dir ./outputs
```

**Option 3: Run Tests**
```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
PINN/
├── config.py              # Physical constants, mission configs, PINN hyperparameters
├── orbital_mechanics.py   # High-fidelity orbital propagator (RK8, J2, drag)
├── pinn_model.py          # Physics-Informed Neural Network implementations
├── delta_v_optimizer.py   # ΔV optimization algorithms
├── app.py                 # Streamlit interactive dashboard
├── run_demo.py            # Complete demonstration pipeline
├── tests/
│   ├── __init__.py
│   └── test_orbital_mechanics.py  # Unit and integration tests
├── requirements.txt       # Python dependencies
├── pytest.ini            # pytest configuration
└── README.md             # This file
```

---

## 🛰️ Mission Configurations

### Pre-defined Missions

| Mission | Altitude | Mass | Sail Area | Optimal ΔV | Lifetime |
|---------|----------|------|-----------|------------|----------|
| **PSLV PS4** | 750 km | 920 kg | 10 m² | ~7.5 m/s | <5 yr |
| **SSLV VTM** | 500 km | 100 kg | 5 m² | <1.0 m/s | <5 yr |
| **CARTOSAT-2** | 630 km | 680 kg | 8 m² | ~3.0 m/s | <5 yr |
| **RESOURCESAT-2** | 817 km | 1235 kg | 15 m² | ~12 m/s | <5 yr |

### Custom Missions

You can define custom mission configurations:

```python
from config import SpacecraftConfig, circular_orbit_state

# Define custom spacecraft
my_spacecraft = SpacecraftConfig(
    name="My Satellite",
    dry_mass=500.0,           # kg
    orbit_altitude=550e3,     # m
    drag_sail_area=6.0,       # m²
    Cd=2.2,
    orbit_inclination=97.5,   # degrees
)

# Generate initial state
state0 = circular_orbit_state(
    my_spacecraft.orbit_altitude,
    my_spacecraft.orbit_inclination
)
```

---

## 🧠 PINN Architecture

### Network Structure

```
Input: t (normalized time)
       ↓
[Fourier Features] (optional, for multi-scale learning)
       ↓
Linear(1→128) + Tanh
       ↓
Linear(128→128) + Tanh
       ↓
Linear(128→128) + Tanh
       ↓
Linear(128→128) + Tanh
       ↓
Linear(128→6) → [x, y, z, vx, vy, vz]
```

### Loss Function

```
L_total = λ_data × L_MSE(data) + λ_physics × L_MSE(residual) + λ_initial × L_MSE(IC)
```

Where:
- **L_data**: MSE between PINN output and RK8 reference trajectory
- **L_physics**: MSE of ODE residual (d²r/dt² - a_physics)
- **L_initial**: MSE at t=0 (initial condition enforcement)

### Physics Constraints

The residual enforces:
```
d²r/dt² = a_gravity(J2, J3, J4) + a_drag(ρ, Cd, A, m)
```

Automatic differentiation computes derivatives through the network, ensuring physics consistency.

---

## 🔧 API Reference

### Orbital Mechanics

```python
from orbital_mechanics import propagate_orbit, state_to_keplerian, apply_delta_v

# Propagate orbit
result = propagate_orbit(
    state0,                    # Initial state [r, v]
    t_span_years=10.0,         # Propagation duration
    Cd=2.2,                    # Drag coefficient
    A=10.0,                    # Cross-sectional area
    mass=920.0,                # Spacecraft mass
)

# Convert to Keplerian elements
elements = state_to_keplerian(state0)

# Apply ΔV maneuver
new_state = apply_delta_v(state0, dv=50.0, direction="retrograde")
```

### PINN Training

```python
from pinn_model import train_pinn, predict_trajectory

# Train PINN
model, normalizer, history, rk8_result = train_pinn(
    state0,
    Cd=2.2,
    A=10.0,
    mass=920.0,
    t_span_years=8.0,
    epochs=5000,
)

# Predict trajectory
t_test = np.linspace(0, 8, 100)  # years
states, inference_time = predict_trajectory(model, normalizer, t_test)
```

### ΔV Optimization

```python
from delta_v_optimizer import optimize_delta_v, OptimizationMethod

# Find optimal ΔV
result = optimize_delta_v(
    state0,
    Cd=2.2,
    A=10.0,
    mass=920.0,
    target_lifetime=5.0,
    method=OptimizationMethod.HYBRID,
)

print(f"Optimal ΔV: {result.optimal_dv:.2f} m/s")
print(f"Lifetime: {result.lifetime_at_optimal:.2f} years")
print(f"Fuel savings: {result.fuel_savings_percent:.1f}%")
```

---

## 📊 Output Files

Running `run_demo.py` generates:

```
outputs/
├── models/
│   ├── best_model.pt          # Best PINN weights
│   └── final_model.pt         # Final PINN weights
├── plots/
│   ├── 01_training_history.png
│   ├── 02_trajectory_comparison.png
│   ├── 03_optimization_results.png
│   └── 04_pareto_front.png
├── data/
│   └── results_TIMESTAMP.json
└── reports/
    └── analysis_report_TIMESTAMP.txt
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/ -v -m unit

# Integration tests only
pytest tests/ -v -m integration

# With coverage
pytest tests/ -v --cov=. --cov-report=html
```

### Validation Cases

The test suite includes validation against:
- ISS orbital parameters (408 km, 51.6°)
- Sun-synchronous orbit characteristics
- Known analytical solutions (two-body problem)
- Conservation laws (energy, angular momentum)

---

## 📚 Physics Models

### Gravity

- **Central Body**: Keplerian two-body dynamics
- **J2 Perturbation**: Earth oblateness (primary effect)
- **J3, J4**: Higher-order zonal harmonics

### Atmosphere

- **Exponential Model**: Piecewise exponential density profile
- **US Standard 1976**: Simplified implementation
- **NRLMSISE-00**: Simplified with diurnal/latitudinal variations

### Drag

```
a_drag = -½ × ρ(h) × (Cd × A / m) × |v_rel| × v_rel
```

---

## 📈 Performance Benchmarks

### PINN Training (PSLV @ 750 km)

| Configuration | Epochs | Time | Best Loss |
|---------------|--------|------|-----------|
| Fast | 2,000 | ~30 s | 1e-4 |
| Balanced | 5,000 | ~60 s | 1e-5 |
| Accurate | 8,000 | ~120 s | 1e-6 |
| High Accuracy | 10,000 | ~180 s | 1e-7 |

### Inference Speed

| Method | Time | Speedup |
|--------|------|---------|
| RK8 (DOP853) | 500 ms | 1x |
| PINN (GPU) | 2 ms | 250x |
| PINN (CPU) | 10 ms | 50x |

---

## 🔬 Scientific Background

### Physics-Informed Neural Networks

PINNs were introduced by Raissi et al. (2019) as a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. The key innovation is embedding the PDE residual into the loss function:

```
L(θ) = L_data(θ) + λ × L_physics(θ)
```

### Orbital Decay Dynamics

The orbital decay is governed by:

```
d²r/dt² = -μ/r³ × r × [1 + Σ J_n terms] - ½ × ρ × (Cd A / m) × |v| × v
```

### De-orbit Compliance

Per IADC guidelines:
- **25-year rule**: Natural decay within 25 years
- **5-year target**: Aggressive disposal timeline
- **Direct re-entry**: Immediate de-orbit after mission

---

## 📖 References

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. **Vallado, D. A.** (2013). *Fundamentals of Astrodynamics and Applications* (4th ed.). Microcosm Press.

3. **Montenbruck, O., & Gill, E.** (2000). *Satellite Orbits: Models, Methods, Applications*. Springer.

4. **IADC** (2020). Space Debris Mitigation Guidelines. Inter-Agency Space Debris Coordination Committee.

5. **ISO 24113:2019**. Space systems — Space debris mitigation requirements.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

Developed for **ISRO's Debris-Free Space Missions Initiative** as a final year undergraduate project in collaboration with industry partners.

---

## 🙏 Acknowledgments

- Indian Space Research Organisation (ISRO)
- Physics-Informed Neural Networks research community
- Orbital mechanics and astrodynamics researchers

---

## 📞 Contact

For questions, collaboration, or bug reports, please open an issue on GitHub.

---

<div align="center">

**SMART-DEORBIT System v1.0**

*Enabling Sustainable Space Operations*

</div>
