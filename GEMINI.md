# GEMINI.md - SMART-DEORBIT System Context

This document provides essential context and instructions for AI agents working on the **SMART-DEORBIT System**.

## 🚀 Project Overview

The **SMART-DEORBIT System** is an AI-accelerated solution for rapid orbital decay prediction and fuel-optimal de-orbit maneuver planning for satellite upper stages (e.g., ISRO's PSLV PS4, SSLV VTM). It uses **Physics-Informed Neural Networks (PINNs)** to achieve >100x speedup over traditional numerical integration while maintaining high physical accuracy.

### Core Objectives
- **Prediction**: Rapidly predict long-term (5-25 year) orbital decay.
- **Optimization**: Find the minimum ΔV required for a hybrid de-orbit strategy (retrograde burn + drag sail) to meet the 5-year re-entry target.
- **Compliance**: Ensure adherence to IADC and ISRO space debris mitigation guidelines.

### Technical Stack
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch (PINN implementation)
- **Numerical Physics**: SciPy (`DOP853` / RK8 integrator)
- **Visualization**: Matplotlib, Plotly
- **Interface**: Streamlit (Interactive Dashboard)

---

## 🏗️ Architecture & Key Modules

The project is structured into five functional layers:

1.  **Configuration Layer (`config.py`)**:
    - Central repository for physical constants (IERS 2010).
    - Mission profiles (PSLV, SSLV, CARTOSAT).
    - Multi-fidelity atmospheric models (Exponential, US Standard 1976, NRLMSISE-00).
    - PINN hyperparameters and optimization settings.

2.  **Physics Layer (`orbital_mechanics.py`)**:
    - High-fidelity orbital propagator.
    - Perturbations: J2, J3, J4 zonal harmonics.
    - Aerodynamic drag models with ballistic coefficients.

3.  **AI Layer (`pinn_model.py`)**:
    - **Architectures**: Standard MLP, Residual Networks, Fourier Features (for multi-scale learning), and Attention-enhanced PINNs.
    - **Physics Loss**: Enforces the ODE residual: $d^2r/dt^2 = a_{gravity} + a_{drag}$ via autograd.
    - **Training Pipeline**: Adaptive loss weighting, learning rate scheduling, and normalization strategies.

4.  **Optimization Layer (`delta_v_optimizer.py`)**:
    - Algorithms: Grid Search, Brent's Method, Nelder-Mead, and Hybrid approaches.
    - Pareto analysis for balancing fuel (ΔV) vs. orbital lifetime.

5.  **Application Layer**:
    - `app.py`: Streamlit-based interactive dashboard for real-time training and visualization.
    - `run_demo.py`: Command-line pipeline for automated reports and plot generation.

---

## 🛠️ Building and Running

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Execution
- **Interactive Dashboard (Preferred)**: `streamlit run app.py`
- **Command-Line Demo**: `python run_demo.py --mission PSLV`
- **Run Tests**: `pytest tests/ -v`

### Output Structure
Results are saved in the `outputs/` directory:
- `models/`: Trained PINN weights (`best_model.pt`).
- `plots/`: Performance and trajectory visualizations.
- `reports/`: Detailed technical analysis in text/JSON format.

---

## 📝 Development Conventions

- **Physics-Informed Methodology**: When modifying the model, ensure that the physics residual in `compute_physics_residual` correctly reflects the governing equations.
- **Scaling & Normalization**: Always use the `StateNormalizer` class for training stability. Time is typically normalized to mission duration, and distance to Earth's mean radius.
- **Testing**: New physics features or model architectures MUST be validated against the RK8 ground truth in `tests/test_orbital_mechanics.py`.
- **Modularity**: Keep physical constants in `config.py` and avoid hardcoding values in other modules.
- **Documentation**: Adhere to the scientific and technical standards established in `TECHNICAL_REPORT.md` and `PROJECT_SUMMARY.md`.

---

## 🛰️ Mission Data Reference

| Mission | Altitude | Dry Mass | Sail Area | Target Lifetime |
| :--- | :--- | :--- | :--- | :--- |
| **PSLV PS4** | 750 km | 920 kg | 10.0 m² | < 5 years |
| **SSLV VTM** | 500 km | 100 kg | 5.0 m² | < 5 years |
| **CARTOSAT-2** | 630 km | 680 kg | 8.0 m² | < 5 years |
