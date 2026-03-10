# SMART-DEORBIT System - Technology Stack

## Core Language & Runtime
- **Python 3.12+**: The primary programming language, selected for its rich ecosystem in scientific computing and AI.
- **uv**: The chosen package manager for high-performance dependency resolution and environment management.

## Machine Learning & Physics
- **PyTorch**: Used for building and training Physics-Informed Neural Networks (PINNs). It provides the `autograd` engine necessary for computing physics residuals ($d^2r/dt^2$).
- **SciPy**: Specifically the `DOP853` / RK8 high-fidelity numerical integrator used for ground-truth generation and orbit propagation.
- **NumPy**: The foundation for all numerical arrays and linear algebra operations within the orbital mechanics engine.

## Interface & Visualization
- **Streamlit**: Powers the interactive dashboard, providing a seamless bridge between Python scripts and the user-facing GUI.
- **Plotly**: Used for creating interactive, web-based 3D orbital trajectories and multi-objective Pareto front visualizations.
- **Matplotlib**: Leveraged for high-quality, publication-ready static plots in automated reports.

## Testing & Quality Assurance
- **pytest**: The primary testing framework for unit testing physical constants, orbital conversions, and PINN model integrity.