# SMART-DEORBIT PINN Demo

Implementation of the **SMART-DEORBIT System** proposed for ISRO's PSLV and SSLV upper stage disposal.

This project demonstrates a **Physics-Informed Neural Network (PINN)** that learns orbital mechanics (J2 gravity + atmospheric drag) to rapidly predict orbital decay and optimize the minimum fuel (ΔV) required for 5-year compliance.

## Project Structure

- `app.py`: Main interactive dashboard (Streamlit).
- `pinn_model.py`: PyTorch PINN implementation with physics-based loss function.
- `orbital_mechanics.py`: Numerical propagator (RK8), gravity model, and exponential atmosphere model.
- `delta_v_optimizer.py`: Optimization logic to find minimum ΔV.
- `config.py`: Physical constants (Earth, J2, atmosphere) and mission parameters.

## Setup & Running

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Demo Dashboard:**
   ```bash
   streamlit run app.py
   ```
   Or simply:
   ```bash
   python run_demo.py
   ```

## Demo Workflow

1. **Select Mission**: Choose **PSLV PS4 (750 km)** or **SSLV VTM (500 km)** in the sidebar.
2. **Train PINN**: Go to the first tab and click **Start PINN Training**. Watch the loss curve converge as the network learns orbital physics.
3. **Verify Trajectory**: Switch to the second tab to compare the PINN's rapid prediction against the ground-truth numerical integrator (RK8).
4. **Optimize ΔV**: Go to the third tab and run the optimizer. It will find the sweet spot—minimum fuel burn + drag sail—to satisfy the 5-year de-orbit rule.

## Key Results (Expected)

- **PSLV PS4 (750 km)**: ~7.5 m/s ΔV with a 10 m² sail (vs >45 m/s without sail).
- **SSLV VTM (500 km)**: <1.0 m/s ΔV with a 5 m² sail.
- **PINN Speedup**: >100x faster than traditional Runge-Kutta numerical propagation.
