"""
SMART-DEORBIT SYSTEM — Interactive Demo Dashboard
===================================================
Main Streamlit application for the PINN demo.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch

from config import (
    get_mission_config, circular_orbit_state, R_EARTH,
    PINN_HPARAMS
)
from orbital_mechanics import (
    compute_lifetime, propagate_orbit, apply_delta_v
)
from pinn_model import (
    train_pinn, predict_trajectory, predict_lifetime_pinn
)
from delta_v_optimizer import optimize_delta_v, compute_propulsive_only_dv

# Page Config
st.set_page_config(
    page_title="SMART-DEORBIT Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1, h2, h3 {
        color: #0d47a1;
    }
    .stButton>button {
        width: 100%;
        background-color: #0d47a1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/ISRO_Logo.svg/1200px-ISRO_Logo.svg.png", width=100)
st.sidebar.title("Configuration")

# Mission Selection
mission_name = st.sidebar.selectbox(
    "Select Mission Scenario",
    ["PSLV PS4 (750 km)", "SSLV VTM (500 km)"]
)

if "PSLV" in mission_name:
    base_config = get_mission_config("PSLV")
else:
    base_config = get_mission_config("SSLV")

st.sidebar.markdown("---")
st.sidebar.subheader("Mission Parameters")

# Editable parameters
orbit_alt_km = st.sidebar.number_input(
    "Orbit Altitude (km)",
    value=float(base_config["orbit_altitude"] / 1000),
    step=10.0
)
mass_kg = st.sidebar.number_input(
    "Dry Mass (kg)",
    value=float(base_config["dry_mass"]),
    step=10.0
)
sail_area_m2 = st.sidebar.number_input(
    "Drag Sail Area (m²)",
    value=float(base_config["drag_sail_area"]),
    step=0.5
)
Cd = st.sidebar.number_input(
    "Drag Coefficient",
    value=float(base_config["Cd"]),
    step=0.1
)

# Update config dictionary
current_config = base_config.copy()
current_config["orbit_altitude"] = orbit_alt_km * 1000.0
current_config["dry_mass"] = mass_kg
current_config["drag_sail_area"] = sail_area_m2
current_config["Cd"] = Cd


# ─────────────────────────────────────────────
# Main Page
# ─────────────────────────────────────────────
st.title("🚀 SMART-DEORBIT SYSTEM")
st.markdown(f"**Current Mission:** {mission_name} | **Target:** Re-entry within 5 years")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🧠 PINN Training & Validation",
    "📉 Trajectory Prediction",
    "🔥 ΔV Optimization"
])


# ─────────────────────────────────────────────
# Tab 1: PINN Training
# ─────────────────────────────────────────────
with tab1:
    st.header("Physics-Informed Neural Network Training")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info(
            """
            **Model Architecture:**
            - Input: Time (t)
            - Hidden Layers: 4 x 128 (Tanh)
            - Output: State [x, y, z, vx, vy, vz]
            
            **Loss Function:**
            - Data Loss: MSE(t=0)
            - Physics Loss: ODE Residual (Gravity + Drag)
            """
        )
        train_btn = st.button("Start PINN Training", key="train_btn")
    
    with col2:
        loss_chart_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    if train_btn:
        st.session_state["pinn_trained"] = False
        
        # Initial State
        state0 = circular_orbit_state(current_config["orbit_altitude"])
        
        # Callback for real-time plotting
        loss_history = {'total': [], 'data': [], 'physics': [], 'epoch': []}
        
        def update_plot(epoch, losses):
            loss_history['epoch'].append(epoch)
            loss_history['total'].append(losses['total'])
            loss_history['data'].append(losses['data'])
            loss_history['physics'].append(losses['physics'])
            
            # Update progress
            progress = (epoch + 1) / PINN_HPARAMS["epochs"]
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Epoch {epoch}/{PINN_HPARAMS['epochs']} | Loss: {losses['total']:.2e}")
            
            # Update chart every 500 epochs to avoid lag
            if epoch % 500 == 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.semilogy(loss_history['epoch'], loss_history['total'], label='Total Loss')
                ax.semilogy(loss_history['epoch'], loss_history['physics'], label='Physics Loss', linestyle='--')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss (Log Scale)')
                ax.legend()
                ax.grid(True, which="both", ls="-", alpha=0.5)
                loss_chart_placeholder.pyplot(fig)
                plt.close(fig)

        # Run Training
        with st.spinner("Training PINN... (Physics-Informed Loss Optimization)"):
            model, normalizer, history, rk8_result, train_time = train_pinn(
                state0,
                Cd=current_config["Cd"],
                A=current_config["drag_sail_area"],
                mass=current_config["dry_mass"],
                t_span_years=8.0,  # Train for slightly longer than 5 years
                epochs=PINN_HPARAMS["epochs"],
                callback=update_plot
            )
        
        # Save to session state
        st.session_state["model"] = model
        st.session_state["normalizer"] = normalizer
        st.session_state["rk8_result"] = rk8_result
        st.session_state["pinn_trained"] = True
        st.session_state["train_time"] = train_time
        
        status_text.success(f"Training Complete in {train_time:.2f} seconds!")
        
        # Final Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(loss_history['epoch'], loss_history['total'], label='Total Loss')
        ax.semilogy(loss_history['epoch'], loss_history['physics'], label='Physics Loss', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (Log Scale)')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.5)
        loss_chart_placeholder.pyplot(fig)


# ─────────────────────────────────────────────
# Tab 2: Trajectory Prediction
# ─────────────────────────────────────────────
with tab2:
    st.header("Trajectory Prediction: PINN vs Numerical Integrator (RK8)")
    
    if "pinn_trained" not in st.session_state or not st.session_state["pinn_trained"]:
        st.warning("Please train the PINN model in Tab 1 first.")
    else:
        model = st.session_state["model"]
        normalizer = st.session_state["normalizer"]
        rk8_result = st.session_state["rk8_result"]
        
        # Make PINN Prediction
        t_eval = rk8_result['t_years']
        pinn_states, pinn_time_ms = predict_trajectory(model, normalizer, t_eval)
        
        # Calculate Errors
        pinn_r = np.linalg.norm(pinn_states[:, :3], axis=1)
        rk8_r = np.linalg.norm(rk8_result['states'][:, :3], axis=1)
        
        pinn_alt_km = (pinn_r - R_EARTH) / 1000.0
        rk8_alt_km = rk8_result['altitudes_km']
        
        error_km = np.abs(pinn_alt_km - rk8_alt_km)
        mean_error = np.mean(error_km)
        max_error = np.max(error_km)
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PINN Inference Time", f"{pinn_time_ms:.1f} ms")
        m2.metric("RK8 Compute Time", f"{rk8_result['compute_time_s']*1000:.1f} ms")
        m3.metric("Speedup Factor", f"{int(rk8_result['compute_time_s']*1000 / pinn_time_ms)}x")
        m4.metric("Mean Altitude Error", f"{mean_error:.3f} km")
        
        # Plots
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.subheader("Altitude Decay")
            fig1, ax1 = plt.subplots()
            ax1.plot(t_eval, rk8_alt_km, 'k-', label='RK8 (Ground Truth)', linewidth=2, alpha=0.6)
            ax1.plot(t_eval, pinn_alt_km, 'r--', label='PINN Prediction', linewidth=2)
            ax1.set_xlabel("Time (Years)")
            ax1.set_ylabel("Altitude (km)")
            ax1.legend()
            ax1.grid(True)
            st.pyplot(fig1)
            
        with col_p2:
            st.subheader("Prediction Error")
            fig2, ax2 = plt.subplots()
            ax2.plot(t_eval, error_km, 'b-')
            ax2.set_xlabel("Time (Years)")
            ax2.set_ylabel("Abs. Altitude Error (km)")
            ax2.grid(True)
            st.pyplot(fig2)


# ─────────────────────────────────────────────
# Tab 3: ΔV Optimization
# ─────────────────────────────────────────────
with tab3:
    st.header("Optimal De-Orbit Strategy")
    st.markdown("Find the minimum retrograde ΔV burn required to de-orbit within **5 years**.")
    
    if st.button("Run ΔV Optimization"):
        with st.spinner("Running optimization loop..."):
            state0 = circular_orbit_state(current_config["orbit_altitude"])
            
            # Callback to show progress
            progress_ph = st.empty()
            
            # Scrape optimization range based on mission
            if "PSLV" in mission_name:
                dv_rng = (0, 40)
                dv_stp = 2.0
            else:
                dv_rng = (0, 10)
                dv_stp = 0.5
            
            # Run Optimization
            opt_res = optimize_delta_v(
                state0,
                Cd=current_config["Cd"],
                A=current_config["drag_sail_area"],
                mass=current_config["dry_mass"],
                dv_range=dv_rng,
                dv_step=dv_stp,
                target_lifetime=5.0
            )
            
            # Run Propulsive Comparison (no sail)
            prop_dv = compute_propulsive_only_dv(
                state0, current_config["dry_mass"],
                cross_section=base_config["cross_section_no_sail"]
            )
            
            st.success("Optimization Complete!")
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Optimal ΔV (with Sail)", f"{opt_res['optimal_dv']:.2f} m/s")
            c2.metric("Predicted Lifetime", f"{opt_res['lifetime_at_optimal']:.1f} years")
            c3.metric("Fuel Savings vs No-Sail", f"{(1 - opt_res['optimal_dv']/prop_dv)*100:.1f}%")
            
            # Plot ΔV vs Lifetime Curve
            st.subheader("ΔV vs. Orbital Lifetime Curve")
            
            # Create interactive plot with Plotly
            fig = go.Figure()
            
            # Initial points
            fig.add_trace(go.Scatter(
                x=opt_res['dv_values'], 
                y=opt_res['lifetimes'],
                mode='lines+markers',
                name='Simulated Points'
            ))
            
            # Optimal Point
            fig.add_trace(go.Scatter(
                x=[opt_res['optimal_dv']],
                y=[opt_res['lifetime_at_optimal']],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name='Optimal Solution'
            ))
            
            # 5-Year Threshold Line
            fig.add_shape(
                type="line",
                x0=min(opt_res['dv_values']), x1=max(opt_res['dv_values']),
                y0=5, y1=5,
                line=dict(color="green", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=max(opt_res['dv_values'])-5, y=5.5,
                text="5-Year Compliance Limit",
                showarrow=False,
                font=dict(color="green")
            )
            
            fig.update_layout(
                title="Optimization Landscape",
                xaxis_title="Retrograde ΔV (m/s)",
                yaxis_title="Orbital Lifetime (Years)",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Final Recommendation Text
            st.markdown(f"""
            ### 📝 Recommendation for {mission_name}
            To meet the **5-year de-orbit requirement**, perform a retrograde burn of **{opt_res['optimal_dv']:.2f} m/s** immediately after mission completion, then deploy the **{current_config['drag_sail_area']} m²** drag sail.
            
            Without the sail, you would need a burn of **{prop_dv:.1f} m/s**, making the SMART-DEORBIT system **{prop_dv/opt_res['optimal_dv']:.1f}x more fuel efficient**.
            """)

st.sidebar.markdown("---")
st.sidebar.info("Developed for ISRO Debris-Free Space Missions 2030 Initiative")
