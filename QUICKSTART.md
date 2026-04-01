# SMART-DEORBIT Quickstart Guide

## 5-Minute Setup

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd E:\PINN

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Run the Interactive Dashboard

```bash
# Launch the Streamlit app
streamlit run app.py
```

This will open your web browser to `http://localhost:8501` with the interactive dashboard.

### Step 3: Try a Demo

1. **Select Mission**: Choose "PSLV PS4 (750 km)" from the sidebar
2. **Train PINN**: Click "🚀 Start PINN Training" in the first tab
3. **Wait**: Training takes 1-3 minutes
4. **Validate**: Switch to "📊 Validation" tab to see results
5. **Optimize**: Go to "🔥 ΔV Optimization" tab and run optimization

---

## Command-Line Demo

For automated demonstration:

```bash
# Run complete demo for PSLV mission
python run_demo.py --mission PSLV

# Run for SSLV mission
python run_demo.py --mission SSLV

# Specify output directory
python run_demo.py --mission PSLV --output-dir ./my_results
```

Results will be saved to `./outputs/` with:
- Training plots
- Trajectory comparisons
- Optimization results
- Technical reports

---

## Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=. --cov-report=html

# Open coverage report
open htmlcov/index.html  # On Windows: start htmlcov\index.html
```

---

## Expected Results

### PSLV PS4 @ 750 km

| Metric | Expected Value |
|--------|----------------|
| Optimal ΔV | ~7.5 m/s |
| Lifetime | <5 years |
| Fuel Savings | ~83% |
| PINN Speedup | >100x |

### SSLV VTM @ 500 km

| Metric | Expected Value |
|--------|----------------|
| Optimal ΔV | <1.0 m/s |
| Lifetime | <5 years |
| Fuel Savings | ~90% |
| PINN Speedup | >100x |

---

## Troubleshooting

### "Module not found" errors

```bash
# Ensure you're in the project directory
cd E:\PINN

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### CUDA/GPU issues

The system works on CPU by default. For GPU acceleration:

```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Streamlit port conflicts

```bash
# Run on different port
streamlit run app.py --server.port 8502
```

---

## Next Steps

1. **Explore Missions**: Try different mission configurations
2. **Modify Parameters**: Adjust sail area, mass, altitude
3. **Compare Methods**: Try different optimization algorithms
4. **Review Results**: Check the generated reports in `./outputs/`
5. **Read Documentation**: See `README.md` and `TECHNICAL_REPORT.md`

---

## Getting Help

- **Documentation**: See `README.md` for detailed usage
- **Technical Details**: See `TECHNICAL_REPORT.md`
- **Tests**: See `tests/test_orbital_mechanics.py` for examples
- **Issues**: Report bugs on GitHub

---

## Quick Reference

### Key Commands

```bash
# Dashboard
streamlit run app.py

# Demo
python run_demo.py --mission PSLV

# Tests
pytest tests/ -v

# Help
python run_demo.py --help
```

### Key Files

| File | Purpose |
|------|---------|
| `config.py` | Physical constants, mission configs |
| `orbital_mechanics.py` | Orbit propagation |
| `pinn_model.py` | Neural network implementation |
| `delta_v_optimizer.py` | ΔV optimization |
| `app.py` | Interactive dashboard |
| `run_demo.py` | Automated demo pipeline |

---

**Happy De-orbiting! 🛰️**
