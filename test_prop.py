
import numpy as np
import time
from config import SSLV_CONFIG, circular_orbit_state
from orbital_mechanics import propagate_orbit

state0 = circular_orbit_state(SSLV_CONFIG.orbit_altitude)
start = time.time()
print("Starting propagation...")
result = propagate_orbit(
    state0, 5.0, SSLV_CONFIG.Cd, SSLV_CONFIG.drag_sail_area, SSLV_CONFIG.dry_mass,
    dt_output_days=1.0, max_step_days=0.5
)
end = time.time()
print(f"Propagation success: {result.success}")
print(f"Time taken: {end - start:.2f}s")
print(f"Lifetime: {result.lifetime_years:.2f} years")
