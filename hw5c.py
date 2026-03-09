import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Constants [cite: 107-115] ---
A = 4.909e-4  # m^2
V = 1.473e-4  # m^3
m = 30.0  # kg
beta = 2e9  # Pa (Bulk modulus)
ps = 1.4e7  # Pa (Supply pressure)
rho = 850.0  # kg/m^3 (Fluid density)
pa = 1e5  # Pa (Atmospheric pressure)
K_valve = 2e-5
y = 0.002  # Constant valve input


# --- State Variable Derivative Function  ---
def valve_system_derivatives(t, X):
    """
    Calculates derivatives for the state vector X:
    X[0] = x (displacement)
    X[1] = xdot (velocity)
    X[2] = p1 (pressure 1)
    X[3] = p2 (pressure 2)
    """
    x, xdot, p1, p2 = X

    # Differential Equations [cite: 94, 97, 98, 122]
    d_x = xdot
    d_xdot = (p1 - p2) * A / m
    d_p1 = (y * K_valve * (ps - p1) - rho * A * xdot) * beta / (V * rho)
    d_p2 = -(y * K_valve * (p2 - pa) - rho * A * xdot) * beta / (rho * V)

    return [d_x, d_xdot, d_p1, d_p2]


# --- Simulation Setup ---
# Initial conditions: x=0, xdot=0, p1=pa, p2=pa [cite: 119]
X0 = [0, 0, pa, pa]
t_span = (0, 0.05)  # Simulating for 0.05 seconds
t_eval = np.linspace(0, 0.05, 1000)

# Solve the ODEs
sol = solve_ivp(valve_system_derivatives, t_span, X0, t_eval=t_eval)

# --- Plotting Results [cite: 120, 121] ---

# Plot 1: Displacement (x) vs Time
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], 'b-', label='x (Displacement)')
plt.title('Piston Displacement vs Time')
plt.xlabel('Time (s)')
plt.ylabel('x (m)')
plt.grid(True)
plt.legend()
plt.show()

# Plot 2: Pressures (p1 and p2) vs Time
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[2], 'r-', label='p1')
plt.plot(sol.t, sol.y[3], 'g--', label='p2')
plt.title('Chamber Pressures vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (Pa)')
plt.grid(True)
plt.legend()
plt.show()