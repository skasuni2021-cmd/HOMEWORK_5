import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# --- Engineering Constants ---
NU = 1.08e-5  # Kinematic viscosity of water (ft^2/s)
G = 32.2  # Acceleration of gravity (ft/s^2)


def get_colebrook_f(Re, ed):
    """Solves the implicit Colebrook equation for turbulent flow[cite: 11, 12]."""
    func = lambda f: 1 / np.sqrt(f) + 2.0 * np.log10(ed / 3.7 + 2.51 / (Re * np.sqrt(f)))
    # Initial guess of 0.02 is standard for pipe flow
    return fsolve(func, 0.02)[0]


def calculate_flow_params(d_in, eps_mics, flow_gpm):
    """
    Handles unit conversions and determines friction factor f.
    Returns: Re, f, hf_L, and marker_type
    """
    # 1. Unit Conversions to English Units (feet and seconds)
    d_ft = d_in / 12.0
    area = (np.pi * d_ft ** 2) / 4.0
    # Convert GPM to cubic feet per second (ft^3/s)
    velocity = (flow_gpm * 0.002228) / area
    # Relative roughness (epsilon / diameter)
    ed = (eps_mics * 1e-6) / d_in

    # 2. Calculate Reynolds Number
    Re = (velocity * d_ft) / NU

    # 3. Determine Friction Factor f based on Regime [cite: 10, 60]
    if Re < 2000:
        f = 64 / Re
        marker = 'o'  # Circle for laminar
    elif Re > 4000:
        f = get_colebrook_f(Re, ed)
        marker = 'o'  # Circle for turbulent
    else:
        # Stochastic Transition Logic (2000 < Re < 4000) [cite: 60, 61, 62]
        f_lam = 64 / 2000
        f_cb = get_colebrook_f(4000, ed)
        # Calculate Mean (mu) and Standard Deviation (sigma)
        mu_f = f_lam + (f_cb - f_lam) * (Re - 2000) / 2000
        sigma_f = 0.2 * mu_f
        # Pull a random value from the normal distribution [cite: 60]
        f = np.random.normal(mu_f, sigma_f)
        marker = '^'  # Upward triangle for transition

    # 4. Calculate Head Loss per Foot (hf/L) [cite: 7, 57]
    # hf/L = f * (1/D) * (V^2 / 2g)
    hf_L = f * (1 / d_ft) * (velocity ** 2 / (2 * G))

    return Re, f, hf_L, marker


def main():
    # Setup the plot background (Simplified Moody background for this example)
    plt.ion()  # Enable interactive mode [cite: 59]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Reynolds number $Re$')
    ax.set_ylabel(r'Friction factor $f$')
    ax.grid(True, which='both', alpha=0.5)

    while True:
        print("\n--- Pipe Flow Input ---")
        try:
            # Solicit inputs from the user
            d = float(input("Enter pipe diameter (inches): "))
            eps = float(input("Enter pipe roughness (micro-inches): "))
            q = float(input("Enter flow rate (gallons/min): "))

            # Calculate values
            Re, f, hf_L, marker = calculate_flow_params(d, eps, q)

            # Return results to user
            print(f"\n--- Results ---")
            print(f"Reynolds Number (Re): {Re:.2f}")
            print(f"Friction Factor (f): {f:.5f}")
            print(f"Head Loss per Foot (hf/L): {hf_L:.6f} ft/ft")

            # Add icon to the existing Moody diagram [cite: 58, 59]
            ax.plot(Re, f, marker, markersize=10, markeredgecolor='red', markerfacecolor='none')
            plt.draw()
            plt.pause(0.1)

            # Allow user to re-specify parameters [cite: 59]
            if input("\nAdd another set of parameters? (y/n): ").lower() != 'y':
                break
        except ValueError:
            print("Invalid input. Please enter numeric values.")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()