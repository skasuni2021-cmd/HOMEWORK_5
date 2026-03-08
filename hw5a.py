import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def colebrook(f, Re, ed):
    """Implicit Colebrook equation for fsolve."""
    if f <= 0: return 1e6 # Prevent math domain errors
    return 1/np.sqrt(f) + 2.0 * np.log10(ed/3.7 + 2.51/(Re * np.sqrt(f)))

def get_friction_factor(Re, ed):
    """Calculates f based on the flow regime."""
    if Re <= 2000:
        return 64 / Re
    elif Re >= 4000:
        # Use 0.02 as a common starting guess for the solver
        f_guess = 0.02
        return fsolve(colebrook, f_guess, args=(Re, ed))[0]
    else:
        return None # Transition region handled by plotting logic