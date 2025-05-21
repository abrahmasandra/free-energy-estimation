import numpy as np
from scipy.integrate import quad
from potentials.double_well import DoubleWellPotential


def compute_partition_function(potential, bounds=(-10, 10)):
    """
    Numerically compute the partition function Z = ∫ e^{-U(x)} dx
    """
    def boltzmann(x):
        return np.exp(-potential.energy(np.array([x])).item())  # Ensure scalar

    Z, _ = quad(boltzmann, bounds[0], bounds[1], limit=500)
    return Z


def compute_delta_F_true(potential_A, potential_B, bounds=(-10, 10)):
    """
    Compute ΔF = -log(Z_B / Z_A)
    """
    Z_A = compute_partition_function(potential_A, bounds)
    Z_B = compute_partition_function(potential_B, bounds)
    delta_F = -np.log(Z_B / Z_A)
    return delta_F
