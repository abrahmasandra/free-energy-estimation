import numpy as np
import torch
from torch.utils.data import DataLoader
from pymbar import MBAR
from potentials.lj import LennardJonesPotential
from typing import Tuple, Callable


def compute_delta_F_MBAR(
    data_A: DataLoader,
    data_B: DataLoader,
    u_A: Callable,
    u_B: Callable,
    beta: float = 1.0,
    device: str = "cpu",
) -> float:
    """
    Compute ΔF using MBAR with reduced potentials at two states.
    Returns:
        ΔF = F_B - F_A (float)
    """
    u_A_A, u_B_A = [], []
    u_A_B, u_B_B = [], []

    for (x_batch,) in data_A:
        x_batch = x_batch.to(device)
        u_A_A.append(beta * u_A(x_batch).cpu().numpy())
        u_B_A.append(beta * u_B(x_batch).cpu().numpy())

    for (y_batch,) in data_B:
        y_batch = y_batch.to(device)
        u_A_B.append(beta * u_A(y_batch).cpu().numpy())
        u_B_B.append(beta * u_B(y_batch).cpu().numpy())

    # Flatten
    u_A_A = np.concatenate(u_A_A)
    u_B_A = np.concatenate(u_B_A)
    u_A_B = np.concatenate(u_A_B)
    u_B_B = np.concatenate(u_B_B)

    # Build u_kn matrix: shape (2, N_total)
    u_kn = np.vstack([
        np.concatenate([u_A_A, u_A_B]),  # row 0: u_A(x)
        np.concatenate([u_B_A, u_B_B]),  # row 1: u_B(x)
    ])

    # Sample counts
    N_k = np.array([len(u_A_A), len(u_B_B)])

    # Run MBAR
    mbar = MBAR(u_kn, N_k, verbose=False)
    results = mbar.compute_free_energy_differences()
    delta_F = results["Delta_f"][0, 1]

    return delta_F
