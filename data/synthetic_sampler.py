import os
import argparse
import numpy as np
from typing import Tuple
import json
import torch

# Import available potentials
from potentials.double_well import DoubleWellPotential
from potentials.lj import LennardJonesPotential
# from potentials.muller_brown import MullerBrownPotential  # Add when implemented

import matplotlib.pyplot as plt

def get_potential(name: str, dim: int = 1, **kwargs):
    """
    Factory method to construct a potential by name.
    Add new potentials here as needed.
    """
    if name == "double_well":
        if dim != 1:
            raise NotImplementedError("Double well currently only supports 1D.")
        return DoubleWellPotential(**kwargs)
    elif name == "lj":
        if dim % 3 != 0:
            raise NotImplementedError("Lennard-Jones potential currently only supports 3D.")
        return LennardJonesPotential(**kwargs)
    
    # elif name == "muller_brown":
    #     return MullerBrownPotential(dim=dim, **kwargs)

    else:
        raise ValueError(f"Unknown potential: {name}")

def generate_dataset(potential, n_samples: int = 5000, seed: int = 42, method: str = "metropolis") -> np.ndarray:
    """
    Generate samples from a given potential using the specified sampling method.
    """
    np.random.seed(seed)
    if method == "metropolis":
        return potential.metropolis_sample(n_samples=n_samples, seed=seed)
    elif method == "langevin":
        return potential.langevin_sample(n_samples=n_samples, seed=seed)
    else:
        raise ValueError(f"Unsupported sampling method: {method}")
    
def save_dataset(samples: np.ndarray, filename: str, output_dir: str = "data") -> None:
    """
    Save samples as a .npy file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    np.save(filepath, samples)
    print(f"Saved {len(samples)} samples to {filepath}")

def run_sampling(
    potential_name: str,
    potential_kwargs: dict,
    n_samples: int,
    seed: int,
    dim: int,
    out_file: str,
    method: str = "metropolis"
) -> None:
    """
    Full pipeline to construct a potential, sample from it, and save results.
    """
    potential = get_potential(potential_name, dim=dim, **potential_kwargs)

    print(f"Sampling {n_samples} samples from {potential_name} potential with parameters: {potential_kwargs}")
    samples = generate_dataset(potential, n_samples=n_samples, seed=seed, method=method)
    save_dataset(samples, filename=out_file)


def plot_sample_histogram(samples: np.ndarray, potential=None, title: str = "Sample Histogram", out_path: str = None, beta: float = 1.0) -> None:
    """
    Plot a histogram of 1D samples. Assumes samples are shape (N, 1) or (N,).
    Overlays true Boltzmann density and energy function if potential is provided.
    """
    if samples.ndim == 2 and samples.shape[1] == 1:
        samples = samples[:, 0]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.hist(samples, bins=100, density=True, alpha=0.7, color="tab:blue", label="Sampled Density")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Density", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    if potential is not None:
        x_plot = np.linspace(np.min(samples) - 1, np.max(samples) + 1, 500)
        x_tensor = torch.tensor(x_plot, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            u_x = potential.energy(x_tensor).squeeze().numpy()
        boltzmann = np.exp(-beta * u_x)
        boltzmann /= np.trapezoid(boltzmann, x_plot)  # normalize

        ax1.plot(x_plot, boltzmann, color="tab:red", label="True Density", lw=2)

        ax2 = ax1.twinx()
        ax2.plot(x_plot, u_x, color="tab:green", label="Energy", lw=2, linestyle="--")
        ax2.set_ylabel("Energy", color="tab:green")
        ax2.tick_params(axis='y', labelcolor="tab:green")

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    else:
        ax1.legend(loc="upper right")

    plt.title(title)
    ax1.grid(True)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"Saved histogram to {out_path}")
    else:
        plt.show()

if __name__ == "__main__":
    from potentials.double_well import DoubleWellPotential

    pot = DoubleWellPotential(a=1.0, b=2.0)

    # Generate samples using Metropolis sampling
    samples_metropolis = pot.metropolis_sample(n_samples=50000, step_size=0.1, seed=42)
    plot_sample_histogram(samples_metropolis, potential=pot, title="Metropolis Sampling: Double Well")
    
    samples_lan = pot.langevin_sample(n_samples=50000, step_size=0.1, seed=42)
    plot_sample_histogram(samples_lan, potential=pot, title="Langevin Sampling: Double Well")