import os
import argparse
import numpy as np
from typing import Tuple
import json

# Import available potentials
from potentials.double_well import DoubleWellPotential
from potentials.lj import LennardJonesPotential
# from potentials.muller_brown import MullerBrownPotential  # Add when implemented


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
        return potential.metropolis_sample(n_samples=n_samples)
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