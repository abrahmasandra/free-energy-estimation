import os
import argparse
import numpy as np
from typing import Tuple

# Import available potentials
from potentials.double_well import DoubleWellPotential
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
    samples = generate_dataset(potential, n_samples=n_samples, seed=seed, method=method)
    save_dataset(samples, filename=out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic dataset sampler")
    parser.add_argument("--potential", type=str, required=True, help="Name of the potential (e.g., double_well)")
    parser.add_argument("--dim", type=int, default=1, help="Dimensionality of system")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, required=True, help="Output .npy filename")
    parser.add_argument("--method", type=str, default="metropolis", help="Sampling method")
    parser.add_argument("--potential_kwargs", type=str, default="{}", help="Extra kwargs for potential as JSON string")

    args = parser.parse_args()

    import json
    kwargs = json.loads(args.potential_kwargs)

    run_sampling(
        potential_name=args.potential,
        potential_kwargs=kwargs,
        n_samples=args.n_samples,
        seed=args.seed,
        dim=args.dim,
        out_file=args.output,
        method=args.method,
    )