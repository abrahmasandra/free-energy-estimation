import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Tuple

from models.realnvp import RealNVP
from potentials.double_well import DoubleWellPotential
from potentials.lj import LennardJonesPotential

def load_samples(file_path: str, batch_size: int = 128) -> Tuple[DataLoader, int]:
    """
    Load numpy samples and wrap in DataLoader.
    """
    data = np.load(file_path).astype(np.float32)
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)  # (N, n_particles, 3) → (N, dim)
    dim = data.shape[1]
    data = torch.from_numpy(data)
    # data = torch.from_numpy(data).unsqueeze(1)  # Shape: (N, 1)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), dim


def bootstrap_ci(values: list[float], n_bootstrap: int = 200, ci: float = 0.95) -> Tuple[float, float]:
    """
    Compute percentile bootstrap confidence interval.
    """
    means = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


@torch.no_grad()
def estimate_delta_F_with_ci(
    model: RealNVP,
    u_A: Callable[[torch.Tensor], torch.Tensor],
    u_B: Callable[[torch.Tensor], torch.Tensor],
    data_A: DataLoader,
    data_B: DataLoader,
    device: str = "cpu",
    n_bootstrap: int = 200
) -> Tuple[Tuple[float, float, Tuple[float, float]],
           Tuple[float, float, Tuple[float, float]],
           Tuple[float, float, Tuple[float, float]]]:
    """
    Estimate ΔF and 95% confidence intervals.
    Returns:
        (mean_fwd, mean_rev, mean_sym), each with confidence interval tuple
    """
    model = model.to(device)
    model.eval()

    fwd_vals, rev_vals = [], []

    for (x_batch,) in data_A:
        x_batch = x_batch.to(device)
        z, log_det = model(x_batch)
        val = u_B(z) - log_det
        fwd_vals.extend(val.cpu().numpy())

    for (y_batch,) in data_B:
        y_batch = y_batch.to(device)
        x_inv, log_det_inv = model.inverse(y_batch)
        val = u_A(x_inv) + log_det_inv
        rev_vals.extend(val.cpu().numpy())

    fwd_vals = np.array(fwd_vals)
    rev_vals = np.array(rev_vals)

    fwd_mean = float(np.mean(fwd_vals))
    rev_mean = float(np.mean(rev_vals))
    sym_mean = 0.5 * (fwd_mean + rev_mean)

    fwd_ci = bootstrap_ci(fwd_vals.tolist(), n_bootstrap=n_bootstrap)
    rev_ci = bootstrap_ci(rev_vals.tolist(), n_bootstrap=n_bootstrap)
    sym_ci = (
        0.5 * (fwd_ci[0] + rev_ci[0]),
        0.5 * (fwd_ci[1] + rev_ci[1])
    )

    return (fwd_mean, fwd_ci), (rev_mean, rev_ci), (sym_mean, sym_ci)
