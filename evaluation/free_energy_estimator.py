import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Tuple

from models.realnvp import RealNVP
from potentials.double_well import DoubleWellPotential


def load_samples(file_path: str, batch_size: int = 128) -> DataLoader:
    """
    Load 1D samples from .npy file into a DataLoader.
    """
    data = np.load(file_path).astype(np.float32)
    data = torch.from_numpy(data).unsqueeze(1)  # Shape: (N, 1)
    return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=False)


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
        val = u_B(z.squeeze(1)) - log_det
        fwd_vals.extend(val.cpu().numpy())

    for (y_batch,) in data_B:
        y_batch = y_batch.to(device)
        x_inv, log_det_inv = model.inverse(y_batch)
        val = u_A(x_inv.squeeze(1)) + log_det_inv
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--state_a", type=str, default="data/state_a_samples.npy")
    parser.add_argument("--state_b", type=str, default="data/state_b_samples.npy")
    parser.add_argument("--model_path", type=str, default="realnvp_trained.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load samples
    dataloader_A = load_samples(args.state_a)
    dataloader_B = load_samples(args.state_b)

    # Load trained model
    model = RealNVP(n_coupling_layers=4, hidden_dim=64)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # Define potentials
    potential_A = DoubleWellPotential(a=1.0, b=1.0)
    potential_B = DoubleWellPotential(a=1.0, b=2.0)

    def u_A(x: torch.Tensor) -> torch.Tensor:
        return potential_A.energy(x)

    def u_B(x: torch.Tensor) -> torch.Tensor:
        return potential_B.energy(x)

    # Estimate ΔF
    (fwd_mean, fwd_ci), (rev_mean, rev_ci), (sym_mean, sym_ci) = estimate_delta_F_with_ci(
        model=model,
        u_A=u_A,
        u_B=u_B,
        data_A=dataloader_A,
        data_B=dataloader_B,
        device=args.device,
    )

    print(f"ΔF (A→B, forward KL)     = {fwd_mean:.4f}  (95% CI: {fwd_ci[0]:.4f} to {fwd_ci[1]:.4f})")
    print(f"ΔF (B→A, reverse KL)     = {rev_mean:.4f}  (95% CI: {rev_ci[0]:.4f} to {rev_ci[1]:.4f})")
    print(f"ΔF (symmetrized, BAR-like) = {sym_mean:.4f}  (95% CI: {sym_ci[0]:.4f} to {sym_ci[1]:.4f})")
