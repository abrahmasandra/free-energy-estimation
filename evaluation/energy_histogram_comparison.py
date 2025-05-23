import numpy as np
import torch
import matplotlib.pyplot as plt
from models.realnvp import RealNVP

from scipy.stats import entropy
from scipy.stats import wasserstein_distance

@torch.no_grad()
def plot_energy_histograms(
    model: RealNVP,
    samples_A_path: str,
    samples_B_path: str,
    u_A: callable,
    u_B: callable,
    device: str = "cpu",
    bins: int = 60,
    n_points: int = 2000,
    out_path: str = None
):
    """
    Plot histograms of:
    - U_B(f(x_A)) vs U_B(x_B)
    - U_A(f⁻¹(x_B)) vs U_A(x_A)
    """
    # Load and flatten
    x_A = np.load(samples_A_path).astype(np.float32)
    x_B = np.load(samples_B_path).astype(np.float32)
    if x_A.ndim == 3: x_A = x_A.reshape(x_A.shape[0], -1)
    if x_B.ndim == 3: x_B = x_B.reshape(x_B.shape[0], -1)

    x_A = x_A[:n_points]
    x_B = x_B[:n_points]

    x_A_t = torch.from_numpy(x_A).to(device)
    x_B_t = torch.from_numpy(x_B).to(device)

    # Flow transformations
    z_fwd, _ = model(x_A_t)        # f(x_A)
    z_rev, _ = model.inverse(x_B_t)  # f⁻¹(x_B)

    # Compute energies
    uB_fxA = u_B(z_fwd).cpu().numpy()
    uB_xB = u_B(x_B_t).cpu().numpy()
    uA_fxB = u_A(z_rev).cpu().numpy()
    uA_xA = u_A(x_A_t).cpu().numpy()

    metrics_fwd = compute_histogram_metrics(uB_fxA, uB_xB)
    metrics_rev = compute_histogram_metrics(uA_fxB, uA_xA)

    print(f"\nForward energy comparison (U_B):")
    print(f"  KL Divergence:     {metrics_fwd['kl']:.4f}")
    print(f"  Wasserstein Dist.: {metrics_fwd['wasserstein']:.4f}")

    print(f"\nReverse energy comparison (U_A):")
    print(f"  KL Divergence:     {metrics_rev['kl']:.4f}")
    print(f"  Wasserstein Dist.: {metrics_rev['wasserstein']:.4f}")

    # Plot forward comparison
    plt.figure(figsize=(8, 4.5))
    plt.hist(uB_fxA, bins=bins, alpha=0.5, label="U_B(f(x_A))", density=True)
    plt.hist(uB_xB,  bins=bins, alpha=0.5, label="U_B(x_B)", density=True)
    plt.title("Energy Histogram: U_B")
    plt.xlabel("Energy")
    plt.ylabel("Density")
    plt.legend()
    if out_path:
        plt.savefig(out_path.replace(".png", "_ub.png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()

    # Plot reverse comparison
    plt.figure(figsize=(8, 4.5))
    plt.hist(uA_fxB, bins=bins, alpha=0.5, label="U_A(f⁻¹(x_B))", density=True)
    plt.hist(uA_xA,  bins=bins, alpha=0.5, label="U_A(x_A)", density=True)
    plt.title("Energy Histogram: U_A")
    plt.xlabel("Energy")
    plt.ylabel("Density")
    plt.legend()
    if out_path:
        plt.savefig(out_path.replace(".png", "_ua.png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()

##### COMPUTE STATISTICS #####

def compute_histogram_metrics(p, q, bins=60) -> dict:
    """
    Compute KL divergence and Wasserstein distance between two 1D distributions.
    Returns:
        dict with 'kl' and 'wasserstein'
    """
    # Create shared histogram bins
    hist_p, bin_edges = np.histogram(p, bins=bins, density=True)
    hist_q, _ = np.histogram(q, bins=bin_edges, density=True)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-12
    hist_p += epsilon
    hist_q += epsilon

    # Normalize histograms to sum to 1
    hist_p /= hist_p.sum()
    hist_q /= hist_q.sum()

    kl = entropy(hist_p, hist_q)
    wd = wasserstein_distance(p, q)
    return {"kl": kl, "wasserstein": wd}
