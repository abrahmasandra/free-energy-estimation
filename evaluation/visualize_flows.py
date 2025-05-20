import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Callable
from models.realnvp import RealNVP


@torch.no_grad()
def visualize_flow_mapping(
    model: RealNVP,
    samples_A_path: str,
    samples_B_path: str,
    out_path: str = None,
    bins: int = 100,
    device: str = "cpu",
    potential_fn: Callable[[np.ndarray], np.ndarray] = None,
    potential_label: str = "U_B(x)"
) -> None:
    """
    Compare p_A, f(p_A), p_B and optionally overlay U_B(x)
    """
    x_A = np.load(samples_A_path).astype(np.float32)
    x_B = np.load(samples_B_path).astype(np.float32)
    x_A_tensor = torch.from_numpy(x_A).unsqueeze(1).to(device)

    model.eval()
    model = model.to(device)
    z_mapped, _ = model(x_A_tensor)
    z_mapped = z_mapped.cpu().numpy().squeeze()

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.hist(x_A.squeeze(), bins=bins, density=True, alpha=0.5, label="State A (original)")
    ax1.hist(z_mapped, bins=bins, density=True, alpha=0.5, label="Mapped A → B")
    ax1.hist(x_B.squeeze(), bins=bins, density=True, alpha=0.5, label="State B (target)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper left")

    if potential_fn:
        xs = np.linspace(-3, 3, 300)
        U_vals = potential_fn(xs)
        ax2 = ax1.twinx()
        ax2.plot(xs, U_vals, 'k--', label=potential_label)
        ax2.set_ylabel("Potential energy")
        ax2.legend(loc="upper right")

    plt.title("Forward Flow A → B")
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

@torch.no_grad()
def visualize_inverse_flow_mapping(
    model: RealNVP,
    samples_A_path: str,
    samples_B_path: str,
    out_path: str = None,
    bins: int = 100,
    device: str = "cpu",
    potential_fn: Callable[[np.ndarray], np.ndarray] = None,
    potential_label: str = "U_A(x)"
) -> None:
    """
    Compare p_A vs f⁻¹(p_B) and optionally overlay U_A(x)
    """
    x_A = np.load(samples_A_path).astype(np.float32)
    x_B = np.load(samples_B_path).astype(np.float32)
    x_B_tensor = torch.from_numpy(x_B).unsqueeze(1).to(device)

    model.eval()
    model = model.to(device)
    x_recon, _ = model.inverse(x_B_tensor)
    x_recon = x_recon.cpu().numpy().squeeze()

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.hist(x_B.squeeze(), bins=bins, density=True, alpha=0.5, label="State B (original)")
    ax1.hist(x_recon, bins=bins, density=True, alpha=0.5, label="Mapped B → A")
    ax1.hist(x_A.squeeze(), bins=bins, density=True, alpha=0.5, label="State A (target)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper left")

    if potential_fn:
        xs = np.linspace(-3, 3, 300)
        U_vals = potential_fn(xs)
        ax2 = ax1.twinx()
        ax2.plot(xs, U_vals, 'k--', label=potential_label)
        ax2.set_ylabel("Potential energy")
        ax2.legend(loc="upper right")

    plt.title("Inverse Flow B → A")
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


# if __name__ == "__main__":
#     model = RealNVP(n_coupling_layers=4, hidden_dim=64)
#     model.load_state_dict(torch.load("realnvp_trained.pt", map_location="cpu"))

#     visualize_flow_mapping(
#         model=model,
#         samples_A_path="data/state_a_samples.npy",
#         samples_B_path="data/state_b_samples.npy"
#     )

#     visualize_inverse_flow_mapping(
#         model=model,
#         samples_A_path="data/state_a_samples.npy",
#         samples_B_path="data/state_b_samples.npy",
#     )
