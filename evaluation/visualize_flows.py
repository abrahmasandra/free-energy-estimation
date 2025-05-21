import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Callable
from models.realnvp import RealNVP
from sklearn.decomposition import PCA

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

    print(f"x_A.shape: {x_A.shape}")
    print(f"x_B.shape: {x_B.shape}")
    x_A_tensor = torch.from_numpy(x_A)

    model.eval()
    model = model.to(device)
    z_mapped, _ = model(x_A_tensor)
    z_mapped = z_mapped.cpu().numpy().squeeze()
    print(f"z_mapped.shape: {z_mapped.shape}")

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
def visualize_pca_mapped_vs_target(
    model: RealNVP,
    samples_A_path: str,
    samples_B_path: str,
    device: str = "cpu",
    out_path: str = None,
    n_points: int = 2000,
) -> None:
    """
    Perform PCA and plot:
    - Original A samples
    - Mapped samples from A → B
    - True samples from B
    """
    # Load and flatten
    x_A = np.load(samples_A_path).astype(np.float32)
    x_B = np.load(samples_B_path).astype(np.float32)

    if x_A.ndim == 3: x_A = x_A.reshape(x_A.shape[0], -1)
    if x_B.ndim == 3: x_B = x_B.reshape(x_B.shape[0], -1)

    x_A = x_A[:n_points]
    x_B = x_B[:n_points]

    x_A_tensor = torch.from_numpy(x_A).to(device)

    # Apply flow: A → B
    model.eval()
    model = model.to(device)
    z_mapped, _ = model(x_A_tensor)
    z_mapped = z_mapped.cpu().numpy()

    # Combine and PCA
    all_data = np.vstack([x_A, z_mapped, x_B])
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_data)

    reduced_A      = reduced[:n_points]
    reduced_mapped = reduced[n_points:2*n_points]
    reduced_B      = reduced[2*n_points:]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_A[:, 0], reduced_A[:, 1], alpha=0.4, label="State A (original)")
    plt.scatter(reduced_mapped[:, 0], reduced_mapped[:, 1], alpha=0.4, label="Mapped A → B")
    plt.scatter(reduced_B[:, 0], reduced_B[:, 1], alpha=0.4, label="State B (true)")

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("PCA Projection: A, A→B (mapped), B")
    plt.legend()

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved PCA plot to {out_path}")
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
    x_B_tensor = torch.from_numpy(x_B)

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


@torch.no_grad()
def visualize_pca_inverse_mapping(
    model: RealNVP,
    samples_A_path: str,
    samples_B_path: str,
    device: str = "cpu",
    out_path: str = None,
    n_points: int = 2000,
) -> None:
    """
    Perform PCA and plot:
    - State B samples (true)
    - Mapped samples from B → A via inverse flow
    - State A samples (true)
    """
    # Load and flatten
    x_A = np.load(samples_A_path).astype(np.float32)
    x_B = np.load(samples_B_path).astype(np.float32)

    if x_A.ndim == 3: x_A = x_A.reshape(x_A.shape[0], -1)
    if x_B.ndim == 3: x_B = x_B.reshape(x_B.shape[0], -1)

    x_A = x_A[:n_points]
    x_B = x_B[:n_points]

    x_B_tensor = torch.from_numpy(x_B).to(device)

    # Apply inverse flow: B → A
    model.eval()
    model = model.to(device)
    x_mapped, _ = model.inverse(x_B_tensor)
    x_mapped = x_mapped.cpu().numpy()

    # Combine and PCA
    all_data = np.vstack([x_B, x_mapped, x_A])
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_data)

    reduced_B      = reduced[:n_points]
    reduced_mapped = reduced[n_points:2*n_points]
    reduced_A      = reduced[2*n_points:]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_B[:, 0], reduced_B[:, 1], alpha=0.4, label="State B (true)")
    plt.scatter(reduced_mapped[:, 0], reduced_mapped[:, 1], alpha=0.4, label="Mapped B → A")
    plt.scatter(reduced_A[:, 0], reduced_A[:, 1], alpha=0.4, label="State A (true)")

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("PCA Projection: B, B→A (mapped), A")
    plt.legend()

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved inverse PCA plot to {out_path}")
    else:
        plt.show()
