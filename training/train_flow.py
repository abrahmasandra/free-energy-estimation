import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Callable, Tuple

from models.realnvp import RealNVP
from potentials.double_well import DoubleWellPotential
from potentials.lj import LennardJonesPotential
import matplotlib.pyplot as plt


def load_data(file_path: str, batch_size: int = 128, val_split=0.1, seed=0) -> Tuple[DataLoader, int]:
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

    # Split
    n_total = len(dataset)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val
    torch.manual_seed(seed)
    train_data, val_data = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, data.shape[1]


def train_bidirectional_flow_model(
    model: RealNVP,
    u_A: Callable[[torch.Tensor], torch.Tensor],
    u_B: Callable[[torch.Tensor], torch.Tensor],
    train_A: DataLoader,
    train_B: DataLoader,
    val_A: DataLoader,
    val_B: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu"
) -> None:
    """
    Train RealNVP bidirectionally: forward (A→B) and reverse (B→A).
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for (x_batch,), (y_batch,) in zip(train_A, train_B):
            x_batch = x_batch.to(device)  # From p_A
            y_batch = y_batch.to(device)  # From p_B

            # Forward: A → B
            z, log_det_f = model(x_batch)
    
            loss_fwd = torch.mean(u_B(z) - log_det_f)
            # loss_fwd = torch.mean(u_B(z.squeeze(1)) - log_det_f)
            # Reverse: B → A
            x_rev, log_det_inv = model.inverse(y_batch)
            loss_rev = torch.mean(u_A(x_rev) + log_det_inv)
            # loss_rev = torch.mean(u_A(x_rev.squeeze(1)) + log_det_inv)
            loss = loss_fwd + loss_rev

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)
        
        epoch_loss /= len(train_A.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch:3d} | Train Loss: {epoch_loss:.6f}")
        
        model.eval()
        with torch.no_grad():
            val_loss_fwd, val_loss_rev = 0.0, 0.0
            for (xa_batch,), (xb_batch,) in zip(val_A, val_B):
                xa_batch = xa_batch.to(device)
                xb_batch = xb_batch.to(device)

                z, log_det = model(xa_batch)
                loss_f = u_B(z) - log_det
                val_loss_fwd += loss_f.mean().item()

                z_inv, log_det_inv = model.inverse(xb_batch)
                loss_r = u_A(z_inv) + log_det_inv
                val_loss_rev += loss_r.mean().item()

            val_loss = 0.5 * (val_loss_fwd + val_loss_rev) / len(val_A)
            val_losses.append(val_loss)
            print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}")
    return train_losses, val_losses

def plot_loss_curves(train_losses, val_losses, title="Flow Training Loss", out_path=None):
    """
    Plot training and validation loss over epochs.
    
    Args:
        train_losses (list of float): Training loss per epoch.
        val_losses (list of float): Validation loss per epoch.
        title (str): Title for the plot.
        out_path (str): Optional path to save the plot (e.g., "loss_curve.png").
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"Saved loss curve to {out_path}")
    else:
        plt.show()
