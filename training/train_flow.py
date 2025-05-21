import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Tuple

from models.realnvp import RealNVP
from potentials.double_well import DoubleWellPotential
from potentials.lj import LennardJonesPotential


def load_data(file_path: str, batch_size: int = 128) -> Tuple[DataLoader, int]:
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


def train_bidirectional_flow_model(
    model: RealNVP,
    u_A: Callable[[torch.Tensor], torch.Tensor],
    u_B: Callable[[torch.Tensor], torch.Tensor],
    dataloader_A: DataLoader,
    dataloader_B: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu"
) -> None:
    """
    Train RealNVP bidirectionally: forward (A→B) and reverse (B→A).
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0

        for (x_batch,), (y_batch,) in zip(dataloader_A, dataloader_B):
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

        epoch_loss /= len(dataloader_A.dataset)
        print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.6f}")
