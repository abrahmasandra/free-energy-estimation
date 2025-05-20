import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable

from models.realnvp import RealNVP
from potentials.double_well import DoubleWellPotential


def load_data(file_path: str, batch_size: int = 128) -> DataLoader:
    """
    Load numpy samples and wrap in DataLoader.
    """
    data = np.load(file_path).astype(np.float32)
    data = torch.from_numpy(data).unsqueeze(1)  # Shape: (N, 1)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


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
            loss_fwd = torch.mean(u_B(z.squeeze(1)) - log_det_f)

            # Reverse: B → A
            x_rev, log_det_inv = model.inverse(y_batch)
            loss_rev = torch.mean(u_A(x_rev.squeeze(1)) + log_det_inv)

            loss = loss_fwd + loss_rev

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(x_batch)

        epoch_loss /= len(dataloader_A.dataset)
        print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--state_a", type=str, default="data/state_a_samples.npy")
    parser.add_argument("--state_b", type=str, default="data/state_b_samples.npy")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load samples
    dataloader_A = load_data(args.state_a, batch_size=args.batch_size)
    dataloader_B = load_data(args.state_b, batch_size=args.batch_size)

    # Define potentials for state A and B
    potential_A = DoubleWellPotential(a=1.0, b=1.0)
    potential_B = DoubleWellPotential(a=1.0, b=2.0)

    def u_A(x: torch.Tensor) -> torch.Tensor:
        return potential_A.energy(x)

    def u_B(x: torch.Tensor) -> torch.Tensor:
        return potential_B.energy(x)

    # Initialize model
    model = RealNVP(n_coupling_layers=4, hidden_dim=64)

    # Train symmetrically
    train_bidirectional_flow_model(
        model=model,
        u_A=u_A,
        u_B=u_B,
        dataloader_A=dataloader_A,
        dataloader_B=dataloader_B,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=args.device,
    )

    # Save trained model
    torch.save(model.state_dict(), "realnvp_trained.pt")
    print("Model saved to realnvp_trained.pt")
