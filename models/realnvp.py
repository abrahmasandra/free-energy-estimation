import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class MLP(nn.Module):
    """
    Small MLP used to produce scale and shift in a coupling layer.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CouplingLayer(nn.Module):
    """
    Affine coupling layer for RealNVP with binary masking.
    """
    def __init__(self, mask: Tensor, hidden_dim: int = 32) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.scale_net = MLP(input_dim=1, hidden_dim=hidden_dim, output_dim=1)
        self.shift_net = MLP(input_dim=1, hidden_dim=hidden_dim, output_dim=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_masked: Tensor = x * self.mask
        x_pass: Tensor = x * (1 - self.mask)

        scale: Tensor = self.scale_net(x_masked)
        shift: Tensor = self.shift_net(x_masked)

        y: Tensor = x_pass * torch.exp(scale) + shift
        y = y * (1 - self.mask) + x_masked

        log_det_jacobian: Tensor = torch.sum((1 - self.mask) * scale, dim=1)
        return y, log_det_jacobian

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        y_masked: Tensor = y * self.mask
        y_pass: Tensor = y * (1 - self.mask)

        scale: Tensor = self.scale_net(y_masked)
        shift: Tensor = self.shift_net(y_masked)

        x: Tensor = (y_pass - shift) * torch.exp(-scale)
        x = x * (1 - self.mask) + y_masked

        log_det_jacobian: Tensor = -torch.sum((1 - self.mask) * scale, dim=1)
        return x, log_det_jacobian


class RealNVP(nn.Module):
    """
    RealNVP model for 1D inputs using a stack of affine coupling layers.
    """
    def __init__(self, n_coupling_layers: int = 4, hidden_dim: int = 32) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()

        for i in range(n_coupling_layers):
            # Alternating binary mask: [0] or [1]
            mask_value: float = float(i % 2)
            mask: Tensor = torch.tensor([mask_value], dtype=torch.float32)
            self.layers.append(CouplingLayer(mask=mask, hidden_dim=hidden_dim))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_total: Tensor = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_total: Tensor = torch.zeros(z.shape[0], device=z.device)
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_total += log_det
        return z, log_det_total
    
if __name__ == "__main__":
    model: RealNVP = RealNVP()
    x: Tensor = torch.randn(128, 1)
    z, log_det = model(x)
    x_rec, inv_log_det = model.inverse(z)
    print("Reconstruction error:", torch.mean((x - x_rec)**2).item())
