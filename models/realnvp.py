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
    def __init__(self, dim: int, mask: Tensor, hidden_dim: int = 128) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.scale_net = MLP(input_dim=dim, hidden_dim=hidden_dim, output_dim=dim)
        self.shift_net = MLP(input_dim=dim, hidden_dim=hidden_dim, output_dim=dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_masked = x * self.mask
        scale = self.scale_net(x_masked) * (1 - self.mask)
        scale = torch.clamp(scale, min=-5.0, max=5.0)  # ⬅️ ADD THIS

        shift = self.shift_net(x_masked) * (1 - self.mask)

        y = x_masked + (1 - self.mask) * (x * torch.exp(scale) + shift)
        log_det = torch.sum(scale, dim=1)

        return y, log_det
    
    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        y_masked = y * self.mask
        scale = self.scale_net(y_masked) * (1 - self.mask)
        scale = torch.clamp(scale, min=-5.0, max=5.0)  # ⬅️ ADD THIS
        
        shift = self.shift_net(y_masked) * (1 - self.mask)

        x = y_masked + (1 - self.mask) * ((y - shift) * torch.exp(-scale))
        log_det = -torch.sum(scale, dim=1)

        return x, log_det

class RealNVP(nn.Module):
    """
    RealNVP model for 1D inputs using a stack of affine coupling layers.
    """
    def __init__(self, dim: int, n_coupling_layers: int = 4, hidden_dim: int = 32) -> None:
        super().__init__()
        self.dim = dim
        self.layers: nn.ModuleList = nn.ModuleList()

        for i in range(n_coupling_layers):
            mask_pattern = [(j + i) % 2 for j in range(dim)]
            mask: Tensor = torch.tensor(mask_pattern, dtype=torch.float32)
            self.layers.append(CouplingLayer(dim=dim, mask=mask, hidden_dim=hidden_dim))

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
