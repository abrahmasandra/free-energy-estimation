import torch
import numpy as np
from typing import Callable

class LennardJonesPotential:
    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0, n_particles: int = 5):
        self.epsilon = epsilon
        self.sigma = sigma
        self.n = n_particles

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch_size, n_particles * 3)
        Returns: Tensor of shape (batch_size,)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, self.n, 3)

        # Pairwise differences
        rij = x[:, :, None, :] - x[:, None, :, :]  # (B, N, N, 3)
        dij = torch.norm(rij, dim=-1)              # (B, N, N)

        # Mask self-interactions before computing inverse powers
        mask = torch.ones_like(dij)
        mask[:, range(self.n), range(self.n)] = 0.0

        # Avoid NaNs: replace diagonal with large number before division
        dij_safe = dij + (1.0 - mask) * 1e5

        inv_r6 = (self.sigma / dij_safe)**6
        lj_pairwise = 4 * self.epsilon * (inv_r6**2 - inv_r6)

        # Zero out diagonal terms explicitly (optional)
        lj_pairwise *= mask

        energy = 0.5 * torch.sum(lj_pairwise, dim=(1, 2))
        return energy
    
    def metropolis_sample(
        self,
        n_samples: int = 5000,
        step_size: float = 0.1,
        burn_in: int = 100,
        device: str = "cpu",
        seed: int = 42
    ) -> np.ndarray:
        """
        Metropolis-Hastings sampler for N-particle 3D system.
        Returns:
            samples: np.ndarray of shape (n_samples, n_particles, 3)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        dim = self.n * 3
        x = torch.randn(1, dim, device=device)  # Initial configuration
        samples = []

        accepted = 0
        for i in range(n_samples + burn_in):
            x_prop = x + step_size * torch.randn_like(x)
            dE = self.energy(x_prop) - self.energy(x)

            accept = torch.rand(1, device=device) < torch.exp(-dE)
            x = torch.where(accept.view(-1, 1), x_prop, x)

            if i >= burn_in:
                samples.append(x.detach().cpu().view(self.n, 3).numpy())
                accepted += int(accept.item())

        print(f"Acceptance rate: {accepted / n_samples:.2f}")
        print(f"Final sample shape: {np.stack(samples, axis=0).shape}")
        return np.stack(samples, axis=0)
