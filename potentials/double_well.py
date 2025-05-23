import numpy as np
import matplotlib.pyplot as plt
import torch

class DoubleWellPotential:
    def __init__(self, a=1.0, b=1.0):
        """
        Initialize a double well potential of the form:
        U(x) = a * (x^2 - b)^2
        """
        self.a = a
        self.b = b

    def energy(self, x):
        """
        Evaluate the potential energy U(x).
        """
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.squeeze(1)
        
        return self.a * (x**2 - self.b)**2

    def grad(self, x):
        """
        Compute the gradient of the potential ∇U(x).
        dU/dx = 4a * x * (x^2 - b)
        """
        return 4 * self.a * x * (x**2 - self.b)
    
    def langevin_sample(
        self,
        n_samples=5000,
        step_size=0.01,
        n_steps=100,
        burn_in=100,
        n_chains=10,         # Number of parallel chains
        seed=None
    ):
        """
        Parallel Unadjusted Langevin Dynamics (ULA) sampling.
        Returns: numpy array of shape (n_samples * n_chains, 1)
        """
        import torch

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        x = torch.randn((n_chains, 1), dtype=torch.float32, requires_grad=True)
        samples = []

        optimizer = torch.optim.SGD([x], lr=step_size)

        total_steps = n_steps + n_samples + burn_in
        for step in range(total_steps):
            optimizer.zero_grad()
            u = self.energy(x).sum()  # Sum over chains
            u.backward()

            with torch.no_grad():
                noise = torch.randn_like(x)
                x -= 0.5 * step_size * x.grad
                x += (step_size ** 0.5) * noise
                x.grad.zero_()

                if step >= n_steps + burn_in:
                    samples.append(x.detach().clone())  # shape: (n_chains, 1)

        return torch.cat(samples, dim=0).numpy()  # shape: (n_samples * n_chains, 1)

    def metropolis_sample(
        self,
        n_samples: int = 5000,
        step_size: float = 0.1,
        burn_in: int = 100,
        device: str = "cpu",
        seed: int = None,
    ) -> np.ndarray:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        x = torch.randn(1, 1, device=device)  # Initial configuration
        samples = []

        accepted = 0
        for i in range(n_samples + burn_in):
            x_prop = x + step_size * torch.randn_like(x)
            dE = self.energy(x_prop) - self.energy(x)

            accept = torch.rand(1, device=device) < torch.exp(-dE)
            x = torch.where(accept.view(-1, 1), x_prop, x)

            if i >= burn_in:
                samples.append(x.detach().cpu().view(1).numpy())
                accepted += int(accept.item())

        print(f"Acceptance rate: {accepted / n_samples:.2f}")
        print(f"Samples shape: {np.stack(samples, axis=0).shape}")
        return np.stack(samples, axis=0)
    
    def plot(self, x_range=(-3, 3), num_points=300, show=True, ax=None, label=None):
        """
        Plot the potential energy curve U(x) over a range.
        """
        xs = np.linspace(*x_range, num_points)
        us = self.energy(xs)

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(xs, us, label=label or f'a={self.a}, b={self.b}')
        ax.set_xlabel("x")
        ax.set_ylabel("U(x)")
        ax.set_title("Double Well Potential")
        if label or ax.get_legend_handles_labels()[0]:
            ax.legend()

        if show:
            plt.show()
