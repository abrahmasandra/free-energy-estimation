import numpy as np
import matplotlib.pyplot as plt

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
        return self.a * (x**2 - self.b)**2

    def grad(self, x):
        """
        Compute the gradient of the potential ∇U(x).
        dU/dx = 4a * x * (x^2 - b)
        """
        return 4 * self.a * x * (x**2 - self.b)

    def metropolis_sample(self, n_samples=1000, step_size=0.5, burn_in=100, x0=None):
        """
        Simple Metropolis-Hastings sampler.
        """
        if x0 is None:
            x = np.random.randn()
        else:
            x = x0

        samples = []
        accepted = 0

        for i in range(n_samples + burn_in):
            x_prop = x + np.random.normal(scale=step_size)
            dE = self.energy(x_prop) - self.energy(x)
            if dE < 0 or np.random.rand() < np.exp(-dE):
                x = x_prop
                accepted += 1
            if i >= burn_in:
                samples.append(x)

        print(f"Acceptance rate: {accepted / (n_samples + burn_in):.2f}")
        return np.array(samples)
    
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
