import argparse
import torch
import json
import numpy as np
import os

from data.synthetic_sampler import run_sampling, plot_sample_histogram
from training.train_flow import train_bidirectional_flow_model, load_data, plot_loss_curves
from evaluation.free_energy_estimator import estimate_delta_F_with_ci
from evaluation.energy_histogram_comparison import plot_energy_histograms
from evaluation.visualize_flows import (
    visualize_flow_mapping,
    visualize_inverse_flow_mapping,
    visualize_pca_mapped_vs_target,
    visualize_pca_inverse_mapping,
)
from comparison.ground_truth_double_well import compute_delta_F_true
from comparison.ground_truth_lj_bar import compute_delta_F_MBAR
from comparison.plot_delta_F_comparison import plot_delta_f_comparison

from models.realnvp import RealNVP
from potentials.double_well import DoubleWellPotential
from potentials.lj import LennardJonesPotential


def get_potential(name: str, dim: int = 1, **kwargs):
    """
    Factory method to construct a potential by name.
    Add new potentials here as needed.
    """
    if name == "double_well":
        if dim != 1:
            raise NotImplementedError("Double well currently only supports 1D.")
        return DoubleWellPotential(**kwargs)
    elif name == "lj":
        if dim % 3 != 0:
            raise NotImplementedError("Lennard-Jones potential currently only supports 3D.")
        return LennardJonesPotential(**kwargs)
    
    # elif name == "muller_brown":
    #     return MullerBrownPotential(dim=dim, **kwargs)

    else:
        raise ValueError(f"Unknown potential: {name}")


def main():
    parser = argparse.ArgumentParser(description="Run free energy workflow")
    parser.add_argument("--potential", type=str, choices=["double_well", "lj"], required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--sampling_method", type=str, default="metropolis", choices=["metropolis", "langevin"])
    parser.add_argument("--plot_samples", action="store_true", help="Plot generated samples")
    
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--mbar", action="store_true", help="Estimate ΔF using MBAR (LJ systems only)")
    parser.add_argument("--ground_truth", action="store_true", help="Compute ground truth ΔF for double well")

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_inverse", action="store_true")
    parser.add_argument("--visualize_pca", action="store_true")
    parser.add_argument("--visualize_inverse_pca", action="store_true")
    parser.add_argument("--visualize_energy_histograms", action="store_true", help="Plot energy histograms")
    parser.add_argument("--potential_kwargs_a", type=str, required=True, help="JSON string of potential kwargs for state A")
    parser.add_argument("--potential_kwargs_b", type=str, required=True, help="JSON string of potential kwargs for state B")
    args = parser.parse_args()

    model_path = "realnvp_trained.pt"

    kwargs_A = json.loads(args.potential_kwargs_a)
    kwargs_B = json.loads(args.potential_kwargs_b)

    state_a_file = f"data/state_a_{args.potential}.npy"
    state_b_file = f"data/state_b_{args.potential}.npy"
    model_path = f"realnvp_{args.potential}_{args.dim}d.pt"
    
    if args.generate:
        print("Generating samples...")
        run_sampling(
            potential_name=args.potential,
            potential_kwargs=kwargs_A,
            n_samples=args.n_samples,
            seed=args.seed,
            out_file=f"state_a_{args.potential}.npy",
            method=args.sampling_method,
            dim=args.dim,
        )
        run_sampling(
            potential_name=args.potential,
            potential_kwargs=kwargs_B,
            n_samples=args.n_samples,
            seed=args.seed,
            out_file=f"state_b_{args.potential}.npy",
            method=args.sampling_method,
            dim=args.dim,
        )

        print(f"Generated samples saved to {state_a_file} and {state_b_file}")
        if args.plot_samples and args.dim != 1:
            # cannot plot samples for >1D
            raise ValueError("Plotting samples is only supported for 1D systems.")
        elif args.plot_samples:
            samples_A = np.load(state_a_file)
            samples_B = np.load(state_b_file)
            plot_sample_histogram(
                samples=samples_A,
                potential=get_potential(name=args.potential, dim=args.dim, **kwargs_A),
                title=f"Samples from {args.potential} potential (state A)",
                out_path=f"state_a_{args.potential}_histogram.png",
                beta=kwargs_A.get("beta", 1.0)
            )

            plot_sample_histogram(
                samples=samples_B,
                potential=get_potential(name=args.potential, dim=args.dim, **kwargs_B),
                title=f"Samples from {args.potential} potential (state B)",
                out_path=f"state_b_{args.potential}_histogram.png",
                beta=kwargs_B.get("beta", 1.0)
            )

    if args.train:
        print("Training flow model...")
        train_A, val_A, dim_A = load_data(state_a_file, args.batch_size)
        train_B, val_B, dim_B = load_data(state_b_file, args.batch_size)
        assert dim_A == dim_B == args.dim

        potential_A = get_potential(name=args.potential, dim=args.dim, **kwargs_A)
        potential_B = get_potential(name=args.potential, dim=args.dim, **kwargs_B)

        model = RealNVP(dim=args.dim, n_coupling_layers=8, hidden_dim=128)
        # if the model already exists, load it
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=args.device))
        else:
            print("No existing model found. Training a new one.")

        train_losses, val_losses = train_bidirectional_flow_model(
            model=model,
            u_A=lambda x: potential_A.energy(x),
            u_B=lambda x: potential_B.energy(x),
            train_A=train_A,
            train_B=train_B,
            val_A=val_A,
            val_B=val_B,
            n_epochs=args.n_epochs,
            lr=args.lr,
            device=args.device,
        )

        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        plot_loss_curves(train_losses, val_losses, out_path="loss_curves.png")
        print("Training complete. Loss curves saved to loss_curves.png")
    if args.evaluate:
        print("Estimating free energy difference...")
        train_A, val_A, _ = load_data(state_a_file, args.batch_size)
        train_B, val_B, _ = load_data(state_b_file, args.batch_size)

        # combine train and val loaders for evaluation
        dataloader_A = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([train_A.dataset, val_A.dataset]),
            batch_size=args.batch_size,
            shuffle=False,
        )
        dataloader_B = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([train_B.dataset, val_B.dataset]),
            batch_size=args.batch_size,
            shuffle=False,
        )

        potential_A = get_potential(name=args.potential, dim=args.dim, **kwargs_A)
        potential_B = get_potential(name=args.potential, dim=args.dim, **kwargs_B)

        model = RealNVP(dim=args.dim, n_coupling_layers=8, hidden_dim=128)
        model.load_state_dict(torch.load(model_path, map_location=args.device))

        (fwd_mean, fwd_ci), (rev_mean, rev_ci), (sym_mean, sym_ci) = estimate_delta_F_with_ci(
            model=model,
            u_A=lambda x: potential_A.energy(x),
            u_B=lambda x: potential_B.energy(x),
            data_A=dataloader_A,
            data_B=dataloader_B,
            device=args.device,
        )

        print(f"\nΔF (A→B): {fwd_mean:.4f}  [95% CI: {fwd_ci[0]:.4f}, {fwd_ci[1]:.4f}]")
        print(f"ΔF (B→A): {rev_mean:.4f}  [95% CI: {rev_ci[0]:.4f}, {rev_ci[1]:.4f}]")
        print(f"ΔF (Sym) : {sym_mean:.4f}  [95% CI: {sym_ci[0]:.4f}, {sym_ci[1]:.4f}]\n")

        delta_F_true = None
        delta_F_mbar = None
        if args.ground_truth:
            if args.potential == "double_well":
                print("Computing ground truth ΔF via numerical integration...")
                delta_F_true = compute_delta_F_true(potential_A, potential_B, bounds=(-10, 10))
                print(f"ΔF (ground truth via integration): {delta_F_true:.6f}")
            elif args.potential == "lj":
                if args.mbar:
                    print("Computing ΔF via MBAR...")
                    delta_F_mbar = compute_delta_F_MBAR(
                        data_A=dataloader_A,
                        data_B=dataloader_B,
                        u_A=lambda x: potential_A.energy(x),
                        u_B=lambda x: potential_B.energy(x),
                    )
                    print(f"ΔF (ground truth via MBAR): {delta_F_mbar:.6f}")
                else:
                    print("Ground truth MBAR is only available for Lennard-Jones systems. Use --mbar to compute it.")
            else:
                print("Ground truth computation is not implemented for this potential.")
            
            plot_delta_f_comparison(
                delta_f_fwd=fwd_mean,
                delta_f_rev=rev_mean,
                delta_f_sym=sym_mean,
                delta_f_ref=delta_F_true or delta_F_mbar,
                label_ref="True ΔF" if args.potential == "double_well" else "MBAR ΔF",
                out_path="delta_f_comparison.png",
            )
                

    if args.visualize:
        model = RealNVP(dim=args.dim, n_coupling_layers=8, hidden_dim=128)
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        visualize_flow_mapping(model, state_a_file, state_b_file, device=args.device)

    if args.visualize_inverse:
        model = RealNVP(dim=args.dim, n_coupling_layers=8, hidden_dim=128)
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        visualize_inverse_flow_mapping(model, state_a_file, state_b_file, device=args.device)

    if args.visualize_pca:
        model = RealNVP(dim=args.dim, n_coupling_layers=8, hidden_dim=128)
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        visualize_pca_mapped_vs_target(model, state_a_file, state_b_file, device=args.device)

    if args.visualize_inverse_pca:
        model = RealNVP(dim=args.dim, n_coupling_layers=8, hidden_dim=128)
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        visualize_pca_inverse_mapping(model, state_a_file, state_b_file, device=args.device)
    
    if args.visualize_energy_histograms:
        model = RealNVP(dim=args.dim, n_coupling_layers=8, hidden_dim=128)
        model.load_state_dict(torch.load(model_path, map_location=args.device))

        potential_A = get_potential(name=args.potential, dim=args.dim, **kwargs_A)
        potential_B = get_potential(name=args.potential, dim=args.dim, **kwargs_B)
        plot_energy_histograms(
            model=model,
            samples_A_path=state_a_file,
            samples_B_path=state_b_file,
            u_A=lambda x: potential_A.energy(x),
            u_B=lambda x: potential_B.energy(x),
            device=args.device
        )


if __name__ == "__main__":
    main()
