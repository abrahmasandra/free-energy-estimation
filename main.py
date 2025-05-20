import argparse
import torch
from data.synthetic_sampler import run_sampling
from training.train_flow import train_bidirectional_flow_model
from evaluation.free_energy_estimator import estimate_delta_F_with_ci
from evaluation.visualize_flows import visualize_flow_mapping
from models.realnvp import RealNVP
from potentials.double_well import DoubleWellPotential


def get_potentials():
    potential_A = DoubleWellPotential(a=1.0, b=1.0)
    potential_B = DoubleWellPotential(a=1.0, b=2.0)

    def u_A(x): return potential_A.energy(x)
    def u_B(x): return potential_B.energy(x)

    return potential_A, potential_B, u_A, u_B


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data")
    parser.add_argument("--train", action="store_true", help="Train the normalizing flow")
    parser.add_argument("--evaluate", action="store_true", help="Estimate free energy")
    parser.add_argument("--visualize", action="store_true", help="Plot distribution overlap")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--visualize_inverse", action="store_true", help="Visualize inverse flow (B → A)")
    args = parser.parse_args()

    model_path = "realnvp_trained.pt"
    file_A = "data/state_a_samples.npy"
    file_B = "data/state_b_samples.npy"

    potential_A, potential_B, u_A, u_B = get_potentials()
    
    if args.generate:
        print("Generating data...")
        run_sampling("double_well", {"a": 1.0, "b": 1.0}, n_samples=5000, seed=0, dim=1, out_file="state_a_samples.npy")
        run_sampling("double_well", {"a": 1.0, "b": 2.0}, n_samples=5000, seed=1, dim=1, out_file="state_b_samples.npy")

    if args.train:
        print("Training model...")
        from training.train_flow import load_data
        dataloader_A = load_data(file_A)
        dataloader_B = load_data(file_B)

        model = RealNVP(n_coupling_layers=4, hidden_dim=64)
        train_bidirectional_flow_model(
            model=model,
            u_A=u_A,
            u_B=u_B,
            dataloader_A=dataloader_A,
            dataloader_B=dataloader_B,
            n_epochs=100,
            lr=1e-3,
            device=args.device
        )
        torch.save(model.state_dict(), model_path)

    if args.evaluate:
        print("Estimating ΔF...")
        from evaluation.free_energy_estimator import load_samples
        dataloader_A = load_samples(file_A)
        dataloader_B = load_samples(file_B)
        potential_A, potential_B, u_A, u_B = get_potentials()

        model = RealNVP(n_coupling_layers=4, hidden_dim=64)
        model.load_state_dict(torch.load(model_path, map_location=args.device))

        (fwd, fwd_ci), (rev, rev_ci), (sym, sym_ci) = estimate_delta_F_with_ci(
            model=model,
            u_A=u_A,
            u_B=u_B,
            data_A=dataloader_A,
            data_B=dataloader_B,
            device=args.device
        )

        print(f"ΔF (A→B):   {fwd:.4f}   [95% CI: {fwd_ci[0]:.4f}, {fwd_ci[1]:.4f}]")
        print(f"ΔF (B→A):   {rev:.4f}   [95% CI: {rev_ci[0]:.4f}, {rev_ci[1]:.4f}]")
        print(f"ΔF (Sym):   {sym:.4f}   [95% CI: {sym_ci[0]:.4f}, {sym_ci[1]:.4f}]")

    if args.visualize:
        print("Visualizing flow mapping...")
        model = RealNVP(n_coupling_layers=4, hidden_dim=64)
        model.load_state_dict(torch.load(model_path, map_location=args.device))

        visualize_flow_mapping(
            model=model,
            samples_A_path=file_A,
            samples_B_path=file_B,
            potential_fn=u_B,
            potential_label="U_B(x)",
            device=args.device
        )

    if args.visualize_inverse:
        print("Visualizing inverse flow mapping...")
        model = RealNVP(n_coupling_layers=4, hidden_dim=64)
        model.load_state_dict(torch.load(model_path, map_location=args.device))

        from evaluation.visualize_flows import visualize_inverse_flow_mapping
        visualize_inverse_flow_mapping(
            model=model,
            samples_A_path=file_A,
            samples_B_path=file_B,
            potential_fn=u_A,
            potential_label="U_A(x)",
            device=args.device
        )


if __name__ == "__main__":
    main()
