# Free Energy Estimation Using Normalizing Flows

This project estimates free energy differences (ΔF) between thermodynamic states using **normalizing flows**. Inspired by methods like BAR and TFEP, it uses invertible neural networks (RealNVP) to learn transformations between Boltzmann-distributed samples and computes ΔF from energy and Jacobian terms.

---

## 🔬 Overview

- Learn a bijective mapping between state A and state B
- Estimate ΔF via:
  $$\Delta F = \mathbb{E}_{x \sim p_A} \left[ U_B(f(x)) - \log |\det J_f(x)| \right]$$
- Train flows in both directions (A → B and B → A)
- Evaluate overlap using PCA, energy histograms, and KL/Wasserstein metrics
- Compare ΔF estimates to ground truth (1D double well) or MBAR (Lennard-Jones)

---

## 📁 Project Structure

```plaintext
ml_free_energy_estimation/
├── README.md
├── main.py # Main script to run the project
├── data/
│   ├── synthetic_sampler.py # Generates and saves samples from potentials
├── models/
│   ├── realnvp.py # RealNVP normalizing flow model (1D, soon N-D)
├── potentials/
│   ├── double_well.py # 1D and multi-particle double well potential
│   ├── lj.py # Lennard-Jones potential
├── training/
│   ├── train_flow.py # Bidirectional training (A→B and B→A)
├── evaluation/
│   ├── free_energy_estimator.py # Estimate ΔF with confidence intervals
│   ├── visualize_flows.py # Visualize distribution mappings and potentials
│   ├── energy_histogram_comparison.py # Compare energy histograms
├── comparison/
│   ├── ground_truth_double_well.py # Ground truth for double well potential
│   ├── ground_truth_lj_bar.py # MBAR for Lennard-Jones potential
│   ├── plot_delta_F_comparison.py # Compare ΔF estimates
```

---

## 🧪 Setup

**Dependencies (via `requirements.txt`)**:
- `numpy`
- `torch`
- `matplotlib`
- `seaborn` (optional for multi-d plots)

Install:

```bash
git clone https://github.com/your-username/ml_free_energy_estimation.git
cd ml_free_energy_estimation

python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Usage

Run all steps via the CLI:

```bash
# Step 1: Generate data
python main.py \
  --potential double_well \
  --dim 1 --seed 1 --n_samples 100000\
  --potential_kwargs_a '{"a": 1.0, "b": 1.0}' \
  --potential_kwargs_b '{"a": 5.0, "b": 2.0}' \
  --generate --sampling_method langevin --plot_samples
```

```bash
# Step 2: Train flow from State A → B and B → A
python main.py \
  --potential double_well \
  --dim 1 --seed 1 \
  --potential_kwargs_a '{"a": 1.0, "b": 1.0}' \
  --potential_kwargs_b '{"a": 5.0, "b": 2.0}' \
  --train  --n_epochs 50
```

```bash
# Step 3: Estimate ΔF with confidence intervals
python main.py \
  --potential double_well \
  --dim 1 --seed 1 \
  --potential_kwargs_a '{"a": 1.0, "b": 1.0}' \
  --potential_kwargs_b '{"a": 5.0, "b": 2.0}' \
  --evaluate --ground_truth
```

```bash
# Step 4: Visualize mapped vs. true distributions
python main.py \
  --potential double_well \
  --dim 1 --seed 1 \
  --potential_kwargs_a '{"a": 1.0, "b": 1.0}' \
  --potential_kwargs_b '{"a": 5.0, "b": 2.0}' \
  --visualize

python main.py \
  --potential double_well \
  --dim 1 --seed 1 \
  --potential_kwargs_a '{"a": 1.0, "b": 1.0}' \
  --potential_kwargs_b '{"a": 5.0, "b": 2.0}' \
  --visualize_inverse

python main.py \
  --potential double_well \
  --dim 1 --seed 1 \
  --potential_kwargs_a '{"a": 1.0, "b": 1.0}' \
  --potential_kwargs_b '{"a": 5.0, "b": 2.0}' \
  --visualize_energy_histograms
```

## 📊 Free Energy Estimation

Estimates $\Delta F$ using the following methods:

- **Forward KL Divergence**: $\Delta F_{forward} = \log \left( \frac{p_B(x)}{p_A(x)} \right)$
- **Reverse KL Divergence**: $\Delta F_{reverse} = \log \left( \frac{p_A(x)}{p_B(x)} \right)$
- **Symmetrized KL Divergence**: $\Delta F_{sym} = \frac{1}{2} \left( \Delta F_{forward} + \Delta F_{reverse} \right)$

## 📈 Visualization

- Overlap of original and mapped distributions.
- Overlay potential energy curves $U(x)$ for both states.
- Inverse mappings to validate reversibility.
