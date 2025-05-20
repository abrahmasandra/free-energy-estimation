# Free Energy Estimation via Normalizing Flows

This project estimates the free energy difference \( \Delta F \) between two thermodynamic states by learning a transformation between their probability distributions using **normalizing flows** (specifically, RealNVP). It supports both 1D and multi-particle systems, and provides forward, reverse, and symmetrized estimates of \( \Delta F \), analogous to KL, reverse KL, and BAR/LBAR.

---

## 🧪 Project Goals

- Define two thermodynamic states (State A and State B) using parameterized potential energy functions.
- Sample from each state's Boltzmann distribution \( p(x) \propto \exp(-U(x)) \).
- Train a **normalizing flow** to map samples from State A → State B (and vice versa).
- Estimate the free energy difference \( \Delta F \) using learned mappings.
- Visualize distribution overlap and potential energy landscapes.

---

## 📁 Project Structure

```plaintext
ml_free_energy_estimation/
├── README.md
├── main.py # Main script to run the project
├── data/
│   ├── state_a_samples.npy
│   ├── state_b_samples.npy
│   └── synthetic_sampler.py # Generates and saves samples from potentials
├── models/
│   ├── realnvp.py # RealNVP normalizing flow model (1D, soon N-D)
├── potentials/
│   ├── double_well.py # 1D and multi-particle double well potential
├── training/
│   ├── train_flow.py # Bidirectional training (A→B and B→A)
├── evaluation/
│   ├── free_energy_estimator.py # Estimate ΔF with confidence intervals
│   ├── visualize_flows.py # Visualize distribution mappings and potentials
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
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Usage

Run all steps via the CLI:

```bash
# Step 1: Generate data
python main.py --generate

# Step 2: Train flow from State A → B and B → A
python main.py --train

# Step 3: Estimate ΔF with confidence intervals
python main.py --evaluate

# Step 4: Visualize mapped vs. true distributions
python main.py --visualize
python main.py --visualize_inverse
```

## 📊 Free Energy Estimation

Estimates \( \Delta F \) using the following methods:

- **Forward KL Divergence**: \( \Delta F_{forward} = \log \left( \frac{p_B(x)}{p_A(x)} \right) \)
- **Reverse KL Divergence**: \( \Delta F_{reverse} = \log \left( \frac{p_A(x)}{p_B(x)} \right) \)
- **Symmetrized KL Divergence**: \( \Delta F_{sym} = \frac{1}{2} \left( \Delta F_{forward} + \Delta F_{reverse} \right) \)

## 📈 Visualization

- Overlap of original and mapped distributions.
- Overlay potential energy curves \(U(x)\) for both states.
- Inverse mappings to validate reversibility.
