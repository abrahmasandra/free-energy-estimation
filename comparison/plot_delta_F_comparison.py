import matplotlib.pyplot as plt
import numpy as np


def plot_delta_f_comparison(
    delta_f_fwd: float,
    delta_f_rev: float,
    delta_f_sym: float,
    delta_f_ref: float,
    label_ref: str = "True ΔF",
    out_path: str = None,
):
    """
    Create a bar chart comparing flow-based and reference ΔF estimates.
    """
    methods = ["Flow Fwd KL", "Flow Rev KL", "Flow Sym", label_ref]
    values = [delta_f_fwd, delta_f_rev, delta_f_sym, delta_f_ref]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(methods, values, color=colors, alpha=0.8)
    plt.ylabel("ΔF Estimate")
    plt.title("Free Energy Estimates: Flow vs " + label_ref)
    plt.xticks(rotation=15)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2.0, val + 0.02, f"{val:.3f}", 
                 ha='center', va='bottom', fontsize=10)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"Saved ΔF comparison plot to {out_path}")
    else:
        plt.show()
