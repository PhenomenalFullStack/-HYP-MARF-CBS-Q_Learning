"""
ablation_plots_4_vehicles.py: plots results in results/ablation_study/.
State Space: 4 Vehicles, 10x10 grid.
"""

import os
import csv
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "results/four_vehicles")
CSV_PATH = os.path.join(OUTPUT_DIR, "ablation_results_4_vehicles.csv")


def load_rows():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def plot_ablation(rows, ablation_name):
    subset = [r for r in rows if r["ablation"] == ablation_name]
    if not subset:
        return
    baseline = [r for r in rows if r["ablation"] == "baseline"]
    if baseline:
        subset = baseline + subset

    labels = [r["varied"] for r in subset]
    steps = [float(r["steps"]) for r in subset]
    rewards = [float(r["reward"]) for r in subset]
    collisions = [float(r["collisions"]) for r in subset]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Ablation: {ablation_name}", fontsize=14, fontweight="bold")

    axes[0].bar(range(len(labels)), steps, color="steelblue")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("Steps")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(range(len(labels)), rewards, color="green")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("Total Reward")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(range(len(labels)), collisions, color="red")
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    axes[2].set_ylabel("Collisions")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"ablation_{ablation_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    rows = load_rows()
    ablations = sorted(set(r["ablation"] for r in rows if r["ablation"] != "baseline"))
    for ab in ablations:
        plot_ablation(rows, ab)
    print("All ablation plots generated.")


if __name__ == "__main__":
    main()
