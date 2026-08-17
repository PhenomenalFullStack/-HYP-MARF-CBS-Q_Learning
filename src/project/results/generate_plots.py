# generate_plots.py

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
JSON_PATH = os.path.join(_BASE, "simulation_results.json")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

def load_data(filepath=JSON_PATH):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Could not find {filepath}\n"
            f"Make sure the script is in 'src/project/results/' and the JSON is in 'src/project/'."
        )
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_makespan(schedule):
    """Return the maximum time step across all vehicles."""
    max_time = 0
    for steps in schedule.values():
        if steps:
            last_time = steps[-1]["time"]
            if last_time > max_time:
                max_time = last_time
    return max_time

def compute_accuracy(entry):
    """
    Accuracy = CBS_makespan / hybrid_makespan
    Reflects how efficiently the hybrid replicates the optimal plan length.
    """
    actual = entry.get("actual_schedule", {})
    cbs = entry.get("cbs_schedule", {})

    if not actual or not cbs:
        return 0.0

    cbs_makespan = compute_makespan(cbs)
    hybrid_makespan = compute_makespan(actual)

    if hybrid_makespan == 0:
        return 0.0

    return min(1.0, cbs_makespan / hybrid_makespan)

def main():
    data = load_data()

    noise_groups = defaultdict(list)

    for entry in data:
        cfg = entry.get("configuration", {})
        sensor_noise = cfg.get("sensor_noise", 0)   # 0 or 1
        accuracy = compute_accuracy(entry)
        collisions = 0   

        noise_groups[sensor_noise].append({
            "accuracy": accuracy,
            "collisions": collisions
        })

    noise_values = sorted(noise_groups.keys())   # [0, 1]
    avg_accuracy = []
    avg_collisions = []

    for n in noise_values:
        accs = [d["accuracy"] for d in noise_groups[n]]
        colls = [d["collisions"] for d in noise_groups[n]]
        avg_accuracy.append(np.mean(accs) if accs else 0.0)
        avg_collisions.append(np.mean(colls) if colls else 0.0)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Model Accuracy (Efficiency) Plot
    plt.figure(figsize=(6, 4))
    bars = plt.bar([str(n) for n in noise_values], avg_accuracy,
                   color=['#1f77b4', '#ff7f0e'])
    plt.xlabel("Sensor Noise (0=off, 1=on)")
    plt.ylabel("Model Accuracy")
    # "How close the hybrid stays to the global plan in terms of overall time steps (CBS_makespan / hybrid_makespan)."
    plt.title("HybridModel Accuracy")
    plt.ylim(0, 1.05)
    for bar, acc in zip(bars, avg_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{acc:.3f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_accuracy.png"), dpi=150)
    plt.close()

    # Collisions vs Sensor Noise Plot
    plt.figure(figsize=(6, 4))
    bars = plt.bar([str(n) for n in noise_values], avg_collisions,
                   color=['#2ca02c', '#d62728'])
    plt.xlabel("Sensor Noise (0=off, 1=on)")
    plt.ylabel("Average Collisions")
    plt.title("Collisions vs Sensor Noise (Hybrid Model)")
    for bar, coll in zip(bars, avg_collisions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{coll:.1f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "collisions_vs_noise.png"), dpi=150)
    plt.close()

    print(f"Saved plots to '{PLOTS_DIR}'")

if __name__ == "__main__":
    main()