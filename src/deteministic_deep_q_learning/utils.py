"""
utils.py

Plotting and helper utilities.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def plot_noise_curve(results, out_dir):
    noise = [r["noise_level"] for r in results]
    coll = [r["collision_rate"] for r in results]
    goal = [r["goal_rate"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.set_xlabel("Sensor Noise Level")
    ax1.set_ylabel("Collision Rate", color="tab:red")
    ax1.plot(noise, coll, "o-", color="tab:red", label="Collision Rate")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Goal Rate", color="tab:blue")
    ax2.plot(noise, goal, "s--", color="tab:blue", label="Goal Rate")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("DDPG Performance vs Sensor Noise")
    fig.tight_layout()
    out = os.path.join(out_dir, "ddpg_noise_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved {out}")


def plot_comparison(ddpg_results, baseline_csv, out_dir):
    rows = []
    with open(baseline_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    methods = sorted(set(r["method"] for r in rows))
    noise_levels = [r["noise_level"] for r in ddpg_results]

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:
        xs, ys = [], []
        for r in rows:
            if r["method"] == method:
                xs.append(float(r["noise_level"]))
                ys.append(float(r["collision_rate"]))
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        ax.plot(xs, ys, "o-", label=method)

    ax.plot(
        noise_levels,
        [r["collision_rate"] for r in ddpg_results],
        "k*-", linewidth=2, markersize=10, label="DDPG (ours)"
    )

    ax.set_xlabel("Sensor Noise Level")
    ax.set_ylabel("Collision Rate")
    ax.set_title("Collision Rate Comparison Across Methods")
    ax.legend()
    ax.grid(alpha=0.3)

    out = os.path.join(out_dir, "comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {out}")


def plot_learning_curve(log_csv, out_dir):
    episodes, rewards, collisions = [], [], []
    with open(log_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["ep_reward"]))
            collisions.append(int(row["ep_collisions"]))

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward", color="tab:green")
    ax1.plot(episodes, rewards, color="tab:green", alpha=0.7)
    ax1.tick_params(axis="y", labelcolor="tab:green")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Collisions", color="tab:red")
    ax2.plot(episodes, collisions, color="tab:red", alpha=0.5)
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("DDPG Learning Curve")
    fig.tight_layout()
    out = os.path.join(out_dir, "learning_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved {out}")


def moving_average(x, window=20):
    if len(x) < window:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)