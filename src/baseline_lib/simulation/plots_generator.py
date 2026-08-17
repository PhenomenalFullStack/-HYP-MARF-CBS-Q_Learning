# plots_generator.py

import os
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Define paths for results and plots directories
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(_HERE, "..", "results", "intersection_schedules.json")
PLOTS_DIR = os.path.join(_HERE, "..", "results", "plots")

# Representative number of vehicles for disturbance plots
REPRESENTATIVE_NV = 5

# Helper function to load the JSON results file
def load_results():
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)

# Helper function to extract the number of collisions from a CBS entry
def get_cbs_collisions(entry):
    cbs_data = entry.get("cbs")
    if not cbs_data:
        return None
    return cbs_data.get("collisions")

# Filter the data to include only entries where the specified disturbance parameter varies, 
# while the other two disturbance parameters are zero.
def filter_by_num_vehicles(data, sensor_noise=0.0, comm_latency=0.0, braking_delay=0.0):
    filtered = {}
    for entry in data.values():
        parameters = entry["parameters"]
        noise_matches = abs(parameters["sensor_noise"] - sensor_noise) < 1e-6
        latency_matches = abs(parameters["comm_latency"] - comm_latency) < 1e-6
        braking_matches = abs(parameters["braking_delay"] - braking_delay) < 1e-6
        if noise_matches and latency_matches and braking_matches:
            filtered[parameters["num_vehicles"]] = entry
    return dict(sorted(filtered.items()))

# Filter the data to include only entries where the specified disturbance parameter varies, 
# while the other two disturbance parameters are zero.
def filter_by_single_disturbance_param(data, num_vehicles, vary_param):
    other_params = {"sensor_noise", "comm_latency", "braking_delay"} - {vary_param}
    filtered = {}
    for entry in data.values():
        parameters = entry["parameters"]
        if parameters["num_vehicles"] != num_vehicles:
            continue
        other_params_are_zero = all(
            abs(parameters[other_param] - 0.0) < 1e-6 for other_param in other_params
        )
        if other_params_are_zero:
            filtered[parameters[vary_param]] = entry
    return dict(sorted(filtered.items()))

# Ensure the plots directory exists
def ensure_plots_directory_exists():
    os.makedirs(PLOTS_DIR, exist_ok=True)

# Helper function to save a plot and close the figure
def save_plot(figure, filename):
    path = os.path.join(PLOTS_DIR, filename)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  saved {path}")

# cbs collisions vs number of vehicles plot
def plot_cbs_collisions_vs_vehicles(data):
    by_vehicle_count = filter_by_num_vehicles(data, 0.0, 0.0, 0.0)
    x_values = list(by_vehicle_count.keys())
    y_values = [get_cbs_collisions(by_vehicle_count[nv]) for nv in x_values]

    figure, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_values, y_values, marker="o", color="tab:blue")
    ax.set_xlabel("Number of vehicles")
    ax.set_ylabel("Collisions")
    ax.set_title("CBS: Collisions vs Number of Vehicles (no disturbance)")
    ax.grid(True, alpha=0.3)
    save_plot(figure, "01_cbs_collisions_vs_vehicles.png")

# cbs collisions vs sensor noise plot
def plot_cbs_collisions_vs_sensor_noise(data):
    by_noise_level = filter_by_single_disturbance_param(
        data, REPRESENTATIVE_NV, "sensor_noise"
    )
    x_values = list(by_noise_level.keys())
    y_values = [get_cbs_collisions(by_noise_level[v]) for v in x_values]

    figure, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_values, y_values, marker="o", color="tab:orange")
    ax.set_xlabel("Sensor noise")
    ax.set_ylabel("Collisions")
    ax.set_title(
        f"CBS: Collisions vs Sensor Noise " f"(nv={REPRESENTATIVE_NV}, others=0)"
    )
    ax.grid(True, alpha=0.3)
    save_plot(figure, "02_cbs_collisions_vs_sensor_noise.png")

# cbs collisions vs communication latency plot
def plot_cbs_collisions_vs_comm_latency(data):
    by_latency_level = filter_by_single_disturbance_param(
        data, REPRESENTATIVE_NV, "comm_latency"
    )
    x_values = list(by_latency_level.keys())
    y_values = [get_cbs_collisions(by_latency_level[v]) for v in x_values]

    figure, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_values, y_values, marker="o", color="tab:green")
    ax.set_xlabel("Communication latency")
    ax.set_ylabel("Collisions")
    ax.set_title(
        f"CBS: Collisions vs Communication Latency "
        f"(nv={REPRESENTATIVE_NV}, others=0)"
    )
    ax.grid(True, alpha=0.3)
    save_plot(figure, "03_cbs_collisions_vs_comm_latency.png")

# cbs collisions vs braking delay plot
def plot_cbs_collisions_vs_braking_delay(data):
    by_braking_level = filter_by_single_disturbance_param(
        data, REPRESENTATIVE_NV, "braking_delay"
    )
    x_values = list(by_braking_level.keys())
    y_values = [get_cbs_collisions(by_braking_level[v]) for v in x_values]

    figure, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_values, y_values, marker="o", color="tab:purple")
    ax.set_xlabel("Braking delay")
    ax.set_ylabel("Collisions")
    ax.set_title(
        f"CBS: Collisions vs Braking Delay " f"(nv={REPRESENTATIVE_NV}, others=0)"
    )
    ax.grid(True, alpha=0.3)
    save_plot(figure, "04_cbs_collisions_vs_braking_delay.png")

# cbs success rate vs total disturbance (sensor_noise + comm_latency + braking_delay) plot
def plot_cbs_success_rate_vs_total_delay(data):
    buckets = {}  # total_delay_steps -> [plans_found, total_attempts]

    for entry in data.values():
        cbs_data = entry.get("cbs")
        if not cbs_data:
            continue
        total_delay_steps = entry["parameters"]["delay_steps"]
        plan_was_found = bool(cbs_data.get("plan_found"))
        if total_delay_steps not in buckets:
            buckets[total_delay_steps] = [0, 0]
        buckets[total_delay_steps][0] += int(plan_was_found)
        buckets[total_delay_steps][1] += 1

    x_values = sorted(buckets.keys())
    y_values = [
        buckets[delay_steps][0] / buckets[delay_steps][1] for delay_steps in x_values
    ]

    figure, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_values, y_values, marker="o", color="tab:blue")
    ax.set_xlabel("Total delay steps (sensor_noise + comm_latency + braking_delay)")
    ax.set_ylabel("CBS plan-found success rate")
    ax.set_title("CBS: Planning Success Rate vs Total Disturbance Severity")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    save_plot(figure, "05_cbs_success_rate_vs_delay.png")

# main
def main():
    print(f"Loading results from {RESULTS_PATH}")
    data = load_results()
    print(f"Loaded {len(data)} configurations.")
    ensure_plots_directory_exists()

    print("\nGenerating plots...")
    plot_cbs_collisions_vs_vehicles(data)
    plot_cbs_collisions_vs_sensor_noise(data)
    plot_cbs_collisions_vs_comm_latency(data)
    plot_cbs_collisions_vs_braking_delay(data)
    plot_cbs_success_rate_vs_total_delay(data)

    print(f"\nAll 5 plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
