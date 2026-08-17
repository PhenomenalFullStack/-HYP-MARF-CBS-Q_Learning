"""
episode_analysis.py
Runs the CBS + Q‑learning pipeline for different numbers of training episodes
and plots the performance metrics (steps, collisions, total reward).
Saves the plot in results/episodic_return/ with a timestamp.
"""

import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from grid import Grid
from vehicle import Vehicle
from cbs_planner import CBSPlanner
from q_learning import QLearningAgent

from grid import Grid
from vehicle import Vehicle
from schedule import Schedule
from cbs_planner import CBSPlanner
from q_learning import QLearningAgent
from simulation import distribute_schedules, reset_simulation, get_position_on_path, run_episode

# Constants (same as simulation.py)
GRID_SIZE = 10
STARTS = {'A': (5, 9), 'B': (9, 4), 'C': (4, 0), 'D': (1, 5)}
GOALS  = {'A': (5, 0), 'B': (0, 4), 'C': (9, 5), 'D': (9, 5)}
VEHICLE_IDS = ['A', 'B', 'C', 'D']
MAX_STEPS = 50

# Config to test: worst‑case disturbance (all three on)
TEST_CONFIG = (1, 1, 1)   # sensor, brake, comm

def run_experiment(episodes, config):
    """
    Run one complete training + evaluation for a given number of episodes
    and a specific disturbance configuration.
    Returns: (avg_reward, collisions, steps)
    """
    sn, bd, cl = config
    delays = {vid: 0 for vid in VEHICLE_IDS}
    delays['A'] = sn + bd + cl

    # Create grid and vehicles
    grid = Grid(GRID_SIZE, GRID_SIZE)
    vehicles = [Vehicle(vid, STARTS[vid], GOALS[vid]) for vid in VEHICLE_IDS]

    # Generate CBS schedule (once)
    planner = CBSPlanner(grid, vehicles)
    schedules = planner.plan(STARTS, GOALS)
    if schedules is None:
        print("CBS failed.")
        return None
    distribute_schedules(vehicles, schedules)

    # Create agents
    agents = {v.vehicle_id: QLearningAgent(v, actions=3, lr=0.1, gamma=0.9, epsilon=1.0)
              for v in vehicles}

    # Training
    for ep in range(episodes):
        run_episode(vehicles, grid, delays, agents, train=True)
        for agent in agents.values():
            agent.decay_epsilon()

    # Evaluation (greedy)
    for agent in agents.values():
        agent.epsilon = 0.0
    eval_paths, total_reward, collisions = run_episode(vehicles, grid, delays, agents, train=False)

    # Compute steps taken (max path length over vehicles)
    steps = max(len(path) - 1 for path in eval_paths.values())  # -1 because we start at time 0

    return total_reward, collisions, steps


def main():
    # Episode counts to test
    episode_counts = [50, 100, 200, 300, 500, 750, 1000]
    rewards = []
    collisions = []
    steps_taken = []

    print("Running experiments for different episode counts...")
    print(f"Config: sensor={TEST_CONFIG[0]}, brake={TEST_CONFIG[1]}, comm={TEST_CONFIG[2]}")
    print("-" * 50)

    for episodes in episode_counts:
        print(f"Training {episodes} episodes...")
        reward, coll, steps = run_experiment(episodes, TEST_CONFIG)
        rewards.append(reward)
        collisions.append(coll)
        steps_taken.append(steps)
        print(f"  Reward: {reward:.2f}, Collisions: {coll}, Steps: {steps}")

    # Create output folder
    output_dir = os.path.join("results", "episodic_return")
    os.makedirs(output_dir, exist_ok=True)

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(f'Effect of Training Episodes on Performance\n'
                 f'Disturbance: Sensor={TEST_CONFIG[0]}, Brake={TEST_CONFIG[1]}, Comm={TEST_CONFIG[2]}',
                 fontsize=14)

    axes[0].plot(episode_counts, rewards, marker='o', color='black')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(episode_counts, collisions, marker='s', color='black')
    axes[1].set_ylabel('Collisions')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(episode_counts, steps_taken, marker='^', color='black')
    axes[2].set_xlabel('Training Episodes')
    axes[2].set_ylabel('Steps Taken')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"episode_analysis_s{TEST_CONFIG[0]}b{TEST_CONFIG[1]}c{TEST_CONFIG[2]}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.show()

    print(f"\nPlot saved as '{filepath}'.")


if __name__ == "__main__":
    main()