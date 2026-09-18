"""
ablation_study_8_vehicles.py: A script that changes one hyperparameter at a time (one factor at a time ablation), trains DQN, and records metrics.
State Space: 8 Vehicles, 10x10 grid.
"""

import os
import sys
import csv
import random
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(_HERE, "..", "..", "project")
DQN_PARENT = os.path.join(_HERE, "..")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DQN_PARENT)
sys.path.insert(0, _HERE)

import torch

from grid import Grid
from vehicle import Vehicle
from cbs_planner import CBSPlanner
from deep_q_learning_agent import DQNAgent

GRID_SIZE = 10
MAX_STEPS = 60
STATE_SIZE = 14
TRAINING_EPISODES = 800
SEED = 42

OUTPUT_DIR = os.path.join(_HERE, "results", "eight_vehicles")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STARTS = {
    "A": (5, 8),
    "B": (8, 4),
    "C": (4, 1),
    "D": (2, 5),
    "E": (5, 9),
    "F": (9, 4),
    "G": (4, 0),
    "H": (1, 5),
}

GOALS = {
    "A": (5, 1),
    "B": (1, 4),
    "C": (4, 8),
    "D": (8, 5),
    "E": (5, 0),
    "F": (0, 4),
    "G": (4, 9),
    "H": (9, 5),
}

VEHICLE_IDS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distribute_schedules(vehicles, schedules):
    for sch in schedules:
        for v in vehicles:
            if v.vehicle_id == sch.vehicle_id:
                v.set_schedule(sch)
                break


def reset_simulation(vehicles, grid):
    for r in range(grid.height):
        for c in range(grid.width):
            grid.cells[r][c] = "empty"
    grid.vehicles.clear()
    for v in vehicles:
        v.current_position = v.start
        grid.place_vehicle(v.vehicle_id, v.start)


def get_position_on_path(vehicle, index):
    if index < 0:
        return vehicle.start
    if index >= len(vehicle.path):
        return vehicle.goal
    return vehicle.path[index]


def run_episode(
    vehicles, grid, delays, agents, vehicle_ids, time_limit=MAX_STEPS, train=True
):
    reset_simulation(vehicles, grid)
    paths = {v.vehicle_id: [v.start] for v in vehicles}
    done = False
    step = 0
    total_reward = 0.0

    delay = {vid: delays.get(vid, 0) for vid in vehicle_ids}
    path_index = {vid: 0 for vid in vehicle_ids}
    prev_states = {}
    prev_actions = {}

    while not done and step < time_limit:
        scheduled_current = {}
        for vid in vehicle_ids:
            v = [v for v in vehicles if v.vehicle_id == vid][0]
            eff_idx = step - delay[vid]
            scheduled_current[vid] = get_position_on_path(v, eff_idx)

        actions = {}
        for vid in vehicle_ids:
            v = [v for v in vehicles if v.vehicle_id == vid][0]
            obs = v.get_observation(grid)
            allowed = [0, 1, 2] if delay[vid] > 0 else [0, 1]
            action = agents[vid].select_action(
                obs, scheduled_current[vid], v.current_position, delay[vid], allowed
            )
            actions[vid] = action
            if train:
                prev_states[vid] = agents[vid].build_state_key(
                    obs, scheduled_current[vid], v.current_position, delay[vid]
                )
                prev_actions[vid] = action

        for vid in vehicle_ids:
            if actions[vid] == 1:
                delay[vid] += 1
            elif actions[vid] == 2 and delay[vid] > 0:
                delay[vid] -= 1

        new_positions = {}
        new_indices = {}
        for vid in vehicle_ids:
            v = [v for v in vehicles if v.vehicle_id == vid][0]
            current_idx = path_index[vid]
            if actions[vid] == 0:
                new_idx = current_idx + 1
            elif actions[vid] == 2:
                new_idx = current_idx + 2
            else:
                new_idx = current_idx
            new_idx = min(new_idx, len(v.path) - 1) if v.path else current_idx
            new_pos = get_position_on_path(v, new_idx)
            new_positions[vid] = new_pos
            new_indices[vid] = new_idx

        collision_flags = {vid: False for vid in vehicle_ids}
        collision_count = 0
        for i, vid1 in enumerate(vehicle_ids):
            for vid2 in vehicle_ids[i + 1 :]:
                v1 = [v for v in vehicles if v.vehicle_id == vid1][0]
                v2 = [v for v in vehicles if v.vehicle_id == vid2][0]
                if v1.has_reached_goal() or v2.has_reached_goal():
                    continue
                if new_positions[vid1] == new_positions[vid2]:
                    collision_flags[vid1] = True
                    collision_flags[vid2] = True
                    collision_count += 1

        for vid in vehicle_ids:
            v = [v for v in vehicles if v.vehicle_id == vid][0]
            old_pos = v.current_position
            grid.move_vehicle(vid, new_positions[vid])
            v.update_position(new_positions[vid])
            paths[vid].append(new_positions[vid])
            path_index[vid] = new_indices[vid]

            true_next_sched_pos = get_position_on_path(v, step + 1)
            reward = agents[vid].compute_reward(
                old_pos,
                new_positions[vid],
                true_next_sched_pos,
                delay[vid],
                collision_flags[vid],
                v.goal,
            )
            total_reward += reward

            if train:
                next_obs = v.get_observation(grid)
                noisy_next_sched_pos = get_position_on_path(v, (step + 1) - delay[vid])
                next_state = agents[vid].build_state_key(
                    next_obs, noisy_next_sched_pos, new_positions[vid], delay[vid]
                )
                agents[vid].update_q_table(
                    prev_states[vid], prev_actions[vid], reward, next_state
                )

        done = all(v.has_reached_goal() for v in vehicles)
        step += 1

    return paths, total_reward, collision_count, done


def train_and_evaluate(agent_kwargs, config=(1, 1, 1), seed=SEED):
    set_seed(seed)
    sn, bd, cl = config
    grid = Grid(GRID_SIZE, GRID_SIZE)
    vehicles = [Vehicle(vid, STARTS[vid], GOALS[vid]) for vid in VEHICLE_IDS]

    planner = CBSPlanner(grid, vehicles)
    schedules = planner.plan(STARTS, GOALS)
    if schedules is None:
        return None
    distribute_schedules(vehicles, schedules)

    delays = {vid: 0 for vid in VEHICLE_IDS}
    delays["A"] = sn + bd + cl

    agents = {}
    for v in vehicles:
        agents[v.vehicle_id] = DQNAgent(
            vehicle=v, state_size=STATE_SIZE, action_size=3, **agent_kwargs
        )

    for ep in range(TRAINING_EPISODES):
        run_episode(vehicles, grid, delays, agents, VEHICLE_IDS, train=True)
        for agent in agents.values():
            agent.decay_epsilon()

    for agent in agents.values():
        agent.epsilon = 0.0
    eval_paths, eval_reward, eval_collisions, success = run_episode(
        vehicles, grid, delays, agents, VEHICLE_IDS, train=False
    )

    steps = max(len(p) - 1 for p in eval_paths.values())
    return {
        "collisions": eval_collisions,
        "reward": eval_reward,
        "steps": steps,
        "success": int(success),
    }


BASELINE = {
    "hidden_size": 128,
    "num_layers": 2,
    "activation": "relu",
    "loss_fn": "mse",
    "optimizer": "adam",
    "lr": 0.001,
    "gamma": 0.9,
    "epsilon_decay": 0.995,
    "batch_size": 64,
    "target_update": 100,
    "use_target_network": True,
    "use_replay": True,
    "double_dqn": False,
    "epsilon": 1.0,
}


def run_ablation():
    results = []

    print("\n[Baseline - 8 vehicles]")
    base_metrics = train_and_evaluate(BASELINE)
    row = dict(BASELINE)
    row.update(base_metrics)
    row["ablation"] = "baseline"
    row["varied"] = "baseline"
    results.append(row)
    print(f"  My Parameter Setup Results: {base_metrics}")

    sweeps = {
        "activation": ["relu", "leaky_relu", "tanh", "sigmoid"],
        "loss_fn": ["mse", "huber", "mae"],
        "optimizer": ["adam", "sgd", "rmsprop"],
        "hidden_size": [32, 64, 128, 256],
        "num_layers": [1, 2, 3],
        "lr": [0.0001, 0.0005, 0.001, 0.005],
        "gamma": [0.85, 0.9, 0.95, 0.99],
        "epsilon_decay": [0.99, 0.995, 0.999],
        "batch_size": [32, 64, 128],
        "target_update": [50, 100, 200],
        "use_target_network": [True, False],
        "use_replay": [True, False],
        "double_dqn": [False, True],
    }

    for key, values in sweeps.items():
        for val in values:
            if val == BASELINE[key]:
                continue
            kwargs = dict(BASELINE)
            kwargs[key] = val
            print(f"\n[{key} = {val}]")
            metrics = train_and_evaluate(kwargs)
            if metrics is None:
                continue
            row = dict(kwargs)
            row.update(metrics)
            row["ablation"] = key
            row["varied"] = val
            results.append(row)
            print(f"  Ablation Parameter Setup Results: {metrics}")

    csv_path = os.path.join(OUTPUT_DIR, "ablation_results_8_vehicles.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} runs to {csv_path}")


if __name__ == "__main__":
    run_ablation()
