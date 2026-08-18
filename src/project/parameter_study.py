# parameter_study.py

import itertools
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from grid import Grid
from vehicle import Vehicle
from schedule import Schedule
from cbs_planner import CBSPlanner
from q_learning import QLearningAgent

GRID_SIZE = 10
MAX_STEPS = 50

OUTPUT_DIR = os.path.join("results", "parameter_study")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_scenario(num_vehicles, seed=None):
    base_ids = ["A", "B", "C", "D"]
    sides = [
        {"lane_axis": "row", "lane_value": 5, "start_depth_base": 9, "start_depth_dir": -1, "goal_depth_base": 0},
        {"lane_axis": "col", "lane_value": 4, "start_depth_base": 9, "start_depth_dir": -1, "goal_depth_base": 0},
        {"lane_axis": "row", "lane_value": 4, "start_depth_base": 0, "start_depth_dir": 1, "goal_depth_base": 9},
        {"lane_axis": "col", "lane_value": 5, "start_depth_base": 1, "start_depth_dir": 1, "goal_depth_base": 9},
    ]
    occupants = [list() for _ in range(4)]
    if num_vehicles <= 4:
        for i in range(num_vehicles):
            occupants[i].append(base_ids[i])
    else:
        for i in range(4):
            occupants[i].append(base_ids[i])
        extra = num_vehicles - 4
        next_id_ord = ord("E")
        side_idx = 0
        for _ in range(extra):
            vid = chr(next_id_ord)
            next_id_ord += 1
            occupants[side_idx].insert(0, vid)
            side_idx = (side_idx + 1) % 4

    starts = {}
    goals = {}
    for side, queue in zip(sides, occupants):
        if not queue:
            continue
        lane_axis = side["lane_axis"]
        lane_value = side["lane_value"]
        start_base = side["start_depth_base"]
        start_dir = side["start_depth_dir"]
        goal_base = side["goal_depth_base"]
        goal_dir = -start_dir
        for offset, vid in enumerate(queue):
            start_depth = start_base + offset * start_dir
            goal_depth = goal_base + offset * goal_dir
            if lane_axis == "row":
                starts[vid] = (lane_value, start_depth)
                goals[vid] = (lane_value, goal_depth)
            else:
                starts[vid] = (start_depth, lane_value)
                goals[vid] = (goal_depth, lane_value)
    return starts, goals

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

def run_episode(vehicles, grid, delays, agents, vehicle_ids, time_limit=MAX_STEPS, train=True):
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
            state = agents[vid].build_state_key(obs, scheduled_current[vid], v.current_position, delay[vid])
            if train:
                action = agents[vid].select_action(obs, scheduled_current[vid], v.current_position, delay[vid])
            else:
                q_vals = agents[vid].get_q_values(state)
                action = max(range(len(q_vals)), key=lambda i: q_vals[i])
            actions[vid] = action
            if train:
                prev_states[vid] = state
                prev_actions[vid] = action

        for vid in vehicle_ids:
            if actions[vid] == 1:
                delay[vid] += 1
            elif actions[vid] == 2:
                if delay[vid] > 0:
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
            for vid2 in vehicle_ids[i+1:]:
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

            if train:
                true_next_sched_pos = get_position_on_path(v, step + 1)
                noisy_next_sched_pos = get_position_on_path(v, (step + 1) - delay[vid])
                reward = agents[vid].compute_reward(
                    old_pos,
                    new_positions[vid],
                    true_next_sched_pos,
                    delay[vid],
                    collision_flags[vid],
                    v.goal
                )
                total_reward += reward

                next_obs = v.get_observation(grid)
                next_state = agents[vid].build_state_key(next_obs, noisy_next_sched_pos, new_positions[vid], delay[vid])
                agents[vid].update_q_table(prev_states[vid], prev_actions[vid], reward, next_state)

        done = all(v.has_reached_goal() for v in vehicles)
        step += 1

    return paths, total_reward, collision_count

def run_experiment(num_vehicles, config, lr, gamma, initial_epsilon, epsilon_decay, training_episodes):
    sn, bd, cl = config
    starts, goals = generate_scenario(num_vehicles)
    vehicle_ids = sorted(starts.keys())
    grid = Grid(GRID_SIZE, GRID_SIZE)
    vehicles = [Vehicle(vid, starts[vid], goals[vid]) for vid in vehicle_ids]

    planner = CBSPlanner(grid, vehicles)
    schedules = planner.plan(starts, goals)
    if schedules is None:
        return None
    distribute_schedules(vehicles, schedules)

    cbs_steps = 0
    for sch in schedules:
        cbs_steps = max(cbs_steps, len(sch.steps)-1)

    delays = {vid: 0 for vid in vehicle_ids}
    delays[vehicle_ids[0]] = sn + bd + cl

    agents = {}
    for v in vehicles:
        agent = QLearningAgent(v, actions=3, lr=lr, gamma=gamma, epsilon=initial_epsilon)
        agent.epsilon_decay = epsilon_decay
        agents[v.vehicle_id] = agent

    for ep in range(training_episodes):
        run_episode(vehicles, grid, delays, agents, vehicle_ids, train=True)
        for agent in agents.values():
            agent.decay_epsilon(min_eps=0.01, decay=epsilon_decay)

    for agent in agents.values():
        agent.epsilon = 0.0
    eval_paths, total_reward, collisions = run_episode(vehicles, grid, delays, agents, vehicle_ids, train=False)
    steps = max(len(p)-1 for p in eval_paths.values())

    return {
        'num_vehicles': num_vehicles,
        'config': config,
        'lr': lr,
        'gamma': gamma,
        'initial_epsilon': initial_epsilon,
        'epsilon_decay': epsilon_decay,
        'training_episodes': training_episodes,
        'collisions': collisions,
        'steps': steps,
        'cbs_steps': cbs_steps,
        'delay': steps - cbs_steps,
        'reward': total_reward
    }

def plot_worst_best_comparison(all_results):
    zero_coll = [r for r in all_results if r['collisions'] == 0]
    if not zero_coll:
        print("No zero-collision results found.")
        return

    vehicle_numbers = sorted(set(r['num_vehicles'] for r in zero_coll))
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()

    for idx, nv in enumerate(vehicle_numbers):
        subset = [r for r in zero_coll if r['num_vehicles'] == nv]
        if not subset:
            continue

        best = min(subset, key=lambda x: x['steps'])
        worst = max(subset, key=lambda x: x['steps'])

        ax = axes[idx]
        categories = ['Steps']
        best_vals = [best['steps']]
        worst_vals = [worst['steps']]

        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, worst_vals, width, label='Worst', color='red')
        ax.bar(x + width/2, best_vals, width, label='Best', color='green')

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_title(f'Vehicles = {nv}\nWorst vs Best')
        ax.legend()

        y_max = max(worst_vals[0], best_vals[0]) + 5
        ax.set_ylim(0, y_max)

        worst_text = (f"collisions={worst['collisions']}\nEpsilon={worst['initial_epsilon']}\n"
                      f"Epsilon Decay={worst['epsilon_decay']}\ngamma={worst['gamma']}\n"
                      f"lr={worst['lr']}\nEpisodes={worst['training_episodes']}")
        best_text = (f"collisions={best['collisions']}\nEpsilon={best['initial_epsilon']}\n"
                     f"Epsilon Decay={best['epsilon_decay']}\ngamma={best['gamma']}\n"
                     f"lr={best['lr']}\nEpisodes={best['training_episodes']}")

        ax.text(-width/2, worst_vals[0] + 1, worst_text, ha='center', va='bottom', fontsize=8, color='red')
        ax.text(width/2, best_vals[0] + 1, best_text, ha='center', va='bottom', fontsize=8, color='green')

    for j in range(len(vehicle_numbers), 8):
        axes[j].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'worst_best_comparison.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Worst vs Best comparison plot saved to {plot_path}")

def main():
    param_space = {
        'initial_epsilon': [0.5, 1.0],
        'lr': [0.05, 0.1, 0.2],
        'gamma': [0.85, 0.9, 0.95],
        'epsilon_decay': [0.99, 0.995, 0.999],
        'training_episodes': [200, 500, 1000]
    }
    vehicle_counts = list(range(1, 9))
    configs = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]

    all_results = []
    total_combos = len(vehicle_counts) * len(configs) * len(list(itertools.product(*param_space.values())))
    print(f"Total experiments to run: {total_combos}")
    print("Starting parameter study...")

    for num_veh in vehicle_counts:
        print(f"\n Testing {num_veh} vehicles ")
        for config in configs:
            print(f"  Config: sensor={config[0]}, brake={config[1]}, comm={config[2]}")
            for combo in itertools.product(
                param_space['initial_epsilon'],
                param_space['lr'],
                param_space['gamma'],
                param_space['epsilon_decay'],
                param_space['training_episodes']
            ):
                initial_eps, lr, gamma, eps_decay, train_eps = combo
                print(f"    Running learning rate={lr}, discount factor={gamma}, initial epsilon={initial_eps}, epsilon decay={eps_decay}, training episodes={train_eps}")
                result = run_experiment(num_veh, config, lr, gamma, initial_eps, eps_decay, train_eps)
                if result:
                    all_results.append(result)
                    print(f"       collisions={result['collisions']}, steps={result['steps']}")

    if all_results:
        csv_path = os.path.join(OUTPUT_DIR, 'parameter_study_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {csv_path}")

        plot_worst_best_comparison(all_results)

        best_per_key = {}
        for r in all_results:
            key = (r['num_vehicles'], r['config'])
            score = (r['steps'], r['collisions'], -r['reward'])
            if key not in best_per_key or score < (best_per_key[key]['steps'], best_per_key[key]['collisions'], -best_per_key[key]['reward']):
                best_per_key[key] = r

        print("\n")
        print("BEST PARAMETERS per (Num Vehicles, Config) minimising steps and collisions:")
        print("\n")
        for key, best in sorted(best_per_key.items()):
            num_veh, (sn, bd, cl) = key
            print(f"\nnum_vehicles={num_veh}, sensor={sn}, brake={bd}, comm={cl}")
            print(f"  learning_rate        = {best['lr']}")
            print(f"  discount_factor      = {best['gamma']}")
            print(f"  initial_epsilon      = {best['initial_epsilon']}")
            print(f"  epsilon_decay        = {best['epsilon_decay']}")
            print(f"  training_episodes    = {best['training_episodes']}")
            print(f"  steps                = {best['steps']}")
            print(f"  collisions           = {best['collisions']}")
    else:
        print("No results collected.")

if __name__ == "__main__":
    main()