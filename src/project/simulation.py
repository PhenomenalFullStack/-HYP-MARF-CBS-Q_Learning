# simulation.py

import json
import numpy as np
from grid import Grid
from vehicle import Vehicle
from schedule import Schedule
from cbs_planner import CBSPlanner
from q_learning import QLearningAgent

# Constants
GRID_SIZE = 10
MAX_STEPS = 50
TRAINING_EPISODES = 500

# Scenario Generation
def generate_scenario(num_vehicles, width=10, height=10, seed=None):
    """Generate starts and goals for 1..8 vehicles."""
    if seed is not None:
        np.random.seed(seed)

    base_ids = ["A", "B", "C", "D"]

    sides = [
        # East edge: lane=row 5, start_depth=9, goal_depth=0
        {"lane_axis": "row", "lane_value": 5, "start_depth_base": 9, "start_depth_dir": -1, "goal_depth_base": 0},
        # South edge: lane=col 4, start_depth=9, goal_depth=0
        {"lane_axis": "col", "lane_value": 4, "start_depth_base": 9, "start_depth_dir": -1, "goal_depth_base": 0},
        # West edge: lane=row 4, start_depth=0, goal_depth=9
        {"lane_axis": "row", "lane_value": 4, "start_depth_base": 0, "start_depth_dir": 1, "goal_depth_base": 9},
        # North edge: lane=col 5, start_depth=1, goal_depth=9
        {"lane_axis": "col", "lane_value": 5, "start_depth_base": 1, "start_depth_dir": 1, "goal_depth_base": 9},
    ]

    # Build occupancy queues for each side
    occupants = [list() for _ in range(4)]
    if num_vehicles <= 4:
        for i in range(num_vehicles):
            occupants[i].append(base_ids[i])
    else:
        # Add base vehicles first one per side
        for i in range(4):
            occupants[i].append(base_ids[i])
        # Add extra vehicles, pushing existing ones inward
        extra = num_vehicles - 4
        next_id_ord = ord("E")
        side_idx = 0
        for _ in range(extra):
            vid = chr(next_id_ord)
            next_id_ord += 1
            occupants[side_idx].insert(0, vid)   # insert at front, pushing older ones inward
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

# Helper functions
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

# Episode runner
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
        # Scheduled positions
        scheduled_current = {}
        for vid in vehicle_ids:
            v = [v for v in vehicles if v.vehicle_id == vid][0]
            eff_idx = step - delay[vid]
            scheduled_current[vid] = get_position_on_path(v, eff_idx)

        # Actions
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

        # Delay update
        for vid in vehicle_ids:
            if actions[vid] == 1:   # wait
                delay[vid] += 1
            elif actions[vid] == 2: # skip
                if delay[vid] > 0:
                    delay[vid] -= 1

        # New positions
        new_positions = {}
        new_indices = {}
        for vid in vehicle_ids:
            v = [v for v in vehicles if v.vehicle_id == vid][0]
            current_idx = path_index[vid]
            if actions[vid] == 0:      # forward
                new_idx = current_idx + 1
            elif actions[vid] == 2:    # skip
                new_idx = current_idx + 2
            else:                      # wait
                new_idx = current_idx
            new_idx = min(new_idx, len(v.path) - 1) if v.path else current_idx
            new_pos = get_position_on_path(v, new_idx)
            new_positions[vid] = new_pos
            new_indices[vid] = new_idx

        # Collisions
        collision_flags = {vid: False for vid in vehicle_ids}
        collision_count = 0
        for i, vid1 in enumerate(vehicle_ids):
            for vid2 in vehicle_ids[i+1:]:
                if new_positions[vid1] == new_positions[vid2]:
                    collision_flags[vid1] = True
                    collision_flags[vid2] = True
                    collision_count += 1

        # Move and update
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

# Main training loop over vehicle counts
def main():
    vehicle_counts = list(range(1, 9))   # 1 to 8 vehicles
    all_results = []

    for num_vehicles in vehicle_counts:
        print(f"\n{'='*60}")
        print(f"Running for {num_vehicles} vehicles")
        print(f"{'='*60}")

        # Generate scenario
        starts, goals = generate_scenario(num_vehicles)
        vehicle_ids = sorted(starts.keys())

        # Create grid and vehicles
        grid = Grid(GRID_SIZE, GRID_SIZE)
        vehicles = [Vehicle(vid, starts[vid], goals[vid]) for vid in vehicle_ids]

        # CBS planning
        planner = CBSPlanner(grid, vehicles)
        schedules = planner.plan(starts, goals)
        if schedules is None:
            print(f"CBS failed for {num_vehicles} vehicles, skipping.")
            continue
        distribute_schedules(vehicles, schedules)

        # Extract CBS schedule for JSON and printing
        cbs_schedule_data = {}
        for sch in schedules:
            cbs_schedule_data[sch.vehicle_id] = [{"time": t, "position": pos} for pos, t in sch.steps]

        # Disturbance configurations
        configs = [(sn, bd, cl) for sn in [0,1] for bd in [0,1] for cl in [0,1]]

        for sn, bd, cl in configs:
            config_name = f"sensorNoise{sn}_brakingDelay{bd}_commLatency{cl}"
            print(f"\n  Training config: Sensor={sn}, Brake={bd}, Comm={cl}")

            # Delays: only the first vehicle gets the sum
            delays = {vid: 0 for vid in vehicle_ids}
            delays[vehicle_ids[0]] = sn + bd + cl

            # Create agents
            agents = {v.vehicle_id: QLearningAgent(v, actions=3, lr=0.1, gamma=0.9, epsilon=1.0)
                      for v in vehicles}

            # Training
            for ep in range(TRAINING_EPISODES):
                run_episode(vehicles, grid, delays, agents, vehicle_ids, train=True)
                for agent in agents.values():
                    agent.decay_epsilon()

            # Evaluation (greedy)
            for agent in agents.values():
                agent.epsilon = 0.0
            eval_paths, eval_collisions, _ = run_episode(vehicles, grid, delays, agents, vehicle_ids, train=False)

            # Build actual schedule
            actual_schedule = {}
            for vid in vehicle_ids:
                steps = [{"time": t, "position": pos} for t, pos in enumerate(eval_paths[vid])]
                actual_schedule[vid] = steps

            # Store result
            entry = {
                "num_vehicles": num_vehicles,
                "configuration": {"sensor_noise": sn, "braking_delay": bd, "comm_latency": cl},
                "cbs_schedule": cbs_schedule_data,
                "actual_schedule": actual_schedule,
                "collisions": eval_collisions,
                "training_episodes": TRAINING_EPISODES,
            }
            all_results.append(entry)

            # PRINT schedules
            print(f"\n CBS Schedule (Best):")
            for vid, steps in cbs_schedule_data.items():
                print(f"Vehicle {vid}: {steps}")
            print(f"\n Actual Schedule after Q‑learning ({config_name}):")
            for vid, steps in actual_schedule.items():
                print(f"Vehicle {vid}: {steps}")

    # Save all results to the original JSON filename
    with open("simulation_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll results saved to simulation_results.json (total {len(all_results)} entries).")

if __name__ == "__main__":
    main()