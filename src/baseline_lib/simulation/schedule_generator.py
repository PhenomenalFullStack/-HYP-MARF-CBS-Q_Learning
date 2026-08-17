# schedule_generator.py

import os
import sys
import json
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "cbs"))

from grid_env import GridEnv
from scenario_generator import generate_scenario
from cbs_planner import CBSPlanner
from disturbance_utils import compute_delay_steps, pick_disturbed_vehicle

NUM_VEHICLES_RANGE = list(range(1, 9))  # 1 to 8 vehicles
DISTURBANCE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # applied to each parameter

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "results",
    "intersection_schedules.json",
)

# Generate a unique key for each configuration
def config_key(num_vehicles, sensor_noise, comm_latency, braking_delay):
    return f"nv{num_vehicles}_sn{sensor_noise}_cl{comm_latency}_bd{braking_delay}"

# Run CBS for a specific configuration
def run_cbs_for_config(
    num_vehicles, sensor_noise, comm_latency, braking_delay, trials=3
):
    starts, goals = generate_scenario(num_vehicles)
    vehicle_ids = sorted(starts.keys())

    delay_steps = compute_delay_steps(sensor_noise, comm_latency, braking_delay)
    disturbed_vid, has_follower = pick_disturbed_vehicle(vehicle_ids, starts, goals)
    delay_map = {
        vid: (delay_steps if vid == disturbed_vid else 0) for vid in vehicle_ids
    }

    # Plan using CBS
    plan_env = GridEnv(width=10, height=10)
    for vid in vehicle_ids:
        plan_env.add_vehicle(vid, starts[vid], goals[vid], delay_steps=delay_map[vid])

    planner = CBSPlanner(plan_env, vehicle_ids)
    schedules = planner.plan(starts, goals)
    plan_found = schedules is not None

    if schedules is not None:
        schedules_dict = {s.vehicle_id: s.steps for s in schedules}
    else:
        schedules_dict = {}
        for vid in vehicle_ids:
            path = planner.find_path_for_vehicle(
                starts[vid], goals[vid], [], delay_map[vid]
            )
            schedules_dict[vid] = path if path is not None else [(0, starts[vid])]

    # Execute the schedule with disturbances
    best = None
    for _ in range(trials):
        exec_env = GridEnv(
            width=10,
            height=10,
            braking_delay_prob=braking_delay,
            sensor_noise=sensor_noise,
            comm_latency=comm_latency,
        )
        for vid in vehicle_ids:
            exec_env.add_vehicle(
                vid, starts[vid], goals[vid], delay_steps=delay_map[vid]
            )

        # Record the starting grid state
        grid_states = [
            {
                vid: {
                    "row": exec_env.vehicles[vid]["current"][0],
                    "column": exec_env.vehicles[vid]["current"][1],
                }
                for vid in vehicle_ids
            }
        ]

        done = False
        step_count = 0

        while not done and step_count < 100:
            done, _ = exec_env.step_from_schedule(schedules_dict)
            step_count += 1
            grid_states.append(
                {
                    vid: {
                        "row": exec_env.vehicles[vid]["current"][0],
                        "column": exec_env.vehicles[vid]["current"][1],
                    }
                    for vid in vehicle_ids
                }
            )

        # Keep the trial with the fewest collisions
        if best is None or exec_env.collisions < best["collisions"]:
            best = {
                "collisions": exec_env.collisions,
                "steps": step_count,
                "vehicles_reached_goal": sum(
                    1 for v in exec_env.vehicles.values() if v["reached"]
                ),
                "vehicles_total": num_vehicles,
                "grid_states": grid_states,
                "schedule": {
                    vid: [
                        {"time": t, "row": p[0], "column": p[1]}
                        for t, p in schedules_dict[vid]
                    ]
                    for vid in vehicle_ids
                },
                "disturbed_vehicle": disturbed_vid,
                "disturbed_vehicle_has_follower": has_follower,
                "delay_steps": delay_steps,
                "plan_found": plan_found,
            }

    return best

# Main function to run the simulation
def main():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    all_results = {}
    start_time = time.time()
    total_combos = len(NUM_VEHICLES_RANGE) * len(DISTURBANCE_LEVELS) ** 3
    combo_count = 0

    for num_vehicles in NUM_VEHICLES_RANGE:
        print(f"\n=== num_vehicles = {num_vehicles} ===")

        for sensor_noise in DISTURBANCE_LEVELS:
            for comm_latency in DISTURBANCE_LEVELS:
                for braking_delay in DISTURBANCE_LEVELS:
                    combo_count += 1

                    if combo_count % 50 == 0:
                        elapsed = time.time() - start_time
                        print(
                            f"  [{combo_count}/{total_combos}] "
                            f"nv={num_vehicles} sn={sensor_noise} "
                            f"cl={comm_latency} bd={braking_delay} "
                            f"({elapsed:.1f}s elapsed)"
                        )

                    cbs_result = run_cbs_for_config(
                        num_vehicles, sensor_noise, comm_latency, braking_delay
                    )

                    key = config_key(
                        num_vehicles, sensor_noise, comm_latency, braking_delay
                    )
                    all_results[key] = {
                        "parameters": {
                            "num_vehicles": num_vehicles,
                            "sensor_noise": sensor_noise,
                            "comm_latency": comm_latency,
                            "braking_delay": braking_delay,
                            "delay_steps": compute_delay_steps(
                                sensor_noise, comm_latency, braking_delay
                            ),
                        },
                        "cbs": cbs_result,
                    }

        # Save progress after each vehicle count completes
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved {len(all_results)} configs so far -> {RESULTS_PATH}")

    print(f"\nDone. {len(all_results)} total configurations saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
