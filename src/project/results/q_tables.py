# q_tables.py
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from grid import Grid
from vehicle import Vehicle
from cbs_planner import CBSPlanner
from q_learning import QLearningAgent

from simulation import (
    GRID_SIZE, STARTS, GOALS, VEHICLE_IDS, TRAINING_EPISODES,
    distribute_schedules, get_position_on_path, run_episode
)

Q_TABLE_ROOT = "q_tables"


def main():
    grid = Grid(GRID_SIZE, GRID_SIZE)
    vehicles = [Vehicle(vid, STARTS[vid], GOALS[vid]) for vid in VEHICLE_IDS]

    planner = CBSPlanner(grid, vehicles)
    schedules = planner.plan(STARTS, GOALS)
    if schedules is None:
        print("CBS failed, exiting.")
        return
    distribute_schedules(vehicles, schedules)

    # ground-truth CBS positions
    cbs_positions = {v.vehicle_id: v.path for v in vehicles}

    configs = [(sn, bd, cl) for sn in [0, 1] for bd in [0, 1] for cl in [0, 1]]

    for sn, bd, cl in configs:
        config_name = f"sensorNoise{sn}_brakingDelay{bd}_commLatency{cl}"
        print(f"\nTraining: {config_name}")

        delays = {vid: 0 for vid in VEHICLE_IDS}
        delays['A'] = sn + bd + cl

        agents = {v.vehicle_id: QLearningAgent(v, actions=3, lr=0.1, gamma=0.9, epsilon=1.0)
                  for v in vehicles}

        for ep in range(TRAINING_EPISODES):
            run_episode(vehicles, grid, delays, agents, train=True)
            for agent in agents.values():
                agent.decay_epsilon()

        # save the learned table for every vehicle under this config
        qtable_dir = os.path.join(Q_TABLE_ROOT, config_name)
        os.makedirs(qtable_dir, exist_ok=True)
        for vid, agent in agents.items():
            agent.save_q_table(os.path.join(qtable_dir, f"{vid}.json"))
        print(f"  saved -> {qtable_dir}/")

        # greedy eval match CBS exactly
        for agent in agents.values():
            agent.epsilon = 0.0
        eval_paths, _ = run_episode(vehicles, grid, delays, agents, train=False)
        for vid in VEHICLE_IDS:
            match = eval_paths[vid][:len(cbs_positions[vid])] == cbs_positions[vid]
            print(f"  {vid}: matches CBS = {match}")

    print(f"\nAll Q-tables saved under ./{Q_TABLE_ROOT}/<config_name>/<vehicle_id>.json")
    print("Run `python view_q_tables.py` to inspect them.")


if __name__ == "__main__":
    main()