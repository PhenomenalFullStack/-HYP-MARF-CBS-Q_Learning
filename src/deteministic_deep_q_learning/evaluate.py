"""
evaluate.py

Evaluate a trained DDPG agent across sensor-noise levels and
compare against baselines.
"""

import os
import csv
import numpy as np

import config
from environment import ContinuousIntersectionEnv
from ddpg_agent import DDPGAgent
from utils import plot_noise_curve, plot_comparison


def load_cbs_schedule():
    path = os.path.join(config.LOG_DIR, "cbs_schedule.npy")
    if os.path.exists(path):
        try:
            return np.load(path, allow_pickle=True).item()
        except Exception:
            return None
    return None


def evaluate_agent(agent, cbs_schedule, noise_level, episodes):
    env = ContinuousIntersectionEnv(
        config, cbs_schedule=cbs_schedule, noise_level=noise_level
    )

    total_collisions = 0
    total_steps = 0
    total_goals = 0
    schedule_devs = []
    rewards = []

    for _ in range(episodes):
        states = env.reset()
        ep_reward = 0.0
        for step in range(config.MAX_STEPS):
            actions = np.stack([
                agent.select_action(states[i], add_noise=False)
                for i in range(config.NUM_VEHICLES)
            ])
            next_states, r, done, info = env.step(actions)
            ep_reward += float(np.mean(r))
            total_collisions += len(info["collisions"])
            total_steps += 1

            if cbs_schedule is not None:
                for i in range(config.NUM_VEHICLES):
                    sched = env._get_scheduled_position(i)
                    if sched is not None:
                        schedule_devs.append(
                            float(np.linalg.norm(env.positions[i] - sched))
                        )

            states = next_states
            if done:
                break

        total_goals += int(info["all_reached"])
        rewards.append(ep_reward)

    return {
        "noise_level": noise_level,
        "collision_rate": total_collisions / max(total_steps, 1),
        "goal_rate": total_goals / episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_deviation": float(np.mean(schedule_devs))
                          if schedule_devs else 0.0,
    }


def main():
    cbs_schedule = load_cbs_schedule()

    agent = DDPGAgent(config.STATE_DIM, config.ACTION_DIM, config)
    model_path = os.path.join(config.MODEL_DIR, "best")
    if os.path.exists(f"{model_path}_actor.pth"):
        agent.load(model_path)
        print("[info] Loaded best model.")
    else:
        print("[warn] No trained model — evaluating random policy.")

    results = []
    for noise in config.SENSOR_NOISE_LEVELS:
        res = evaluate_agent(
            agent, cbs_schedule, noise, config.EVAL_EPISODES
        )
        results.append(res)
        print(
            f"[noise={noise:.1f}] "
            f"collision_rate={res['collision_rate']:.3f} "
            f"goal_rate={res['goal_rate']:.2f} "
            f"reward={res['mean_reward']:.2f} "
            f"deviation={res['mean_deviation']:.3f}"
        )

    out_csv = os.path.join(config.LOG_DIR, "eval_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[done] Saved evaluation to {out_csv}")

    plot_noise_curve(results, config.RESULT_DIR)

    baseline_path = os.path.join(config.LOG_DIR, "baseline_results.csv")
    if os.path.exists(baseline_path):
        plot_comparison(results, baseline_path, config.RESULT_DIR)


if __name__ == "__main__":
    main()