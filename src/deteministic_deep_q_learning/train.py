"""
train.py

Training loop with noise decay, curriculum support and periodic noise-free evaluation.
"""

import os
import csv
import time
import random
import numpy as np
import torch

import config
from environment import ContinuousIntersectionEnv
from ddpg_agent import DDPGAgent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cbs_schedule():
    path = os.path.join(config.LOG_DIR, "cbs_schedule.npy")
    if os.path.exists(path):
        try:
            return np.load(path, allow_pickle=True).item()
        except Exception as e:
            print(f"[warn] Could not load CBS schedule: {e}")
    else:
        # The path is anchored to config.py's folder, not the working directory
        print(f"[info] Looked for the CBS schedule at: {path}")
    return None


def noise_scale_for_episode(episode):
    """Linear decay from 1.0 -> NOISE_MIN_SCALE between start and end."""
    s, e = config.NOISE_DECAY_START, config.NOISE_DECAY_END
    if episode <= s:
        return 1.0
    if episode >= e:
        return config.NOISE_MIN_SCALE
    frac = (episode - s) / (e - s)
    return 1.0 - frac * (1.0 - config.NOISE_MIN_SCALE)


def evaluate(agent, env, n_episodes):
    """Noise-free rollouts. Returns (success_rate, crash_rate, mean_episode_reward)."""
    # Solo random-slot stage: evaluate every slot once (one deterministic episode each).
    # Otherwise the eval is deterministic (no sensor noise), so `n_episodes` copies are identical.
    if env.num_vehicles == 1 and getattr(config, "SOLO_RANDOM_SLOT", False):
        slots = list(range(config.MAX_VEHICLES))
    else:
        slots = [None] * n_episodes

    successes, crashes, rewards = 0, 0, []
    for slot in slots:
        states = env.reset(slot=slot)      # an explicit slot never draws from the RNG
        ep_reward, crashed, done = 0.0, False, False
        while not done:
            actions = np.stack([
                agent.select_action(states[i], add_noise=False)   # deterministic policy
                for i in range(env.num_vehicles)
            ])
            states, r, done, info = env.step(actions)
            ep_reward += float(np.mean(r))    # sum of per-step means == mean of per-vehicle sums
            crashed = crashed or len(info["collisions"]) > 0
        successes += int(info["all_reached"])
        crashes += int(crashed)
        rewards.append(ep_reward)
    n = len(rewards)
    return successes / n, crashes / n, float(np.mean(rewards))


def train():
    set_seed(config.SEED)
    print(f"[info] Device: {config.DEVICE}")
    print(f"[info] NUM_VEHICLES = {config.NUM_VEHICLES} "
          f"(curriculum stage)")
    if config.NUM_VEHICLES == 1 and getattr(config, "SOLO_RANDOM_SLOT", False):
        print("[info] Solo stage drives a RANDOM start/goal slot each episode.")

    cbs_schedule = load_cbs_schedule()
    if cbs_schedule is None:
        print("[info] No CBS schedule found — schedule reward disabled.")

    env = ContinuousIntersectionEnv(
        config, cbs_schedule=cbs_schedule, noise_level=0.0
    )
    # Separate instance so evaluation can never disturb the training episode state
    eval_env = ContinuousIntersectionEnv(
        config, cbs_schedule=cbs_schedule, noise_level=0.0
    )
    agent = DDPGAgent(config.STATE_DIM, config.ACTION_DIM, config)

    # With no sensor noise, both the env and the noise-free policy are deterministic,
    # so repeating the identical episode EVAL_EPISODES times adds nothing.
    n_eval = config.EVAL_EPISODES if eval_env.noise_level > 0 else 1

    log_path = os.path.join(config.LOG_DIR, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "episode", "ep_reward", "ep_collisions",
        "ep_steps", "critic_loss", "actor_loss",
        "noise_scale", "elapsed_s",
        "ep_reached", "q_mean"            # new columns appended at the end
    ])

    start_time = time.time()
    best_key = None    # (success_rate, -crash_rate, mean_reward) of the best evaluation so far

    for episode in range(1, config.MAX_EPISODES + 1):
        states = env.reset()
        agent.reset_noise()
        nscale = noise_scale_for_episode(episode)

        ep_reward = np.zeros(config.NUM_VEHICLES)
        ep_collisions = 0     # vehicle-collisions: a two-car crash on one step counts 2
        critic_losses, actor_losses = [], []

        for step in range(config.MAX_STEPS):
            actions = np.stack([
                agent.select_action(
                    states[i], noise_scale=nscale, vehicle_id=i   # independent noise per vehicle
                )
                for i in range(config.NUM_VEHICLES)
            ])

            next_states, rewards, done, info = env.step(actions)
            ep_collisions += len(info["collisions"])

            n_stored = 0
            for i in range(config.NUM_VEHICLES):
                if not info["active"][i]:
                    continue    # parked vehicle: nothing to learn from
                # Terminal for THIS vehicle if the episode truly ended (all arrived / crash) or it just
                # reached its goal. The 100-step time limit is deliberately NOT terminal.
                done_i = float(info["terminated"] or info["newly_reached"][i])
                agent.store(
                    states[i], actions[i], rewards[i],
                    next_states[i], done_i
                )
                n_stored += 1

            # One gradient step per stored transition (a standard 1:1 update-to-data ratio).
            # Set this to range(1) if you want the old 1-update-per-env-step behaviour.
            for _ in range(n_stored):
                cl, al = agent.update()
                if cl is not None:
                    critic_losses.append(cl)
                    actor_losses.append(al)

            ep_reward += rewards
            states = next_states
            if done:
                break

        mean_reward = float(np.mean(ep_reward))
        mean_cl = float(np.mean(critic_losses)) if critic_losses else 0.0
        mean_al = float(np.mean(actor_losses)) if actor_losses else 0.0
        reached = int(info["reached"].sum())
        elapsed = time.time() - start_time

        writer.writerow([
            episode, mean_reward, ep_collisions,
            step + 1, mean_cl, mean_al, nscale, round(elapsed, 2),
            reached, round(agent.last_q_mean, 3)
        ])
        log_file.flush()

        if episode % config.PRINT_EVERY == 0:
            # critic/actor loss print 0.0 until the warm-up data exists (no updates yet)
            print(
                f"[ep {episode:4d}] reward={mean_reward:8.2f} "
                f"reached={reached}/{config.NUM_VEHICLES} "
                f"collisions={ep_collisions:3d} "
                f"steps={step+1:3d} "
                f"critic_loss={mean_cl:8.4f} actor_loss={mean_al:8.4f} "
                f"Q={agent.last_q_mean:7.2f} "
                f"noise={nscale:.2f} time={elapsed:.1f}s"
            )
            # Failure diagnostic: where did a failed episode end? (stalled vs. wandering vs. crashed)
            if reached < config.NUM_VEHICLES:
                print(f"    [fail] dist_to_goal={np.round(info['distances'], 2).tolist()} "
                      f"speed={np.round(np.linalg.norm(env.velocities, axis=1), 2).tolist()}")

        # Noise-free evaluation; "best" = highest success rate, then fewest crashes, then reward
        if episode % config.EVAL_EVERY == 0:
            sr, cr, er = evaluate(agent, eval_env, n_eval)
            print(f"[eval {episode:4d}] success={sr:.2f} "
                  f"crash={cr:.2f} reward={er:8.2f}")
            key = (sr, -cr, round(er, 1))          # rounded so genuinely equal evals compare equal
            if best_key is None or key >= best_key:   # >= : among equally good evals keep the LATEST model
                best_key = key
                agent.save(os.path.join(config.MODEL_DIR, "best"))

        if episode % config.SAVE_EVERY == 0:
            agent.save(os.path.join(config.MODEL_DIR, f"ep{episode}"))

    log_file.close()
    if best_key is not None:
        print(f"[done] Training complete. Best eval: success={best_key[0]:.2f} "
              f"crash={-best_key[1]:.2f} reward={best_key[2]:.2f}")
    else:
        print("[done] Training complete.")


if __name__ == "__main__":
    train()