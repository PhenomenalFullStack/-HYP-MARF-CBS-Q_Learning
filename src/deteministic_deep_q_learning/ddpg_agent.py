"""
ddpg_agent.py

DDPG agent with TD3-style stabilisers.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer
from noise import OUNoise


class DDPGAgent:
    def __init__(self, state_dim, action_dim, config):
        self.cfg = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(config.DEVICE)
        self.reward_scale = getattr(config, "REWARD_SCALE", 1.0)
        self.policy_delay = getattr(config, "POLICY_DELAY", 2)
        self.target_noise_std = getattr(config, "TARGET_NOISE_STD", 0.2)
        self.target_noise_clip = getattr(config, "TARGET_NOISE_CLIP", 0.5)

        # main actor
        self.actor = Actor(
            state_dim, action_dim,
            config.ACTION_LOW, config.ACTION_HIGH,
            hidden_dim=config.HIDDEN_DIM
        ).to(self.device)
        
        # slowly copying the actor
        self.actor_target = Actor(
            state_dim, action_dim,
            config.ACTION_LOW, config.ACTION_HIGH,
            hidden_dim=config.HIDDEN_DIM
        ).to(self.device)

        # Twin critics (Q1, Q2) and their targets
        self.critic = Critic(
            state_dim, action_dim, hidden_dim=config.HIDDEN_DIM
        ).to(self.device)
        self.critic2 = Critic(
            state_dim, action_dim, hidden_dim=config.HIDDEN_DIM
        ).to(self.device)
        
        # slow copy of Q1
        self.critic_target = Critic(
            state_dim, action_dim, hidden_dim=config.HIDDEN_DIM
        ).to(self.device)
        
        # slow copy of Q2
        self.critic2_target = Critic(
            state_dim, action_dim, hidden_dim=config.HIDDEN_DIM
        ).to(self.device)

        # Make each target start as an exact copy of its main network
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 
        self.actor_opt = optim.Adam( 
            self.actor.parameters(), lr=config.ACTOR_LR # 1e-4: slow
        )
        # One optimiser over both critics. The two losses touch disjoint parameters, and
        # Adam is per-parameter, so this is equivalent to two separate optimisers.
        self.critic_opt = optim.Adam(
            list(self.critic.parameters()) + list(self.critic2.parameters()),
            lr=config.CRITIC_LR # 3e-4: faster
        )

        self.replay = ReplayBuffer(config.BUFFER_SIZE) # # shared memory, up to 500,000

        # One independent OU noise process per vehicle slot
        # Ornstein-Uhlenbeck - push, brake. Exploration
        self.noises = [
            OUNoise(
                size=action_dim,
                mu=config.NOISE_MU,
                theta=config.NOISE_THETA,
                sigma=config.NOISE_SIGMA,
                low=config.ACTION_LOW,
                high=config.ACTION_HIGH,
            )
            for _ in range(config.MAX_VEHICLES)
        ]

        self.total_steps = 0         # counts critic updates
        self.last_q_mean = 0.0       # mean Q1 on the last batch, in ORIGINAL reward units
        self.last_actor_loss = 0.0   # the actor only updates every POLICY_DELAY calls

    def select_action(self, state, add_noise=True, noise_scale=1.0,
                      vehicle_id=0):
        # Warm-up: uniform-random actions until the buffer holds WARMUP_STEPS transitions.
        # Training only (add_noise=True), so evaluation stays deterministic.
        if add_noise and len(self.replay) < self.cfg.WARMUP_STEPS:
            return np.random.uniform(
                self.cfg.ACTION_LOW, self.cfg.ACTION_HIGH,
                size=self.action_dim
            ).astype(np.float32)

        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if add_noise:
            action = action + noise_scale * self.noises[vehicle_id].sample()
        return np.clip(
            action, self.cfg.ACTION_LOW, self.cfg.ACTION_HIGH
        )

    def reset_noise(self):
        for n in self.noises:
            n.reset()

    def store(self, s, a, r, s2, d):
        self.replay.push(s, a, r, s2, d)

    def update(self):
        # No learning until the warm-up data exists
        if len(self.replay) < max(self.cfg.BATCH_SIZE, self.cfg.WARMUP_STEPS):
            return None, None

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.cfg.BATCH_SIZE
        )
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        low, high = self.cfg.ACTION_LOW, self.cfg.ACTION_HIGH
        half_range = (high - low) / 2.0

        with torch.no_grad():
            # Target-policy smoothing: clipped noise on the target action, so the critic
            # can't be exploited through sharp peaks in Q
            noise = (torch.randn_like(actions) * self.target_noise_std * half_range).clamp(
                -self.target_noise_clip * half_range,
                self.target_noise_clip * half_range,
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(low, high)

            # Clipped double-Q: take the MIN of the two target critics
            target_q = torch.min(
                self.critic_target(next_states, next_actions),
                self.critic2_target(next_states, next_actions),
            )
            # Match reward/done to the critic's output shape so nothing broadcasts to (B, B)
            r = rewards.reshape(target_q.shape) / self.reward_scale
            d = dones.reshape(target_q.shape)
            y = r + self.cfg.GAMMA * (1.0 - d) * target_q   # (1 - d): no bootstrapping past terminals


        # Calculating the loss
        q1 = self.critic(states, actions)
        q2 = self.critic2(states, actions)
        assert q1.shape == y.shape and q2.shape == y.shape, (q1.shape, q2.shape, y.shape)

        loss1 = F.mse_loss(q1, y)
        loss2 = F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        (loss1 + loss2).backward() # Back propagation
        
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)    # clip each critic separately
        nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic_opt.step()

        self.total_steps += 1
        # Multiply back by the reward scale so Q is comparable to the logged episode reward
        self.last_q_mean = float(q1.mean().item()) * self.reward_scale

        # Delayed actor + target updates (TD3): the critics get several updates per actor step
        if self.total_steps % self.policy_delay == 0:
            actor_loss = -self.critic(states, self.actor(states)).mean()   # actor follows Q1 only

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()
            self.last_actor_loss = float(actor_loss.item())

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.critic2, self.critic2_target)

        # Logged critic loss = mean of the two critic losses, comparable to the single-critic scale
        return 0.5 * float(loss1.item() + loss2.item()), self.last_actor_loss

    def _soft_update(self, source, target):
        tau = self.cfg.TAU
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    def save(self, path_prefix):
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")
        torch.save(self.critic2.state_dict(), f"{path_prefix}_critic2.pth")

    def load(self, path_prefix):
        self.actor.load_state_dict(
            torch.load(f"{path_prefix}_actor.pth", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f"{path_prefix}_critic.pth", map_location=self.device)
        )
        c2_path = f"{path_prefix}_critic2.pth"
        if os.path.exists(c2_path):
            self.critic2.load_state_dict(torch.load(c2_path, map_location=self.device))
        else:
            # single-critic checkpoint: start the twin from the same weights
            self.critic2.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())