"""
actor.py

Deterministic policy network for DDPG.
Maps state -> continuous action.

Uses LayerNorm (not BatchNorm) so it works with batch size 1
during action selection.
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high,
                 hidden_dim=256):
        super().__init__()

        self.action_low = action_low
        self.action_high = action_high

        self.register_buffer(
            "action_scale",
            torch.tensor((action_high - action_low) / 2.0,
                         dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_high + action_low) / 2.0,
                         dtype=torch.float32)
        )

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)
                nn.init.zeros_(self.net[-2].bias)

    def forward(self, state):
        x = self.net(state)
        return x * self.action_scale + self.action_bias