"""
critic.py

Action-value network for DDPG.
Maps (state, action) -> scalar Q-value.
Uses LayerNorm for batch-size-1 compatibility.
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in list(self.state_net) + list(self.q_net):
            if isinstance(m, nn.Linear):
                nn.init.uniform_(self.q_net[-1].weight, -3e-3, 3e-3)
                nn.init.zeros_(self.q_net[-1].bias)

    def forward(self, state, action):
        s = self.state_net(state)
        x = torch.cat([s, action], dim=-1)
        return self.q_net(x)