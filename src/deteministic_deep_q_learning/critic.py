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

        # process the state
        self.state_net = nn.Sequential( 
            nn.Linear(state_dim, hidden_dim), # (25->256)
            nn.LayerNorm(hidden_dim), # normalise the 256 values (mean 0 and spread 1)
            nn.ReLU(),
        )

        # combine the state features with the ACTION
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), # (256+2) -> 256
            nn.ReLU(),
            nn.Linear(hidden_dim, 1), # 256 -> 1: single Q-Value
        )

        self._init_weights()

    def _init_weights(self):
        for m in list(self.state_net) + list(self.q_net): # join the two layer lists into one
            if isinstance(m, nn.Linear):
                nn.init.uniform_(self.q_net[-1].weight, -3e-3, 3e-3) # initialise the weight to smaller number
                nn.init.zeros_(self.q_net[-1].bias) # # re-initialises the last layer

    def forward(self, state, action):
        s = self.state_net(state) # (B, 256) features from the state
        x = torch.cat([s, action], dim=-1) # glue state features and action side by side -> (B, 258)
        return self.q_net(x) # (B, 1) one Q-value per sample