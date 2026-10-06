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
    # state_dim = 25
    # action_dim = 2
    # low = -0.5 and high = 0.5
    def __init__(self, state_dim, action_dim, action_low, action_high,
                 hidden_dim=256):
        super().__init__()

        self.action_low = action_low
        self.action_high = action_high

        # (0.5 - (-0.5)) / 2 = 0.5. The half-range, used to stretch tanh's [-1, 1] to [-0.5, 0.5].
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

        # Actor network foward pass
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), # input -> hidden layer 1 (25->256)
            nn.LayerNorm(hidden_dim), # normalise the 256 values (mean 0 and spread 1)
            nn.ReLU(), # negative to 0
            nn.Linear(hidden_dim, hidden_dim), # first hidden layer to second hidden layer (256->265)
            nn.ReLU(), # negative to 0
            nn.Linear(hidden_dim, action_dim), # second hidden layer to output (256->3)
            nn.Tanh(), # squash into [-1,1] so action can't exceed the limints
        )

        self._init_weights() # now override the starting weights of the final layer

    # initializing weights (start with zero accelleration)
    def _init_weights(self):
        for m in self.net: # loop over every layer in the pipe
            if isinstance(m, nn.Linear):
                # Set the weights to tiny random values between -0.003 and +0.003.
                nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3) # second from the end = the LAST Linear layer (the very last is Tanh).
                nn.init.zeros_(self.net[-2].bias) # and its biases to 0

    # foward pass
    def forward(self, state):
        x = self.net(state) # run the pipe: 25 numbers and get 2 numbers in [-1, 1]
        return x * self.action_scale + self.action_bias 