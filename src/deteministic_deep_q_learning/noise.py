"""
noise.py

Ornstein-Uhlenbeck exploration noise.
"""

import numpy as np


class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2,
                 low=-1.0, high=1.0):
        self.size = size
        self.mu = mu * np.ones(size, dtype=np.float32)
        self.theta = theta
        self.sigma = sigma
        self.low = low
        self.high = high
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = (
            self.theta * (self.mu - self.state)
            + self.sigma * np.random.randn(self.size).astype(np.float32)
        )
        self.state = self.state + dx
        return np.clip(self.state, self.low, self.high)