"""
Updated deep_q_learning_agent.py: A fully configurable DQN (activation, loss, hidden size, layers, optimiser, target network toggle, replay toggle, double DQN toggle).
Configurable DQN agent used only by the ablation study.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


def get_activation(name):
    return {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }[name]


def get_loss(name):
    return {
        "mse": nn.MSELoss,
        "huber": nn.SmoothL1Loss,
        "mae": nn.L1Loss,
    }[name]


def get_optimizer(name, params, lr):
    if name == "adam":
        return optim.Adam(params, lr=lr)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


class QNetwork(nn.Module):
    def __init__(
        self, state_size, action_size, hidden_size=128, num_layers=2, activation="relu"
    ):
        super().__init__()
        act_cls = get_activation(activation)
        layers = []
        in_dim = state_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(act_cls())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r), np.array(s2), np.array(d))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        vehicle,
        state_size,
        action_size=3,
        lr=0.001,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        batch_size=64,
        target_update=100,
        hidden_size=128,
        num_layers=2,
        activation="relu",
        loss_fn="mse",
        optimizer="adam",
        use_target_network=True,
        use_replay=True,
        double_dqn=False,
    ):

        self.vehicle = vehicle
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update = target_update
        self.use_target_network = use_target_network
        self.use_replay = use_replay
        self.double_dqn = double_dqn
        self.update_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(
            state_size, action_size, hidden_size, num_layers, activation
        ).to(self.device)

        if use_target_network:
            self.target_network = QNetwork(
                state_size, action_size, hidden_size, num_layers, activation
            ).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
        else:
            self.target_network = None

        self.optimizer = get_optimizer(optimizer, self.q_network.parameters(), lr)
        self.loss_fn = get_loss(loss_fn)()

        self.memory = ReplayBuffer()

    def build_state_key(self, observation, schedule_pos, actual_pos, delay):
        obs = np.array(observation, dtype=np.float32)
        sched = np.array(schedule_pos, dtype=np.float32)
        actual = np.array(actual_pos, dtype=np.float32)
        d = np.array([delay], dtype=np.float32)
        return np.concatenate([obs, sched, actual, d])

    def select_action(
        self, observation, schedule_pos, actual_pos, delay, allowed_actions=None
    ):
        if allowed_actions is None:
            allowed_actions = list(range(self.action_size))

        if random.random() < self.epsilon:
            return random.choice(allowed_actions)

        state = self.build_state_key(observation, schedule_pos, actual_pos, delay)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q_network(state_t).squeeze(0).cpu().numpy()

        masked = [
            q_vals[i] if i in allowed_actions else float("-inf")
            for i in range(self.action_size)
        ]
        return int(np.argmax(masked))

    def compute_reward(
        self, old_pos, new_pos, next_schedule_pos, delay, collision, goal
    ):
        reward = 0.0
        if collision:
            reward -= 10.0
        if new_pos == goal:
            reward += 5.0
        reward -= 0.5 * delay
        if delay == 0:
            reward += 0.1
        old_dist = abs(old_pos[0] - goal[0]) + abs(old_pos[1] - goal[1])
        new_dist = abs(new_pos[0] - goal[0]) + abs(new_pos[1] - goal[1])
        if new_dist < old_dist:
            reward += 0.1
        return reward

    def update_q_table(self, state, action, reward, next_state):
        if self.use_replay:
            self.memory.push(state, action, reward, next_state, False)
            if len(self.memory) >= self.batch_size:
                self._train_batch()
        else:
            self._train_single(state, action, reward, next_state)

    def _train_single(self, state, action, reward, next_state):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        s2 = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        a = torch.LongTensor([action]).to(self.device)
        r = torch.FloatTensor([reward]).to(self.device)

        current_q = self.q_network(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q = self._compute_target(s2, r)
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._maybe_update_target()

    def _train_batch(self):
        s, a, r, s2, d = self.memory.sample(self.batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        current_q = self.q_network(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q = self._compute_target(s2, r, d)
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._maybe_update_target()

    def _compute_target(self, s2, r, done=None):
        if self.use_target_network:
            net_for_target = self.target_network
        else:
            net_for_target = self.q_network

        if self.double_dqn:
            online_q = self.q_network(s2)
            best_actions = online_q.argmax(dim=1, keepdim=True)
            target_vals = net_for_target(s2).gather(1, best_actions).squeeze(1)
        else:
            target_vals = net_for_target(s2).max(1)[0]

        if done is None:
            return r + self.gamma * target_vals
        return r + self.gamma * target_vals * (1 - done)

    def _maybe_update_target(self):
        if not self.use_target_network:
            return
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
