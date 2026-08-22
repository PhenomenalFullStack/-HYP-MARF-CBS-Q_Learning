# deep_q_learning_agent.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, vehicle, state_size, action_size=3,
                 lr=0.001, gamma=0.9, epsilon=1.0, epsilon_decay=0.995,
                 min_epsilon=0.01, batch_size=64, target_update=100):
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
        self.update_counter = 0

        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.memory = ReplayBuffer()

        # Copy weights to target
        self.target_network.load_state_dict(self.q_network.state_dict())

    def build_state_key(self, observation, schedule_pos, actual_pos, delay):
        """Flatten the state into a 1D vector."""
        obs = np.array(observation, dtype=np.float32)
        sched = np.array(schedule_pos, dtype=np.float32)
        actual = np.array(actual_pos, dtype=np.float32)
        delay = np.array([delay], dtype=np.float32)
        return np.concatenate([obs, sched, actual, delay])

    def select_action(self, observation, schedule_pos, actual_pos, delay, allowed_actions=None):
        if allowed_actions is None:
            allowed_actions = list(range(self.action_size))
        if random.random() < self.epsilon:
            return random.choice(allowed_actions)
        state = self.build_state_key(observation, schedule_pos, actual_pos, delay)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t).squeeze(0).cpu().numpy()
        
        # Mask disallowed actions
        masked_q = [q_values[i] if i in allowed_actions else float('-inf') for i in range(self.action_size)]
        return int(np.argmax(masked_q))

    def compute_reward(self, old_pos, new_pos, next_schedule_pos, delay, collision, goal):
        """Same reward function as Q-Learning agent."""
        reward = 0.0
        if collision:
            reward -= 10.0
        if new_pos == goal:
            reward += 5.0
        reward -= 0.5 * delay
        if delay == 0:
            reward += 0.1
        old_dist = abs(old_pos[0]-goal[0]) + abs(old_pos[1]-goal[1])
        new_dist = abs(new_pos[0]-goal[0]) + abs(new_pos[1]-goal[1])
        if new_dist < old_dist:
            reward += 0.1
        return reward

    def update_q_table(self, state, action, reward, next_state):
        """Store experience and train if buffer is large enough."""
        self.memory.push(state, action, reward, next_state, False)  # done not used
        if len(self.memory) >= self.batch_size:
            self._train()

    def _train(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss and update
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']