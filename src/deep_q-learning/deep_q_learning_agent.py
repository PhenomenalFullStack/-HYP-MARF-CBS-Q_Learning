# deep_q_learning_agent.py

# imports
import random # radom number generator.
import numpy as np # numpy array numbers.
import torch # neural network library.
import torch.nn as nn # neural network module (layers, loss functions, activations).
import torch.optim as optim # optimization algorithms (SGD, Adam).
from collections import deque # a list that drops it's oldest items when full.


# Neural network for DQN for each vehicle.
class DQNNetwork(nn.Module): # inherits from nn.Module.
    # state_size: size of the input state vector (14 numbers: 9 for observation, 2 for schedule_pos, 2 for actual_pos, 1 for delay).
    # action_size: number of possible actions (3: move forward, wait, catch up).
    # hidden_size: number of neurons in the hidden layers (128).
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__() # call the constructor of the parent class (nn.Module).
        
        self.fc1 = nn.Linear(state_size, hidden_size) # first fully connected layer that takes the state vector and outputs hidden_size neurons.
        self.fc2 = nn.Linear(hidden_size, hidden_size) # second fully connected layer that takes the output of the first layer and outputs hidden_size neurons.
        self.fc3 = nn.Linear(hidden_size, action_size) # third fully connected layer that takes the output of the second layer and outputs action_size neurons.

    # forward pass through the network
    def forward(self, x):
        # how data flows through the network. x = the state.
        x = torch.relu(self.fc1(x)) # for pass through the first layer and apply ReLU activation function. (non-linear function that outputs the input if it's positive and 0 if it's negative).
        x = torch.relu(self.fc2(x)) # for pass through the second layer and apply ReLU activation function.
        return self.fc3(x) # Outputs are Q-values (expected future reward per action).


# The agent's memory buffer for storing experiences (state, action, reward, next_state, done).
class ReplayBuffer:
    def __init__(self, capacity=100000): # how many experiences to store in the buffer
        self.buffer = deque(maxlen=capacity) # forget the oldest experiences when the buffer is full. deque is a double-ended.

    # add a new experience to the buffer
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done)) #  Stored as one tuple of 5 items

    # pick batch_size random experiences.
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # pick batch_size random experiences from the buffer
        states, actions, rewards, next_states, dones = zip(*batch)  # same as: states, actions, rewards, next_states, dones # unzip the batch into separate lists
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones)) # convert the lists to numpy arrays and return them

    # return the number of experiences in the buffer (is the buffer full?)
    def __len__(self):
        return len(self.buffer)


# the DQN agent that interacts with the environment and learns from experiences.
class DQNAgent:
    # Setting hyperparameters for the DQN agent.
    def __init__(self, vehicle, state_size, action_size=3,
                 lr=0.001, gamma=0.9, epsilon=1.0, epsilon_decay=0.995,
                 min_epsilon=0.01, batch_size=64, target_update=100):
        self.vehicle = vehicle # agent
        self.state_size = state_size # input size (9+2+2+1=14)
        self.action_size = action_size # output size (3)
        self.lr = lr # how big each weight update is
        self.gamma = gamma # discount factor how much future rewards matter.
        self.epsilon = epsilon # exploration rate vs exploitation rate.
        self.epsilon_decay = epsilon_decay # how fast the exploration rate decreases.
        self.min_epsilon = min_epsilon # smallest exploration rate (to avoid getting stuck in local optima).
        self.batch_size = batch_size # how many experiences to sample from the replay buffer for each training step.
        self.target_update = target_update # how often to update the target network (in number of training steps).
        self.update_counter = 0 # number of training steps since the last target network update.

        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU if available, otherwise use CPU
        self.q_network = DQNNetwork(state_size, action_size).to(self.device) # main networ that lerns and make dicisions.
        self.target_network = DQNNetwork(state_size, action_size).to(self.device) # identical copy of the main network that is fixed for a while and used to compute target Q-values during training.
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr) # Adam optimizer that updates the weights of the main network based on the gradients computed from the loss function.
        self.loss_fn = nn.MSELoss() # mean squared error loss function that measures the difference between the predicted Q-values and the target Q-values.

        # Replay buffer
        self.memory = ReplayBuffer() # create an empty memory for this agent.

        # Copy weights to target
        self.target_network.load_state_dict(self.q_network.state_dict())


    # Build the state key by flattening the observation, schedule position, actual position, and delay into a 1D vector.
    def build_state_key(self, observation, schedule_pos, actual_pos, delay):
        """Flatten the state into a 1D vector."""
        obs = np.array(observation, dtype=np.float32) # 9 sensor numbers to array of decimals
        sched = np.array(schedule_pos, dtype=np.float32) # (row, col) where vehicle should be to array of 2
        actual = np.array(actual_pos, dtype=np.float32) # (row, col) where vehicle is to array of 2
        delay = np.array([delay], dtype=np.float32) # 1 number to array of 1
        return np.concatenate([obs, sched, actual, delay]) # combine them into 9+2+2+1=14 numbers and return the array


    # Select an action based on the current state using epsilon-greedy policy.
    def select_action(self, observation, schedule_pos, actual_pos, delay, allowed_actions=None):
        # decision to move. Allowed action restrict choices to only those that are valid in the current state. If allowed_actions is None, all actions are allowed.
        if allowed_actions is None: 
            allowed_actions = list(range(self.action_size)) # List of all possible actions
            
        # exploration: choose a random action from the allowed actions.
        if random.random() < self.epsilon:
            return random.choice(allowed_actions)
        
        # exploitation: choose the action with the highest Q-value from the allowed actions.
        state = self.build_state_key(observation, schedule_pos, actual_pos, delay)
        # the network expects a batch of states, so unsqueeze(0) adds a batch dimension (1, 14) and move the tensor to the appropriate device (CPU or GPU).
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device) # convert state to tenstor and add a batch dimension (1, 14) and move to device (CPU or GPU)
        with torch.no_grad(): # Don't track gradients during action selection: prediction, not learning.
            q_values = self.q_network(state_t).squeeze(0).cpu().numpy() # squeeze(0) to remove the batch dimension and move back to CPU and convert to numpy array.
        
        # Mask disallowed actions
        masked_q = [q_values[i] if i in allowed_actions else float('-inf') for i in range(self.action_size)]
        # for each action i: keep its Q-value if it's allowed, otherwise set it to negative infinity (so it won't be chosen).
        return int(np.argmax(masked_q)) # index of the largest value = the best allowed action.


    # Compute the reward, reward for good actions and penalties for bad actions.
    def compute_reward(self, old_pos, new_pos, next_schedule_pos, delay, collision, goal):
        """reward function"""
        reward = 0.0
        if collision:
            reward -= 10.0 # penalty for collision
        if new_pos == goal:
            reward += 5.0 # reward for reaching the goal
        reward -= 0.5 * delay # penalty for delay
        if delay == 0: 
            reward += 0.1 # reward for being on time
            
        old_dist = abs(old_pos[0]-goal[0]) + abs(old_pos[1]-goal[1]) # Manhattan distance before moving
        new_dist = abs(new_pos[0]-goal[0]) + abs(new_pos[1]-goal[1]) # Manhattan distance after moving
        
        # reward for moving closer to the goal
        if new_dist < old_dist: 
            reward += 0.1 # reward for moving closer to the goal
        return reward

    # Store experience in the replay buffer.
    def update_q_table(self, state, action, reward, next_state):
        """Store experience and train if buffer is large enough."""
        self.memory.push(state, action, reward, next_state, False)  # done not used
        if len(self.memory) >= self.batch_size: # check if the buffer has enough experiences to sample a batch for training.
            self._train() # train the network using a batch of experiences from the replay buffer.

    # Train the Q-network using a batch of experiences from the replay buffer.
    def _train(self): # call it only in the deep_q_learning_agent.py file, not in the main.py file.
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size) # sample a batch of random memories.
        states = torch.FloatTensor(states).to(self.device) # convert states to tensor and move to device (CPU or GPU)
        actions = torch.LongTensor(actions).to(self.device) # Actions are whole numbers (0, 1, 2) so we use LongTensor and move to device (CPU or GPU)
        rewards = torch.FloatTensor(rewards).to(self.device) # convert rewards to tensor and move to device (CPU or GPU)
        next_states = torch.FloatTensor(next_states).to(self.device) # convert next states to tensor and move to device (CPU or GPU)
        dones = torch.FloatTensor(dones).to(self.device) # convert dones to tensor and move to device (CPU or GPU)

        # Current Q-values
        # gather(1, actions.unsqueeze(1)) selects the Q-values corresponding to the actions taken in the batch. 
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1) # unsqueeze(1) adds a dimension so that actions can be used to index into the Q-values. squeeze(1) removes that extra dimension after gathering.

        # Target Q-values
        with torch.no_grad(): # the target is a fixed goal, no learning.
            next_q = self.target_network(next_states).max(1)[0] # returns the maximum Q-value for each next state in the batch.
            
            # Bellman equation: target Q-value = reward + gamma * max(next Q-value) * (1 - done) zero if the future state is not done, otherwise target Q-value = reward.
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss and update
        loss = self.loss_fn(current_q, target_q)  # How wrong was the prediction vs the target?
        self.optimizer.zero_grad() # cleer gradients.
        loss.backward() # Backpropagation: Compute how each weight contributed to the error.
        self.optimizer.step() # Adjust the weights to reduce the error.

        # Update target network
        self.update_counter += 1 # increment the counter for how many training steps have been taken since the last target network update.
        if self.update_counter % self.target_update == 0: # % = reminder; true if the reminder is 0, meaning it's time to update the target network. true every 100 training steps.
            self.target_network.load_state_dict(self.q_network.state_dict()) # Refresh target with the latest weights.

    # reduce epsilon to decrease exploration over time, but not below min_epsilon.
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


    # save the agent's state to a file for later use.
    def save(self, path):
        torch.save({ # Write a dictionary of everything to a file
            'q_network_state_dict': self.q_network.state_dict(), # main network weights
            'target_network_state_dict': self.target_network.state_dict(), # target network weights
            'optimizer_state_dict': self.optimizer.state_dict(), # optimser internal state (for resuming training)
            'epsilon': self.epsilon, # current exploration rate
        }, path)

    # read the saved file back in.
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon'] # restore everything from the file