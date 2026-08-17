# q_learning

import random
import json

class QLearningAgent:
    def __init__(self, vehicle, actions=3, lr=0.1, gamma=0.9, epsilon=1.0):
        self.vehicle = vehicle
        self.actions = actions # (up, wait, skip_one_cell)
        self.lr = lr # learning rate alpha, how much of the new information overrides the old information
        self.gamma = gamma # discount factor, how much we should value the future reward vs immediate rewards
        self.epsilon = epsilon # balancing the exploration vs exploitation
        self.q_table = {} # dictionary of the q_table, (state_string, q_values=3)

    # building the state string for the q_tables
    # 110010000_56_55_1
    def build_state_key(self, observation, schedule_pos, actual_pos, delay):
        obs_str = ''.join(str(x) for x in observation) # convert the observation into string and remove the spaces in between
        return f"{obs_str}_{schedule_pos[0]}{schedule_pos[1]}_{actual_pos[0]}{actual_pos[1]}_{delay}" # return the string

    # use the state string to the the q_values
    def get_q_values(self, state_key): 
        if state_key not in self.q_table: # check if the state string is in the q_table
            self.q_table[state_key] = [0.0] * self.actions # if its not in the q_table initialise the q_values to zeros
        return self.q_table[state_key] # return the q_values

    # selecting an action, using the epsilon.
    def select_action(self, observation, schedule_pos, actual_pos, delay):
        # create a state string
        state = self.build_state_key(observation, schedule_pos, actual_pos, delay)
        
        # get the q_values for each movement
        q_vals = self.get_q_values(state)
        
        # Explore: select an action randomly 
        if random.random() < self.epsilon: # check if the random number is less than the epsilon.
            # If true, explore: pick and return a random action
            return random.randint(0, self.actions - 1) # pick a random action from the available actions (0, 1, 2)
        
        # Exploit: get the max q_value
        max_q = max(q_vals)
        # choose the best move with the q_value
        best = [i for i, q in enumerate(q_vals) if q == max_q] # loop through the q_values and get the index of the max q_value
        return random.choice(best) # if there is a tie between two q_values choose a random index of the max q_value

    # compute the reward for the action taken by the agent
    # reward function: if the vehicle is on schedule, give a positive reward, if it is not on schedule, give a negative reward, if it collides with another vehicle, give a large negative reward, if it reaches the goal, give a positive reward
    def compute_reward(self, old_pos, new_pos, next_schedule_pos, delay, collision, goal):
        reward = 0.0 
        if collision:
            reward -= 10.0 # if collision penalize the agent

        if new_pos == next_schedule_pos:
            reward += 1.0 # if up to schedule reward the agent
        else:
            reward -= 1.0 # if not on schedule penalize the agent

        if new_pos == goal and old_pos != goal: # if reached the goal reward the agent
            reward += 5.0

        return reward

    # update the q_table using the standard formula Q(s, a) = Q(s, a) + lr x [r + gamma x max(Q(s', a')) - Q(s, a)]
    def update_q_table(self, state_string, action, reward, next_state_string):
        # get the current state q_values Q(s, a)
        q_vals = self.get_q_values(state_string)
        # get the next state q_values Q(s', a')
        next_q_vals = self.get_q_values(next_state_string)
        
        # max of the future state max(Q(s', a'))
        best_next = max(next_q_vals) if next_q_vals else 0.0
        
        # update the new q_values
        q_vals[action] += self.lr * (reward + self.gamma * best_next - q_vals[action])

    # reduce the epsilon when learning
    def decay_epsilon(self, min_eps=0.05, decay=0.97):
        self.epsilon = max(min_eps, self.epsilon * decay) # take the max between the min epsilon and the decayed epsilon

    # save the q_tables to a json file.
    def save_q_table(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.q_table, f, indent=4)

    # load the q_tables from a json file.
    def load_q_table(self, filepath):
        try:
            with open(filepath, 'r') as f:
                self.q_table = json.load(f)
        except FileNotFoundError:
            self.q_table = {}