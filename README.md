# README.md

## Project: Hybrid Multi-Agent Coordination for Autonomous Vehicles with Q-Learning and Deep Q-Learning Network.

### 1. Overview

This project investigates the problem of coordinating multiple autonomous vehicles (AVs) at an unsignalised intersection, where each vehicle has a limited field of view (3x3 grid) and is subject to real-world execution disturbances (sensor noise).

Our research is structured into two different approaches, created in separate folders:

- **The Baseline (`src/baseline_lib/`)** - Implement standard Global Planner MAPF CBS Algorithm. Its purpose is to demonstrate the failure modes of existing method cbs under realistic disturbances. This proves our research gap.

- **The Project (`src/project/`)** - Implements the same MAPF CBS algorithm from scratch. It builds a hybrid system that combines CBS, QLearning, Deviation Monitor, CBS Trigger, and Priority-Aware CBS replanning. This is our proposed solution that overcomes the failures identified in the baseline.

The two approaches remain completely separate to ensure that the baseline provides an independent verification of the problem, while the project demonstrates a novel solution suitable for real-world deployment.

---

### 2. The Baseline (`src/baseline_lib/`)

#### 2.1 Purpose
The baseline exists to validate the research gap described in the problem statement. It shows that:

- **CBS alone** fails when sensor noise is introduced, because it assumes perfect execution and global knowledge.

#### 2.2 Implementation
- **CBS**: Uses the same conflict-based search logic,  but we inject disturbances during execution.
- **Simulation**: A grid environment that allows us to vary sensor noise (0.0-0.5), and the number of AVs (2-8).

#### 2.3 Experiments
We run sets of experiments:

1. **CBS-only** under all increasing disturbance levels - measuring collision rate.

The results (stored in `results/`) are plotted to show that the method become unsafe and inefficient under realistic conditions. This serves as our problem motivation

---

### 3. The Project (`src/project/`)

#### 3.1 Purpose
The project is our proposed solution - a hybrid system that combines the safety guarantee of CBS with the adaptability of reinforcement learning, while adding a recovery mechanism for execution disturbances.

#### 3.2 Architecture
The project consists of four main modules:

1. **CBS Planner** (`src/project/cbs/`) - Our existing CBS algorithm. It generates an optimal, collision-free schedule for all vehicles, assuming ideal conditions.

2. **QLearning** (`src/project/q_learning/`) - A implementation of Reinforcement QLearing algorithm for vehicle schedule recovery. It includes:
   - A Q-table for each vehicle storing state-action values.
   - State space representing the vehicle's current deviation from schedule, position, and time.
   - Action space allowing the vehicle to adjust speed, take alternative routes, or modify dwell times.
   - Epsilon-greedy exploration strategy for action selection.
   - Q-value updates using the Bellman equation with learning rate and discount factor.
   - This serves as a reinforcement mechanism for the vehicles, helping them recover when they deviate from their schedule. The QLearning agent enables each vehicle to learn optimal recovery actions based on its current state, allowing it to catch up to its schedule through trial-and-error learning without requiring neural networks or complex mixing architectures.

3. **CBSTrigger** (`src/project/cbs_trigger/`) - We add a deviation monitor that runs during execution. At each time step, it compares each vehicle's actual position with its cbs scheduled position that the vehicle have to be at. If the deviation exceeds a threshold (2 time steps or 2 cells) it means the QLearning can not reinforce that, then it signals a critical deviation, then CBS Trigger triggers PriorityCBSPlanner. This means that the CBS schedule failed. If the deviation is one step behind then QLearning reinforce that deviation.

4. **PriorityCBSPlanner** (`src/project/priority_cbs/`) - When a critical deviation is triggered, this module replans the schedules for all remaining agents, but with a priority bias: vehicles that are on schedule receive higher priority during conflict resolution. This prevents on-time vehicles from being penalised by the delays of others.

#### 3.3 How It Solves the Problem
- **CBS** provides a globally safe reference plan, avoiding the safety falls of pure Multi Agent Reinforcement Learning.
- **QLearning** allows each vehicle to follow that plan using only local 3x3 observations, adapting to minor disturbances (slight braking -> 0.1 delays, small sensor errors -> 0.1, and comminication latency -> 0.1) without global replanning.
- **CBSTrigger** catches disturbances that are too large for QLearning to handle locally.
- **PriorityCBSPlanner** replans efficiently, restoring safety while minimising disruption to on-schedule vehicles.

#### 3.4 Validation
We test the hybrid system under the **same disturbance conditions** used in the baseline. The results show:
- Collision rate remains near zero even at high disturbance levels.
- Throughput degrades gracefully (much less than pure QLearning).
- Replanning events are triggered only when necessary, demonstrating the effectiveness of the deviation monitor.

---

### 4. Folder Structure
src/
├── baseline_lib/                         # BASELINE
|   ├── cbs/
|   |   └── cbs_planner.py                # CBS implementation.
|   |
|   ├── results/
|   |   ├── plots                         # Saved plots when training.
|   |   └── intersection_schedules.json   # Saved results for demo (auto-generated when training).
|   |
|   ├── simulation/
|   |   ├── grid_env.py                   # Grid environment with disturbances.
|   |   ├── disturbance_utils.py          # Disturbances injection.
|   |   ├── plots_generator.py            # Generate the plots for the 4 metrics against collisions
|   |   ├── scenario_generator.py         # Vehicle start/goal generator.
|   |   └── schedule_genereator.py        # Main simulation runner.
|   |
|   └── README.md                         # This file.
|
├── project/                              # PROPOSED SOLUTION
│   |                              
|   ├── cbs_planner.py                    # Global (Priority Queue - Heap).
|   ├── grid.py                           # 10x10 grid of the intersection.
|   ├── demo.py                           # Demo of the q_learning full project only.
|   ├── schedule.py                       # Each vehicles schedule.
|   ├── vehicle.py                        # Single agent.                          
|   ├── q_learning_agent.py               # QLearning agent (No-lib).
|   └── simulation.py                     # Intersection simulation.
|
├── deep_q-leaning/                       # PROPOSED SOLUTION
│   |                                                     
|   ├── deep_q_learning_agent.py          # Deep Q-Learning Network agent (PyTorch).
|   ├── simulation_dqn.py                 # Intersection simulation.
|   └── ablation_study/
|       ├── deep_q_learning_agent.py      # configurable copy of the The DQN code.
|       ├── ablation_study_4_vehicles.py
|       ├── ablation_plots_4_vehicles.py
|       ├── ablation_study_8_vehicles.py
|       ├── ablation_plots_8_vehicles.py
|       └── results/                      # Aauto-created
|           ├── four_vehicles/
|           │   ├── ablation_results.csv
|           │   └── ablation_*.png
|           └── fivetoeight_vehicles/
|               ├── ablation_results_8_vehicles.csv
|               └── ablation_*_8_vehicles.png
|
└──user_interface/
      ├── static/
      |   ├── css/
      |   |   └── style.css               # css styles
      |   └── js/
      |       └── script.js               # Scripts for movement, animations and interaction with the server.
      |
      ├── templates/
      |   └── index.html                  # Graphics for the UI
      |
      └── ui_controller.py                # UI server (reads from JSON).

---










---

## Parameter Study - Beta

### Q‑learning hyperparameters
  1. epsilon (0.5, 1.0)
  2. learning rate (0.05, 0.1, 0.2)
  3. discount factor (gamma) (0.85, 0.9, 0.95)
  4. epsilon_decay (0.99, 0.995, 0.999)
   
### Training Settings
  1. training_episodes (200, 500, 1000)

### Disturbance Levels
  1. Sensor Noise - 0 or 1 

### Number of Vehicles
  1. 1-8

### Disturbances
Tests only the three worst‑case disturbances:
sensor=1, brake=0, comm=0
sensor=1, brake=1, comm=0
sensor=1, brake=1, comm=1

### Description of the parameter selection strategy
We perform a comprehensive grid search over key Q‑learning hyperparameters (learning rate, discount factor, initial epsilon, epsilon decay, and number of training episodes) for each of the 8 vehicle counts (1–8) and three worst‑case disturbance configurations (sensor+brake, sensor+brake+comm, etc.). For every combination, we train the agent and evaluate it greedily, recording collisions, total steps, and total reward.

The objective is to find parameters that guarantee zero collisions while yielding the shortest possible path (lowest number of timesteps). Since collisions are always zero in our successful runs, we focus solely on minimising the path length.

The bar charts compare, for each vehicle count, the best (shortest steps) and worst (longest steps) parameter sets among those that produced zero collisions. They show that certain values - such as a learning rate of 0.1, discount factor 0.9–0.95, initial epsilon 1.0, epsilon decay 0.999, and 500 training episodes - consistently lead to the shortest routes across different vehicle counts. In contrast, lower decay rates (0.99) or fewer episodes (200) often result in longer paths because the agent either explores too little or fails to fully learn the optimal wait‑skip strategy. This comparison helps us justify our final choice of hyperparameters for the main simulation.


---




---
---




---
# Deep Q-Network (DQNs) - Beta: Plan and Literature Study

## Deep Q-Network
Deep Q Network uses the Q-learning idea and takes it one step further. Instead of using a Q-table, 
we use a Neural Network that takes a state and approximates the Q-values for each action based on that state.

## PROBLEM: 
If our state space is large, a grid with multiple vehicles, each with positions and delays, the Q-table becomes impossibly large to store and update.

## SOLUTION: 
Instead of storing values in a table, the neural network takes a state as input and outputs Q-values for all 
possible actions. The network learns to approximate the Q-function, generalising across similar states.

### Q-Learning vs Deep Q-Network
We do this because using a classic Q-table is not very scalable. It might work for a simple intersection navigation, 
But in a more complex navigation problem with dozens of possible actions and vehicle states, the Q-table will soon become too large and cannot be solved efficiently anymore.


1. Observe the current state.
2. Feed the state into the neural network to get Q-values for all actions.
3. Choose an action (using an epsilon-greedy policy).
4. Execute the action, observe the reward and the next state.
5. Store this experience (state, action, reward, next state) in a replay buffer.
6. Sample a random batch of experiences from the replay buffer and update the neural network using the Q-learning update rule.


### Components of Deep Q-Network.

1. State Representation: What the neural network see. 3x3 observation grid (9 cells), the vehicle's current position, vehicles  scheduled position.

2. Action Space: 3-Actions: Forward, Wait, Skip

3. Reward Function: Penalise collisions and delay, reward progress (moving one step foward), and reachinng the goal.

4. Neural Network Architecture: A simple feedfoward network with 2-3 hidden layers (128, 64 neurons) with ReLU activations. Input size = size of the state vector, output size = 3 (number of actions)

5. Experience Replay Buffer: Stores past experience (state, action, reward, next_state, done). Sample a random batch for training.

6. Target Network: A copy of the main network, updated every N steps (1000 steps). Used to compute stable target Q-Values.

7. Training Looop: For each episode - reset environment, for each step - select action, execute, store experience, sample batch, update network, update target network peroidocally.


### Architecure:
Input Layer (state_size)
         |
   Dense (128, ReLU)
         |
   Dense (128, ReLU)
         |
Output Layer (3 Actions)


### DQN can operate in two modes:

1. Mode A (Replacing Q-Learning, keeping CBS):
DQN replaces the Q-table. The state includes the 3×3 observation, schedule position, actual position, and delay. The actions remain forward, wait, skip. This is a direct upgrade - DQN can handle larger state spaces (e.g., more vehicles, more complex observations) and generalise better. IMPLEMENTED

2. Mode B (End-to-End Learning):
DQN learns the entire policy – it decides both the path and the timing, without relying on a pre-computed CBS schedule. The agent learns directly from the grid state, using the neural network to map observations to actions. This is more challenging but can potentially discover more optimal policies.


### How Deep Q-Network Solves the Problem:
1. Scales to more vehicles: The neural network can handle larger state representations without the Q-table blowing up.
2. Generalisation: DQN can generalise across similar states, potentially learning faster and more robustly.
3. Handles uncertainty: DQN with experience replay can learn from past experiences more efficiently, making it more robust to disturbances.

--- 

# IMPLEMENTATION
1. Mode A (Replacing Q-Learning, keeping the CBS) - Hybrid (CBS + Deep Q-Network).
   The Idea is that the DQL uses a neural network to decide when to wait, skip while still following the CBS schedule.

## DQN Parameters we Use
Parameter       Value       Reason
state_size      14	        Flattened vector: 9 (obs) + 2 (sched) + 2 (actual) + 1 (delay)
hidden_size	    128	        Sufficient for small problems.
learning_rate	  0.001	      Standard for Adam.
gamma	          0.9	        Same as Q-learning Pipeline2.
epsilon_start	  1.0	        Full exploration at start.
epsilon_decay	  0.995	      Slow decay over 500 episodes (~0.08 after 500).
batch_size	    64	        Standard.
target_update	  100	        Update target network every 100 steps.
buffer_size	    100000	    Large enough.


---




---
---




---
# Deep Q-Network (DQNs) - Beta: Ablations Study for 4 Vehicles
The DQN pipeline is highly robust to most architectural and hyperparameter choices: activations, loss functions, hidden‑size, number of layers, learning rate, discount factor, epsilon decay, batch size, target‑update interval, and even the presence of replay or a target network all converge to the same optimal policy on this task. This is because the state space is small with only 4 Autonomous vehicle agents and the optimal action sequence is essentially determined by the CBS route and delay flag. The only ablation that changes the outcome is the choice of optimiser, SGD converges to a weaker policy that yields the same number of steps but produces 52% less reward, confirming that adaptive optimisers are necessary for stable Q‑value learning.
---

---
# Deep Q-Network (DQNs) - Beta: Ablations Study for 4 Vehicles
---