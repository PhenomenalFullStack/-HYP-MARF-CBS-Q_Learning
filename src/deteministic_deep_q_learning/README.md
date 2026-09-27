# Deterministic Policy Gradient (DPG) approach.

- So my project has a grid based actions (wait, skip or move), This works well for the grid-based simulation, but fails with continous space.
- Deterministic Policy Gradient can help solve the continuos control system.

- Handles Continuous Actions: Instead of just Discrete actions "WAIT" or "SKIP," a vehicle could choose any acceleration value. DPG (Continous) is designed for this, whereas DQN (Stochastic) struggles with it


## How is DPG Related to Deep Q-Networks (DQN)?
- DQN is for Discrete Actions: DQN is excellent at handling high-dimensional state spaces (like images) but fails with continuous action spaces because it's impossible to find the maximum Q-value over an infinite number of actions.

- DDPG (Deep Deterministic Policy Gradient): This algorithm combines the best of both worlds. It uses the idea of DQN (replay buffer, target networks) but applies it to a DPG framework. Specifically, DDPG uses a DQN-like "Critic" network to estimate Q-values and a separate "Actor" network to output the deterministic action


## Pipeline 4 Implementation.

1. Environment: A simulation where vehicles have continuous state and action spaces. For example, the state could be (x, y, vx, vy) and the action could be (ax, ay) (continuous accelerations).

2. Actor Network: This is the policy network, μ(s|θ). It takes the state as input and outputs a single, continuous action.

3. Critic Network: This is the Q-value network, Q(s, a|w). It takes a state-action pair and outputs the estimated Q-value.

4. Training Loop:

    4.1. Collect Experience: The agent uses the Actor network to select actions, but to encourage exploration, you add noise (e.g., Ornstein-Uhlenbeck or Gaussian noise) to the output action. Store the (state, action, reward, next_state) transitions in a replay buffer.

    4.2 Update Critic: Sample a batch from the replay buffer and update the Critic to minimize the difference between its predicted Q-value and the target Q-value (using the Bellman equation).

    4.3 Update Actor: Update the Actor using the deterministic policy gradient. The gradient is calculated as the product of the Critic's gradient with respect to the action and the Actor's gradient with respect to its parameters.

    4.4 Update Target Networks: Softly update the target networks for stability.


# Python Files
#	File	            Purpose
1	config.py	        All hyperparameters and settings
2	environment.py	    Continuous intersection environment
3	actor.py	        Actor network (policy)
4	critic.py	        Critic network (Q-value)
5	replay_buffer.py	Experience replay memory
6	noise.py	        Exploration noise process
7	ddpg_agent.py	    DDPG agent (ties actor, critic, noise together)
8	train.py	        Training loop
9	evaluate.py	        Evaluation and metrics
10	utils.py	        Plotting, logging, helpers


# 1. Train
python train.py

# 2. Evaluate
python evaluate.py

# 3. Generate learning curve
python -c "from utils import plot_learning_curve; import config, os; plot_learning_curve(os.path.join(config.LOG_DIR,'training_log.csv'), config.RESULT_DIR)"

# 4. Running Demo
python demo.py --noise 0.0 -> No noise
python demo.py --noise 0.1 -> Noise