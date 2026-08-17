# README.md

## Project: Hybrid Multi-Agent Coordination for Autonomous Vehicles

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
|   ├── cbs_planner.py                # Global (Priority Queue - Heap).
|   ├── grid.py                       # 10x10 grid of the intersection.
|   ├── demo.py                       # Demo of the q_learning full project only.
|   ├── schedule.py                   # Each vehicles schedule.
|   ├── vehicle.py                    # Single agent.                          
|   ├── q_learning_agent.py           # QLearning agent (No-lib).
|   └── simulation.py                 # Intersection simulation.
|
└──user_interface/
      ├── static/
      |   ├── css/
      |   |   └── style.css               # css styles
      |   └── js/
      |       └── script.js               # Scripts for movement, animations and interaction with the server.
      ├── templates/
      |   └── index.html                  # Graphics for the UI
      └── ui_controller.py                # UI server (reads from JSON).

---