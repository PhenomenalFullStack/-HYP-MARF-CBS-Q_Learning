"""
config.py

Intersection geometry for the DDPG pipeline.
World is 20x20. Road is 4 units wide (two 2-unit lanes).
"""

import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULT_DIR = os.path.join(BASE_DIR, "results")
for d in (MODEL_DIR, LOG_DIR, RESULT_DIR):
    os.makedirs(d, exist_ok=True)

# World
GRID_SIZE = 20.0
CENTER = GRID_SIZE / 2.0                # 10.0

NUM_VEHICLES = 4
MAX_VEHICLES = 8
MAX_STEPS = 100
SENSOR_NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
TERMINATE_ON_COLLISION = True

# Road network
ROAD_HALF_WIDTH = 2.0                   # road is 4 units wide
ROAD_MIN = CENTER - ROAD_HALF_WIDTH     # 8.0
ROAD_MAX = CENTER + ROAD_HALF_WIDTH     # 12.0
LANE_OFFSET = 1.0                       # lane centre distance from road centre

EASTBOUND_Y  = CENTER - LANE_OFFSET     # 9.0
WESTBOUND_Y  = CENTER + LANE_OFFSET     # 11.0
NORTHBOUND_X = CENTER + LANE_OFFSET     # 11.0
SOUTHBOUND_X = CENTER - LANE_OFFSET     # 9.0

# Slots: (axis, fixed_lane_coord, start_along, goal_along)
#   axis=0 -> travel along x, axis=1 -> travel along y
SLOT_DEFS = [
    (0, EASTBOUND_Y,   1.0, GRID_SIZE - 1.0),   # 0: eastbound
    (0, WESTBOUND_Y,   GRID_SIZE - 1.0, 1.0),   # 1: westbound
    (1, SOUTHBOUND_X,  GRID_SIZE - 1.0, 1.0),   # 2: southbound
    (1, NORTHBOUND_X,  1.0, GRID_SIZE - 1.0),   # 3: northbound
    # extra slots for scaling
    (0, EASTBOUND_Y,   1.0, GRID_SIZE - 1.0),
    (0, WESTBOUND_Y,   GRID_SIZE - 1.0, 1.0),
    (1, SOUTHBOUND_X,  GRID_SIZE - 1.0, 1.0),
    (1, NORTHBOUND_X,  1.0, GRID_SIZE - 1.0),
]


def _start(slot):
    axis, c, s, _ = SLOT_DEFS[slot]
    return (s, c) if axis == 0 else (c, s)


def _goal(slot):
    axis, c, _, g = SLOT_DEFS[slot]
    return (g, c) if axis == 0 else (c, g)


START_POSITIONS = [_start(k) for k in range(MAX_VEHICLES)]
GOAL_POSITIONS  = [_goal(k) for k in range(MAX_VEHICLES)]

# Vehicle properties
MAX_SPEED = 1.0
MAX_ACCEL = 0.5
VEHICLE_RADIUS = 0.4
GOAL_RADIUS = 0.8

# State / Action
FOV_SIZE = 3
FOV_DIM = FOV_SIZE * FOV_SIZE
STATE_DIM = 4 + 2 + 2 + FOV_DIM + MAX_VEHICLES   # = 25
ACTION_DIM = 2
ACTION_LOW = -MAX_ACCEL
ACTION_HIGH = MAX_ACCEL

# Reward shaping
REWARD_GOAL = 50.0
REWARD_COLLISION = -15.0
REWARD_PROGRESS = 2.0
REWARD_PROGRESS_BONUS = 3.0
REWARD_SCHEDULE_BONUS = 1.0
REWARD_DEVIATION_PENALTY = -1.0
REWARD_STEP_PENALTY = -0.01
REWARD_NO_PROGRESS_PENALTY = -0.3

REWARD_SAFE_DISTANCE = 3.0
REWARD_PROXIMITY_PENALTY = -1.0

LANE_HALF_WIDTH = 0.9
LANE_DEVIATION_PENALTY = -5.0
OFF_ROAD_PENALTY = -20.0

REWARD_SCALE = 10.0

# DDPG hyperparameters
ACTOR_LR = 1e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 500_000
BATCH_SIZE = 64
MAX_EPISODES = 3000
WARMUP_STEPS = 5000
HIDDEN_DIM = 256

POLICY_DELAY = 2
TARGET_NOISE_STD = 0.2
TARGET_NOISE_CLIP = 0.5

# Exploration noise
NOISE_MU = 0.0
NOISE_THETA = 0.15
NOISE_SIGMA = 0.1
NOISE_MIN_SCALE = 0.25
NOISE_DECAY_START = 50
NOISE_DECAY_END = 500

# Training / logging
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_EVERY = 200
EVAL_EVERY = 50
EVAL_EPISODES = 20
PRINT_EVERY = 10
SOLO_RANDOM_SLOT = True