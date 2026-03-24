import torch

# Game Parameters
GRID_SIZE = 20
BLOCK_SIZE = 20
FPS = 10

# RL Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
REPLAY_CAPACITY = 10_000
TARGET_SYNC = 100

# Physics Parameters
GRAVITY_SHIFT = 20  # Steps between gravity changes

# Reward System
REWARD_EAT = 10
REWARD_DIE = -10
REWARD_CLOSER = 1
REWARD_AWAY = -1
REWARD_GRAVITY_FIGHT = -0.5
REWARD_GRAVITY_RIDE = 0.5

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
