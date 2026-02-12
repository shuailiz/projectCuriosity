"""Configuration for Visual-Motor Continuous Learning System."""

import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Visual Encoder Settings
IMAGE_SIZE = (224, 224)  # Standard input size for ResNet
ENCODED_DIM = 512       # Output dimension of the visual encoder

# Action Settings
ACTION_DIM = 2          # Servo 1 delta, Servo 2 delta
ACTION_SCALE = 10.0     # Maximum degree change per step

# Dual Network Settings
# Fast Learner (Hippocampus)
FAST_HIDDEN_DIM = 256
FAST_LEARNING_RATE = 0.005

# Slow Learner (Cortex)
SLOW_HIDDEN_DIM = 1024
SLOW_LEARNING_RATE = 0.0001

# Training Settings
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 5000
CONSOLIDATION_INTERVAL = 50  # Steps between sleep phases
CONSOLIDATION_REPLAYS = 100  # Number of replays during sleep
DISTILLATION_TEMPERATURE = 2.0
WEIGHT_SYNC_ALPHA = 0.05    # EMA coefficient for weight synchronization
