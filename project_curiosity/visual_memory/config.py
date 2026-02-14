"""Configuration for Visual-Motor Continuous Learning System."""

import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Visual Encoder Settings
ENCODER_MODEL = 'resnet50'  # Options: 'resnet18', 'resnet34', 'resnet50'
IMAGE_SIZE = (224, 224)     # Standard input size for ResNet
ENCODED_DIM = 512           # Output dimension of the visual encoder

# Action Settings
ACTION_DIM = 2          # Servo 1 delta, Servo 2 delta
ACTION_SCALE = 10.0     # Maximum degree change per step

# Servo Limits (Degrees)
# S1: Left/Right (Base), S2: Up/Down (Tilt)
SERVO_LIMITS = [
    (-90.0, 90.0),  # S1 Min/Max
    (-40.0, 40.0)   # S2 Min/Max
]

# Dual Network Settings
# Fast Learner (Hippocampus)
FAST_HIDDEN_DIM = 256
FAST_LR = 1e-3              # High LR — reactive, adapts quickly

# Slow Learner (Cortex)
SLOW_HIDDEN_DIM = 1024
SLOW_WAKE_LR = 1e-5         # Very low LR during wake (distill only, no raw targets)
SLOW_SLEEP_LR = 5e-4        # Higher LR during sleep (structural update)

# Replay Buffer
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 5000

# Curiosity Policy Settings
POLICY_HIDDEN_DIM = 256
POLICY_LEARNING_RATE = 0.001
POLICY_NOISE_STD = 0.3      # Exploration noise added to policy output
POLICY_TRAIN_INTERVAL = 4   # Train policy every N steps
POLICY_BATCH_SIZE = 16      # Batch size for policy training from replay buffer
CURIOSITY_WARMUP_STEPS = 50 # Random exploration steps before curiosity kicks in

# ---- Phase Schedule: WAKE → SWS → REM → WAKE → ... ----

# Wake Phase
WAKE_STEPS_PER_CYCLE = 500  # Environment steps before triggering sleep
SLOW_WAKE_UPDATE_INTERVAL = 1  # Update Slow every N wake steps (1 = every step)
# L_wake_slow = w_distill * L_distill + w_raw * L_raw
# L_distill: match Fast predictions (primary)
# L_raw: conservative direct transition learning (keeps Slow grounded in reality)
# Regularization: small LR (SLOW_WAKE_LR) acts as implicit regularization
SLOW_WAKE_W_DISTILL = 1.0   # Weight for Fast distillation (primary)
SLOW_WAKE_W_RAW = 0.1       # Weight for raw transition loss (conservative)

# SWS Phase (Slow-Wave Sleep — Consolidation)
# Fast frozen. Slow trains strongly via distillation from Fast.
# L_SWS = α * L_dyn + β * L_z
SWS_STEPS = 200             # Gradient steps in SWS phase
SWS_ALPHA = 1.0             # Weight for dynamics distillation (core)
SWS_BETA = 0.0              # Weight for representation distillation (0 = single encoder)

# REM Phase (Dreaming — Stabilization)
# Fast frozen. Slow trains via multi-step rollout consistency.
# L_REM = γ * L_ms
REM_STEPS = 200             # Gradient steps in REM phase
REM_K = 3                   # Rollout horizon K (trajectory window = K+1)
REM_GAMMA = 0.5             # Weight for multi-step rollout loss

# Post-Sleep: Tether Fast toward Slow
# Prevents Fast from drifting too far from consolidated knowledge.
# Uses output-level distillation (architectures differ).
TETHER_STEPS = 10           # Gradient steps for Fast tether
TETHER_LR = 0.001           # Learning rate for tether distillation

# Model Storage
# Each model gets its own folder: MODELS_DIR/<name>/
# Contains: config.json, checkpoint.pt, replay.pt, training.log
MODELS_DIR = "models"
CHECKPOINT_SAVE_INTERVAL = 1  # Auto-save every N sleep cycles (0 = manual only)
