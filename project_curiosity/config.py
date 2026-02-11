"""Global configuration for project_curiosity."""
from __future__ import annotations

import torch

VOCAB_SIZE = 1000
EMBED_DIM = 100
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
ACTION_LOSS_WEIGHT = 0.2

# Dual-Network Configuration (Hippocampus-Cortex System)
# Fast Learner (Hippocampus): Small, rapid adaptation
FAST_EMBED_DIM = 64        # Smaller embedding dimension
FAST_HIDDEN_DIM = 128      # Smaller hidden dimension
FAST_LEARNING_RATE = 0.01  # Higher learning rate (10x base)

# Slow Learner (Cortex): Large, stable long-term knowledge
SLOW_EMBED_DIM = 100       # Larger embedding dimension (same as EMBED_DIM)
SLOW_HIDDEN_DIM = 512      # Larger hidden dimension (2x base)
SLOW_LEARNING_RATE = 0.0001  # Lower learning rate (0.1x base)

# Consolidation Parameters (Sleep Phases)
CONSOLIDATION_INTERVAL = 100    # Consolidate every N interaction steps
CONSOLIDATION_REPLAYS = 50      # Number of replays per consolidation
CONSOLIDATION_MODE = 'full'     # Mode: 'deep' (NREM), 'rem', or 'full' (both)
CONSOLIDATION_TEMPERATURE = 2.0 # Temperature for knowledge distillation (deep sleep)
CONSOLIDATION_MIN_DATA = 10     # Minimum tokens with data required for consolidation
WEIGHT_SYNC_ALPHA = 0.05        # EMA coefficient for weight synchronization

ACTION_TOKENS = [
    "oppose",
    "intersect",
    "include",
    "combine",
    "similar",
    "add",
    "subtract",
]

# Actions categorized by type
RELATION_ACTIONS = ["oppose", "similar", "include"]
OPERATION_ACTIONS = ["combine", "add", "subtract", "intersect"]

# Special tokens
UNKNOWN_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
FINISH_TOKEN = "<FINISH>"

# Sequence generation settings
MAX_SEQUENCE_LENGTH = 3  # Maximum number of tokens to generate for multi-word answers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
