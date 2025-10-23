"""Global configuration for project_curiosity."""
from __future__ import annotations

import torch

VOCAB_SIZE = 1000
EMBED_DIM = 100
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
ACTION_LOSS_WEIGHT = 0.2

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
