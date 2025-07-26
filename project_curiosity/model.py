"""PyTorch MLP for project_curiosity."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C

class ConceptActionModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_matrix: torch.Tensor | None = None, freeze_embeddings: bool = False):
        super().__init__()
        # concept and position embeddings
        self.concept_embed = nn.Embedding(vocab_size, C.EMBED_DIM)
        self.pos_embed = nn.Embedding(2, C.EMBED_DIM)  # position 0 for first, 1 for second concept

        if embedding_matrix is not None:
            if embedding_matrix.shape != (vocab_size, C.EMBED_DIM):
                raise ValueError("embedding_matrix has wrong shape")
            self.concept_embed.weight.data.copy_(embedding_matrix)
            self.concept_embed.weight.requires_grad = not freeze_embeddings
        self.action_embed = nn.Embedding(len(C.ACTION_TOKENS), C.EMBED_DIM)
        self.fc1 = nn.Linear(C.EMBED_DIM * 3, C.HIDDEN_DIM)
        self.fc2 = nn.Linear(C.HIDDEN_DIM, vocab_size)
        # secondary heads
        self.fc_action = nn.Linear(C.HIDDEN_DIM, len(C.ACTION_TOKENS))  # action classifier

    def forward(self, a, action, b):
        # positional embeddings
        pos0 = self.pos_embed(torch.zeros_like(a))
        pos1 = self.pos_embed(torch.zeros_like(b) + 1)

        emb_a = self.concept_embed(a) + pos0
        emb_b = self.concept_embed(b) + pos1

        x = torch.cat([
            emb_a,
            self.action_embed(action),
            emb_b,
        ], dim=-1)
        h = F.relu(self.fc1(x))
        concept_logits = self.fc2(h)
        return concept_logits

    # new helper -----------------------------------------------------------
    def propose_action(self, a, b):
        """Return logits over actions based solely on the two concepts."""
        pos0 = self.pos_embed(torch.zeros_like(a))
        pos1 = self.pos_embed(torch.zeros_like(b) + 1)
        emb = torch.cat([
            self.concept_embed(a) + pos0,
            self.concept_embed(b) + pos1,
        ], dim=-1)
        dummy = torch.zeros_like(a)
        zero_action_emb = self.action_embed(torch.zeros_like(dummy))  # same shape
        x = torch.cat([emb, zero_action_emb,], dim=-1)
        h = F.relu(self.fc1(x))
        return self.fc_action(h)
