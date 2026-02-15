"""Visual World Model for project_curiosity.

Composes the visual encoder, dual-network world model, and curiosity policy
into a single module — analogous to language/model.py and
language/dual_network_model_language.py for the discrete domain.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from . import config as C
from .encoder import VisualEncoder
from .curiosity_policy import CuriosityPolicy
from ..dual_network_model import ContinuousDualNetworkModel


class VisualWorldModel(nn.Module):
    """Dual-network visual-motor world model.

    Bundles three components into one ``nn.Module``:

    * **encoder** – frozen ResNet backbone that maps RGB frames → embeddings.
    * **dual_net** – ``ContinuousDualNetworkModel`` (fast + slow learners)
      that predicts next-state embeddings from (state, action).
    * **policy** – ``CuriosityPolicy`` that outputs servo-delta actions
      trained to maximise world-model prediction error.

    Keeping them together simplifies device placement, save/load, and
    provides a clean API for the trainer.
    """

    def __init__(self):
        super().__init__()

        # --- Visual Encoder (frozen backbone) ---
        self.encoder = VisualEncoder()

        # --- Dual-Network World Model ---
        self.dual_net = ContinuousDualNetworkModel(
            input_dim=C.ENCODED_DIM + C.ACTION_DIM,
            hidden_dim_fast=C.FAST_HIDDEN_DIM,
            hidden_dim_slow=C.SLOW_HIDDEN_DIM,
            output_dim=C.ENCODED_DIM,
        )

        # --- Curiosity Policy ---
        self.policy = CuriosityPolicy()

    # ---- Convenience properties (delegate to dual_net) ----

    @property
    def fast_learner(self):
        return self.dual_net.fast_learner

    @property
    def slow_learner(self):
        return self.dual_net.slow_learner

    @property
    def consolidation_count(self):
        return self.dual_net.consolidation_count

    @consolidation_count.setter
    def consolidation_count(self, value):
        self.dual_net.consolidation_count = value

    @property
    def interaction_steps(self):
        return self.dual_net.interaction_steps

    @interaction_steps.setter
    def interaction_steps(self, value):
        self.dual_net.interaction_steps = value

    # ---- Forward / Prediction ----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the currently active network of the dual net.

        Args:
            x: Concatenated (state_emb, action) tensor.

        Returns:
            Predicted next-state embedding.
        """
        return self.dual_net(x)

    def encode_frame(self, frame) -> torch.Tensor:
        """Encode a raw image frame into an embedding.

        Args:
            frame: PIL Image, numpy array (H, W, C), or list of them.

        Returns:
            Embedding tensor of shape ``(ENCODED_DIM,)`` or ``(B, ENCODED_DIM)``.
        """
        return self.encoder.encode(frame)

    def predict_next(self, frame, action) -> torch.Tensor:
        """Predict next-state embedding from a raw frame and action.

        Uses whichever network is currently active (fast or slow).

        Args:
            frame: Raw image (numpy array / PIL).
            action: List ``[d1, d2]``.

        Returns:
            Predicted next-state embedding ``(ENCODED_DIM,)``.
        """
        with torch.no_grad():
            state_emb = self.encode_frame(frame)
            state_b = state_emb.unsqueeze(0)
            action_b = torch.tensor(
                action, dtype=torch.float32, device=C.DEVICE
            ).unsqueeze(0)
            inp = torch.cat([state_b, action_b], dim=-1)
            pred_emb = self.dual_net(inp)
            return pred_emb.squeeze(0)

    # ---- Curiosity helpers ----

    def compute_curiosity_reward(
        self,
        state_emb: torch.Tensor,
        action,
        next_state_emb: torch.Tensor,
    ) -> float:
        """Curiosity reward = fast-learner prediction error (MSE).

        Args:
            state_emb: ``(ENCODED_DIM,)``
            action: list or Tensor ``[d1, d2]``
            next_state_emb: ``(ENCODED_DIM,)``

        Returns:
            Scalar prediction error.
        """
        with torch.no_grad():
            state_b = state_emb.unsqueeze(0)
            if isinstance(action, list):
                action_b = torch.tensor(
                    action, dtype=torch.float32, device=C.DEVICE
                ).unsqueeze(0)
            else:
                action_b = action.unsqueeze(0)
            target_b = next_state_emb.unsqueeze(0)
            inp = torch.cat([state_b, action_b], dim=-1)
            pred = self.fast_learner(inp)
            return F.mse_loss(pred, target_b).item()

    # ---- Dual-net delegation ----

    def use_fast_learner(self):
        self.dual_net.use_fast_learner()

    def use_slow_learner(self):
        self.dual_net.use_slow_learner()

    def get_active_network(self) -> str:
        return self.dual_net.get_active_network()

    def sync_weights(self, alpha: float = 0.05, direction: str = "slow_to_fast"):
        self.dual_net.sync_weights(alpha, direction)

    def dream_rollout(
        self,
        start_state: torch.Tensor,
        actions: List[torch.Tensor],
        network: str = "slow",
    ) -> List[torch.Tensor]:
        return self.dual_net.dream_rollout(start_state, actions, network)

    def get_network_stats(self) -> Dict[str, int]:
        return self.dual_net.get_network_stats()
