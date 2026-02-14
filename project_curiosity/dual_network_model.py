"""Dual-Network Model: Hippocampus-Cortex Analogy for Continual Learning.

General-purpose continuous dual-network architecture. Domain-agnostic.
For the language-specific discrete model, see dual_network_model_language.py.

Dreaming in the continuous domain uses forward simulation (rollouts)
rather than backward-pass inversion. See dream_rollout() on the model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class ContinuousFastLearner(nn.Module):
    """Hippocampus-like network for continuous domains (e.g., visual-motor)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)


class ContinuousSlowLearner(nn.Module):
    """Cortex-like network for continuous domains."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class ContinuousDualNetworkModel(nn.Module):
    """Dual-network system for continuous domains."""
    
    def __init__(self, input_dim: int, hidden_dim_fast: int, hidden_dim_slow: int, output_dim: int):
        super().__init__()
        
        self.fast_learner = ContinuousFastLearner(input_dim, hidden_dim_fast, output_dim)
        self.slow_learner = ContinuousSlowLearner(input_dim, hidden_dim_slow, output_dim)
        
        self.use_fast = True
        self.consolidation_count = 0
        self.interaction_steps = 0
        
    def forward(self, x):
        if self.use_fast:
            return self.fast_learner(x)
        else:
            return self.slow_learner(x)
            
    def use_fast_learner(self):
        self.use_fast = True
        
    def use_slow_learner(self):
        self.use_fast = False
        
    def get_active_network(self) -> str:
        return 'fast_learner' if self.use_fast else 'slow_learner'
        
    def sync_weights(self, alpha: float = 0.05, direction: str = 'slow_to_fast'):
        """Synchronize weights between networks using exponential moving average."""
        if direction == 'slow_to_fast':
            for fast_param, slow_param in zip(self.fast_learner.parameters(), 
                                             self.slow_learner.parameters()):
                if fast_param.shape == slow_param.shape:
                    fast_param.data = (1 - alpha) * fast_param.data + alpha * slow_param.data
        elif direction == 'fast_to_slow':
            for fast_param, slow_param in zip(self.fast_learner.parameters(), 
                                             self.slow_learner.parameters()):
                if fast_param.shape == slow_param.shape:
                    slow_param.data = (1 - alpha) * slow_param.data + alpha * fast_param.data

    # ---- Dreaming / Forward Simulation ----

    def dream_rollout(self, start_state: torch.Tensor, actions: List[torch.Tensor],
                      network: str = 'slow') -> List[torch.Tensor]:
        """Dream a trajectory via forward simulation through the world model.
        
        Chains single-step predictions: s_{t+1} = model(s_t, a_t).
        The model's own predictions feed into the next step (autoregressive).
        
        Args:
            start_state: Initial state embedding (B, output_dim) or (output_dim,)
            actions: List of action tensors, each (B, action_dim) or (action_dim,)
            network: Which network to simulate through ('fast' or 'slow')
            
        Returns:
            List of predicted state embeddings [s_0, s_1, ..., s_T]
            Length = len(actions) + 1 (includes start_state)
        """
        learner = self.fast_learner if network == 'fast' else self.slow_learner
        
        trajectory = [start_state]
        current = start_state.unsqueeze(0) if start_state.dim() == 1 else start_state
        
        for action in actions:
            action_b = action.unsqueeze(0) if action.dim() == 1 else action
            inp = torch.cat([current, action_b], dim=-1)
            next_state = learner(inp)
            trajectory.append(next_state.squeeze(0) if start_state.dim() == 1 else next_state)
            current = next_state
            
        return trajectory

    def get_network_stats(self) -> Dict[str, int]:
        """Get statistics about network sizes."""
        fast_params = sum(p.numel() for p in self.fast_learner.parameters())
        slow_params = sum(p.numel() for p in self.slow_learner.parameters())
        
        return {
            "fast_learner_params": fast_params,
            "slow_learner_params": slow_params,
            "param_ratio": slow_params / fast_params if fast_params > 0 else 0,
            "interaction_steps": self.interaction_steps,
            "consolidation_count": self.consolidation_count
        }
