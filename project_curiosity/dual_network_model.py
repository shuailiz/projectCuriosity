"""Dual-Network Model: Hippocampus-Cortex Analogy for Continual Learning.

This module implements a complementary learning system with:
- Fast Learner (Hippocampus): Small network, high learning rate, rapid adaptation
- Slow Learner (Cortex): Large network, low learning rate, stable long-term knowledge
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from . import config as C


class FastLearner(nn.Module):
    """Hippocampus-like network: small, fast, episodic memory."""
    
    def __init__(self, vocab_size: int, embedding_matrix: torch.Tensor | None = None, freeze_embeddings: bool = False):
        super().__init__()
        # Smaller dimensions for fast learning
        self.embed_dim = C.FAST_EMBED_DIM
        self.hidden_dim = C.FAST_HIDDEN_DIM
        self.vocab_size = vocab_size
        
        # Leaky ReLU for approximate invertibility
        self.leaky_relu_alpha = 0.01
        
        # Concept and position embeddings
        self.concept_embed = nn.Embedding(vocab_size, self.embed_dim)
        self.pos_embed = nn.Embedding(2, self.embed_dim)
        
        if embedding_matrix is not None:
            # Project pretrained embeddings to smaller dimension
            if embedding_matrix.shape[1] != self.embed_dim:
                projection = nn.Linear(embedding_matrix.shape[1], self.embed_dim, bias=False)
                with torch.no_grad():
                    projected_emb = projection(embedding_matrix)
                self.concept_embed.weight.data.copy_(projected_emb)
            else:
                self.concept_embed.weight.data.copy_(embedding_matrix)
            self.concept_embed.weight.requires_grad = not freeze_embeddings
        
        self.action_embed = nn.Embedding(len(C.ACTION_TOKENS), self.embed_dim)
        
        # Smaller network architecture
        self.fc1 = nn.Linear(self.embed_dim * 3, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, vocab_size)
    
    def forward(self, a, action, b=None, is_relation=False):
        """Forward pass through fast learner."""
        if is_relation:
            return self.forward_relation(a, action)
        else:
            return self.forward_operation(a, action, b)
    
    def forward_relation(self, a, action):
        """Forward pass for relation actions."""
        pos0 = self.pos_embed(torch.zeros_like(a))
        emb_a = self.concept_embed(a) + pos0
        
        placeholder = torch.zeros_like(a)
        pos1 = self.pos_embed(placeholder + 1)
        placeholder_emb = torch.zeros((a.shape[0], self.embed_dim), device=C.DEVICE)
        
        x = torch.cat([
            emb_a,
            self.action_embed(action),
            placeholder_emb + pos1,
        ], dim=-1)
        
        h = F.leaky_relu(self.fc1(x), negative_slope=self.leaky_relu_alpha)
        concept_logits = self.fc2(h)
        return concept_logits
    
    def forward_operation(self, a, action, b):
        """Forward pass for operation actions."""
        pos0 = self.pos_embed(torch.zeros_like(a))
        pos1 = self.pos_embed(torch.zeros_like(b) + 1)
        
        emb_a = self.concept_embed(a) + pos0
        emb_b = self.concept_embed(b) + pos1
        
        x = torch.cat([
            emb_a,
            self.action_embed(action),
            emb_b,
        ], dim=-1)
        h = F.leaky_relu(self.fc1(x), negative_slope=self.leaky_relu_alpha)
        concept_logits = self.fc2(h)
        return concept_logits
    
    def inverse_leaky_relu(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse of leaky ReLU activation.
        
        Args:
            y: Output of leaky ReLU
            
        Returns:
            Input that would produce y
        """
        return torch.where(y >= 0, y, y / self.leaky_relu_alpha)
    
    def backward_pass(self, concept_logits: torch.Tensor, is_relation: bool = False) -> Dict[str, torch.Tensor]:
        """Run network backwards from output to input (dreaming).
        
        This performs approximate inversion using transposed weights.
        
        Args:
            concept_logits: Output logits to invert
            is_relation: Whether this is a relation or operation
            
        Returns:
            Dictionary with reconstructed embeddings
        """
        # Backward through fc2: h = W2^T @ concept_logits
        h = torch.matmul(concept_logits, self.fc2.weight)  # [batch, hidden_dim]
        
        # Inverse activation
        x = self.inverse_leaky_relu(h)
        
        # Backward through fc1: input = W1^T @ x
        reconstructed = torch.matmul(x, self.fc1.weight)  # [batch, embed_dim * 3]
        
        # Split into components
        emb_a = reconstructed[:, :self.embed_dim]
        action_emb = reconstructed[:, self.embed_dim:self.embed_dim*2]
        emb_b = reconstructed[:, self.embed_dim*2:]
        
        return {
            'concept_a_emb': emb_a,
            'action_emb': action_emb,
            'concept_b_emb': emb_b,
            'hidden': x,
            'reconstructed_input': reconstructed
        }
    
    def dream_concept(self, concept_logits: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
        """Dream: find concepts that would produce these logits.
        
        Args:
            concept_logits: Target output logits
            top_k: Number of top concepts to return
            
        Returns:
            List of (concept_id, similarity_score) tuples
        """
        # Run backward pass
        backward_result = self.backward_pass(concept_logits)
        reconstructed_emb = backward_result['concept_a_emb']
        
        # Find nearest neighbors in embedding space
        all_embeddings = self.concept_embed.weight  # [vocab_size, embed_dim]
        
        # Cosine similarity
        reconstructed_norm = F.normalize(reconstructed_emb, dim=-1)
        embeddings_norm = F.normalize(all_embeddings, dim=-1)
        similarities = torch.matmul(reconstructed_norm, embeddings_norm.T)  # [batch, vocab_size]
        
        # Get top-k
        top_scores, top_indices = torch.topk(similarities[0], k=top_k)
        
        return [(idx.item(), score.item()) for idx, score in zip(top_indices, top_scores)]


class SlowLearner(nn.Module):
    """Cortex-like network: large, slow, stable long-term knowledge."""
    
    def __init__(self, vocab_size: int, embedding_matrix: torch.Tensor | None = None, freeze_embeddings: bool = False):
        super().__init__()
        # Larger dimensions for stable learning
        self.embed_dim = C.SLOW_EMBED_DIM
        self.hidden_dim = C.SLOW_HIDDEN_DIM
        self.vocab_size = vocab_size
        
        # Leaky ReLU for approximate invertibility
        self.leaky_relu_alpha = 0.01
        
        # Concept and position embeddings
        self.concept_embed = nn.Embedding(vocab_size, self.embed_dim)
        self.pos_embed = nn.Embedding(2, self.embed_dim)
        
        if embedding_matrix is not None:
            if embedding_matrix.shape != (vocab_size, self.embed_dim):
                raise ValueError("embedding_matrix has wrong shape for slow learner")
            self.concept_embed.weight.data.copy_(embedding_matrix)
            self.concept_embed.weight.requires_grad = not freeze_embeddings
        
        self.action_embed = nn.Embedding(len(C.ACTION_TOKENS), self.embed_dim)
        
        # Larger network architecture with deeper layers
        self.fc1 = nn.Linear(self.embed_dim * 3, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, vocab_size)
    
    def forward(self, a, action, b=None, is_relation=False):
        """Forward pass through slow learner."""
        if is_relation:
            return self.forward_relation(a, action)
        else:
            return self.forward_operation(a, action, b)
    
    def forward_relation(self, a, action):
        """Forward pass for relation actions."""
        pos0 = self.pos_embed(torch.zeros_like(a))
        emb_a = self.concept_embed(a) + pos0
        
        placeholder = torch.zeros_like(a)
        pos1 = self.pos_embed(placeholder + 1)
        placeholder_emb = torch.zeros((a.shape[0], self.embed_dim), device=C.DEVICE)
        
        x = torch.cat([
            emb_a,
            self.action_embed(action),
            placeholder_emb + pos1,
        ], dim=-1)
        
        h = F.leaky_relu(self.fc1(x), negative_slope=self.leaky_relu_alpha)
        h = F.leaky_relu(self.fc2(h), negative_slope=self.leaky_relu_alpha)
        concept_logits = self.fc3(h)
        return concept_logits
    
    def forward_operation(self, a, action, b):
        """Forward pass for operation actions."""
        pos0 = self.pos_embed(torch.zeros_like(a))
        pos1 = self.pos_embed(torch.zeros_like(b) + 1)
        
        emb_a = self.concept_embed(a) + pos0
        emb_b = self.concept_embed(b) + pos1
        
        x = torch.cat([
            emb_a,
            self.action_embed(action),
            emb_b,
        ], dim=-1)
        h = F.leaky_relu(self.fc1(x), negative_slope=self.leaky_relu_alpha)
        h = F.leaky_relu(self.fc2(h), negative_slope=self.leaky_relu_alpha)
        concept_logits = self.fc3(h)
        return concept_logits
    
    def inverse_leaky_relu(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse of leaky ReLU activation.
        
        Args:
            y: Output of leaky ReLU
            
        Returns:
            Input that would produce y
        """
        return torch.where(y >= 0, y, y / self.leaky_relu_alpha)
    
    def backward_pass(self, concept_logits: torch.Tensor, is_relation: bool = False) -> Dict[str, torch.Tensor]:
        """Run network backwards from output to input (dreaming).
        
        This performs approximate inversion using transposed weights.
        
        Args:
            concept_logits: Output logits to invert
            is_relation: Whether this is a relation or operation
            
        Returns:
            Dictionary with reconstructed embeddings
        """
        # Backward through fc3: h2 = W3^T @ concept_logits
        h2 = torch.matmul(concept_logits, self.fc3.weight)  # [batch, hidden_dim // 2]
        
        # Inverse activation
        h1 = self.inverse_leaky_relu(h2)
        
        # Backward through fc2: h0 = W2^T @ h1
        h0 = torch.matmul(h1, self.fc2.weight)  # [batch, hidden_dim]
        
        # Inverse activation
        x = self.inverse_leaky_relu(h0)
        
        # Backward through fc1: input = W1^T @ x
        reconstructed = torch.matmul(x, self.fc1.weight)  # [batch, embed_dim * 3]
        
        # Split into components
        emb_a = reconstructed[:, :self.embed_dim]
        action_emb = reconstructed[:, self.embed_dim:self.embed_dim*2]
        emb_b = reconstructed[:, self.embed_dim*2:]
        
        return {
            'concept_a_emb': emb_a,
            'action_emb': action_emb,
            'concept_b_emb': emb_b,
            'hidden_1': h0,
            'hidden_2': h1,
            'reconstructed_input': reconstructed
        }
    
    def dream_concept(self, concept_logits: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
        """Dream: find concepts that would produce these logits.
        
        Args:
            concept_logits: Target output logits
            top_k: Number of top concepts to return
            
        Returns:
            List of (concept_id, similarity_score) tuples
        """
        # Run backward pass
        backward_result = self.backward_pass(concept_logits)
        reconstructed_emb = backward_result['concept_a_emb']
        
        # Find nearest neighbors in embedding space
        all_embeddings = self.concept_embed.weight  # [vocab_size, embed_dim]
        
        # Cosine similarity
        reconstructed_norm = F.normalize(reconstructed_emb, dim=-1)
        embeddings_norm = F.normalize(all_embeddings, dim=-1)
        similarities = torch.matmul(reconstructed_norm, embeddings_norm.T)  # [batch, vocab_size]
        
        # Get top-k
        top_scores, top_indices = torch.topk(similarities[0], k=top_k)
        
        return [(idx.item(), score.item()) for idx, score in zip(top_indices, top_scores)]


class DualNetworkModel(nn.Module):
    """Dual-network system combining fast and slow learners.
    
    This implements a complementary learning system where:
    - Fast learner adapts quickly to new experiences (hippocampus)
    - Slow learner maintains stable long-term knowledge (cortex)
    - Consolidation transfers knowledge from fast to slow during "sleep"
    """
    
    def __init__(self, vocab_size: int, embedding_matrix: torch.Tensor | None = None, 
                 freeze_embeddings: bool = False):
        super().__init__()
        
        self.vocab_size = vocab_size
        
        # Initialize both networks
        if embedding_matrix is not None:
            # Fast learner gets projected embeddings
            self.fast_learner = FastLearner(vocab_size, embedding_matrix, freeze_embeddings)
            # Slow learner gets full embeddings
            self.slow_learner = SlowLearner(vocab_size, embedding_matrix, freeze_embeddings)
        else:
            self.fast_learner = FastLearner(vocab_size, None, freeze_embeddings)
            self.slow_learner = SlowLearner(vocab_size, None, freeze_embeddings)
        
        # Shared action proposal network
        # Uses slow learner's embedding dimension for compatibility
        self.action_proposal_embed_dim = C.SLOW_EMBED_DIM
        self.action_proposal_hidden = 128
        
        self.action_concept_embed = nn.Embedding(vocab_size, self.action_proposal_embed_dim)
        self.action_pos_embed = nn.Embedding(2, self.action_proposal_embed_dim)
        
        # Initialize with slow learner's embeddings if available
        if embedding_matrix is not None:
            if embedding_matrix.shape[1] == self.action_proposal_embed_dim:
                self.action_concept_embed.weight.data.copy_(embedding_matrix)
            self.action_concept_embed.weight.requires_grad = not freeze_embeddings
        
        # Action proposal network
        self.action_fc1 = nn.Linear(self.action_proposal_embed_dim * 2, self.action_proposal_hidden)
        self.action_fc2 = nn.Linear(self.action_proposal_hidden, len(C.ACTION_TOKENS))
        
        # Track which network to use during forward pass
        self.use_fast = True  # Default to fast learner during interaction
        
        # Consolidation statistics
        self.consolidation_count = 0
        self.interaction_steps = 0
    
    def forward(self, a, action, b=None, is_relation=False):
        """Forward pass through the active network."""
        if self.use_fast:
            return self.fast_learner(a, action, b, is_relation)
        else:
            return self.slow_learner(a, action, b, is_relation)
    
    def propose_action(self, a, b):
        """Propose action using shared action proposal network.
        
        This network is shared between fast and slow learners,
        as action proposal is a meta-task that doesn't require
        separate episodic and semantic memory systems.
        """
        pos0 = self.action_pos_embed(torch.zeros_like(a))
        pos1 = self.action_pos_embed(torch.zeros_like(b) + 1)
        
        emb_a = self.action_concept_embed(a) + pos0
        emb_b = self.action_concept_embed(b) + pos1
        
        x = torch.cat([emb_a, emb_b], dim=-1)
        h = F.relu(self.action_fc1(x))
        action_logits = self.action_fc2(h)
        
        return action_logits
    
    def increment_interaction_steps(self):
        """Increment the interaction step counter."""
        self.interaction_steps += 1
    
    def get_active_network(self) -> str:
        """Return the name of the currently active network.
        
        Returns:
            'fast_learner' or 'slow_learner'
        """
        return 'fast_learner' if self.use_fast else 'slow_learner'
    
    def use_fast_learner(self):
        """Switch to using the fast learner for forward passes."""
        self.use_fast = True
    
    def use_slow_learner(self):
        """Switch to using the slow learner for forward passes."""
        self.use_fast = False
    
    def sync_weights(self, alpha: float = 0.1, direction: str = 'slow_to_fast'):
        """Synchronize weights between networks using exponential moving average.
        
        Args:
            alpha: EMA coefficient (0 = no update, 1 = full copy)
            direction: 'slow_to_fast' or 'fast_to_slow'
        """
        if direction == 'slow_to_fast':
            # Regularize fast learner toward slow learner (stability)
            for fast_param, slow_param in zip(self.fast_learner.parameters(), 
                                             self.slow_learner.parameters()):
                if fast_param.shape == slow_param.shape:
                    fast_param.data = (1 - alpha) * fast_param.data + alpha * slow_param.data
        elif direction == 'fast_to_slow':
            # Update slow learner from fast learner (rare, for rapid adaptation)
            for fast_param, slow_param in zip(self.fast_learner.parameters(), 
                                             self.slow_learner.parameters()):
                if fast_param.shape == slow_param.shape:
                    slow_param.data = (1 - alpha) * slow_param.data + alpha * fast_param.data
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def get_network_stats(self, logger=None) -> Dict[str, int]:
        """Get statistics about network sizes."""
        fast_params = sum(p.numel() for p in self.fast_learner.parameters())
        slow_params = sum(p.numel() for p in self.slow_learner.parameters())
        
        stats = {
            "fast_learner_params": fast_params,
            "slow_learner_params": slow_params,
            "param_ratio": slow_params / fast_params if fast_params > 0 else 0,
            "interaction_steps": self.interaction_steps,
            "consolidation_count": self.consolidation_count
        }
        
        # Add logger stats if available
        if logger:
            logger_stats = logger.get_stats()
            stats["total_logged_experiences"] = logger_stats.get("total_steps", 0)
            stats["accuracy"] = logger_stats.get("accuracy", 0.0)
        
        return stats
