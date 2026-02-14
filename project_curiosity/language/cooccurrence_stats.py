"""Co-occurrence statistics for generative replay.

Tracks which tokens and actions fire together to enable sampling
synthetic training examples from the learned distribution.
"""
from __future__ import annotations

import random
from typing import Dict, Tuple, Optional
from collections import defaultdict


class TokenCooccurrenceStats:
    """Track co-occurrence statistics for generative replay.
    
    For each token, tracks which (action, other_token) pairs it has been
    seen with, along with their frequencies. This enables sampling synthetic
    training examples from the learned distribution.
    """
    
    def __init__(self, vocab_size: int, num_actions: int):
        """Initialize co-occurrence tracker.
        
        Args:
            vocab_size: Size of vocabulary
            num_actions: Number of action types
        """
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        
        # token_id -> action_id -> other_token_id -> count
        self.cooccurrence = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Track total updates for statistics
        self.total_updates = 0
    
    def update(self, concept_a_id: int, action_id: int, concept_b_id: int):
        """Update statistics from a training example.
        
        Records that concept_a was seen with (action, concept_b).
        Also records the reverse: concept_b was seen with (action, concept_a).
        
        Args:
            concept_a_id: First concept ID
            action_id: Action ID
            concept_b_id: Second concept ID (can be None for relation actions)
        """
        # Forward: concept_a -> (action, concept_b)
        self.cooccurrence[concept_a_id][action_id][concept_b_id] += 1
        
        # Reverse: concept_b -> (action, concept_a) if concept_b exists
        if concept_b_id is not None:
            self.cooccurrence[concept_b_id][action_id][concept_a_id] += 1
        
        self.total_updates += 1
    
    def sample_for_token(self, token_id: int) -> Tuple[Optional[int], Optional[int]]:
        """Sample an action and associated token for given token.
        
        Samples proportionally to observed frequencies.
        
        Args:
            token_id: Token to sample associations for
            
        Returns:
            Tuple of (action_id, other_token_id) or (None, None) if no data
        """
        if token_id not in self.cooccurrence:
            return None, None
        
        # Get all actions for this token
        action_data = self.cooccurrence[token_id]
        if not action_data:
            return None, None
        
        # Sample action proportional to total frequency
        action_counts = {}
        for action_id, token_counts in action_data.items():
            action_counts[action_id] = sum(token_counts.values())
        
        if not action_counts:
            return None, None
        
        # Weighted sampling of action
        actions = list(action_counts.keys())
        weights = [action_counts[a] for a in actions]
        action_id = random.choices(actions, weights=weights)[0]
        
        # Sample associated token for this action
        token_counts = action_data[action_id]
        tokens = list(token_counts.keys())
        weights = [token_counts[t] for t in tokens]
        other_token_id = random.choices(tokens, weights=weights)[0]
        
        return action_id, other_token_id
    
    def sample_random_pair(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Sample a random (concept_a, action, concept_b) triple from distribution.
        
        Returns:
            Tuple of (concept_a_id, action_id, concept_b_id) or (None, None, None)
        """
        if not self.cooccurrence:
            return None, None, None
        
        # Sample a random token that has co-occurrence data
        tokens_with_data = list(self.cooccurrence.keys())
        if not tokens_with_data:
            return None, None, None
        
        concept_a_id = random.choice(tokens_with_data)
        
        # Sample action and associated token
        action_id, concept_b_id = self.sample_for_token(concept_a_id)
        
        if action_id is None:
            return None, None, None
        
        return concept_a_id, action_id, concept_b_id
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about co-occurrence data.
        
        Returns:
            Dictionary with statistics
        """
        num_tokens_with_data = len(self.cooccurrence)
        
        total_pairs = 0
        total_actions = 0
        for token_data in self.cooccurrence.values():
            total_actions += len(token_data)
            for token_counts in token_data.values():
                total_pairs += len(token_counts)
        
        avg_actions_per_token = total_actions / num_tokens_with_data if num_tokens_with_data > 0 else 0
        avg_pairs_per_action = total_pairs / total_actions if total_actions > 0 else 0
        
        return {
            "total_updates": self.total_updates,
            "tokens_with_data": num_tokens_with_data,
            "total_action_types": total_actions,
            "total_pairs": total_pairs,
            "avg_actions_per_token": avg_actions_per_token,
            "avg_pairs_per_action": avg_pairs_per_action,
            "coverage": num_tokens_with_data / self.vocab_size if self.vocab_size > 0 else 0
        }
    
    def get_top_associations(self, token_id: int, top_k: int = 5) -> list:
        """Get top-k most frequent associations for a token.
        
        Args:
            token_id: Token to get associations for
            top_k: Number of top associations to return
            
        Returns:
            List of (action_id, other_token_id, count) tuples
        """
        if token_id not in self.cooccurrence:
            return []
        
        associations = []
        for action_id, token_counts in self.cooccurrence[token_id].items():
            for other_token_id, count in token_counts.items():
                associations.append((action_id, other_token_id, count))
        
        # Sort by count descending
        associations.sort(key=lambda x: x[2], reverse=True)
        
        return associations[:top_k]
    
    def save(self, filepath: str):
        """Save co-occurrence statistics to file.
        
        Args:
            filepath: Path to save to
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'num_actions': self.num_actions,
                'cooccurrence': dict(self.cooccurrence),
                'total_updates': self.total_updates
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'TokenCooccurrenceStats':
        """Load co-occurrence statistics from file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            TokenCooccurrenceStats instance
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        stats = cls(data['vocab_size'], data['num_actions'])
        stats.cooccurrence = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int)),
            data['cooccurrence']
        )
        stats.total_updates = data['total_updates']
        
        return stats
    
    def load_from_log(self, log_file: str, vocab_encode_fn, action_tokens: list):
        """Load co-occurrence statistics from training log file.
        
        This reads the training_log.jsonl file and builds co-occurrence
        statistics from all logged interactions. Useful for:
        - Warm starting from previous training
        - Resuming training with existing patterns
        - Offline analysis of logged data
        
        Args:
            log_file: Path to training_log.jsonl
            vocab_encode_fn: Function to encode tokens to IDs
            action_tokens: List of action token strings
        
        Returns:
            Number of entries processed
        """
        import json
        import os
        
        if not os.path.exists(log_file):
            return 0
        
        entries_processed = 0
        
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    
                    # Only process valid questions
                    if not entry.get('is_valid_question', True):
                        continue
                    
                    # Extract concepts and action
                    concept_a = entry.get('concept_a')
                    action = entry.get('action')
                    
                    if not concept_a or not action:
                        continue
                    
                    # Check if action is valid
                    if action not in action_tokens:
                        continue
                    action_id = action_tokens.index(action)
                    
                    # Encode concept_a
                    a_id = vocab_encode_fn(concept_a)
                    
                    # Handle concept_b:
                    # - For operations: use the actual concept_b (second operand)
                    # - For relations: concept_b is None (no second concept needed)
                    from . import config as C
                    concept_b = entry.get('concept_b')
                    
                    if action in C.RELATION_ACTIONS:
                        # Relational actions: no concept_b needed
                        b_id = None
                    else:
                        # Operations: need concept_b
                        if not concept_b:
                            continue  # Skip if operation is missing concept_b
                        b_id = vocab_encode_fn(concept_b)
                    
                    # Update co-occurrence
                    self.update(a_id, action_id, b_id)
                    entries_processed += 1
                    
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Skip malformed entries
                    continue
        
        return entries_processed
    
    @classmethod
    def from_log(cls, log_file: str, vocab_size: int, num_actions: int, 
                 vocab_encode_fn, action_tokens: list) -> 'TokenCooccurrenceStats':
        """Create co-occurrence statistics from training log file.
        
        Args:
            log_file: Path to training_log.jsonl
            vocab_size: Size of vocabulary
            num_actions: Number of action types
            vocab_encode_fn: Function to encode tokens to IDs
            action_tokens: List of action token strings
            
        Returns:
            TokenCooccurrenceStats instance populated from log
        """
        stats = cls(vocab_size, num_actions)
        entries = stats.load_from_log(log_file, vocab_encode_fn, action_tokens)
        print(f"Loaded co-occurrence stats from {entries} log entries")
        return stats
