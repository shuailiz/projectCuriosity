"""Dual-Network Trainer for Continual Learning.

This trainer manages the interaction between fast and slow learners,
implementing the hippocampus-cortex analogy for continual learning.
"""
from __future__ import annotations

import json
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from . import config as C
from .vocabulary import Vocabulary
from .dual_network_model_language import DualNetworkModel
from .cooccurrence_stats import TokenCooccurrenceStats
from .llm import ask
from . import questions as Q
from . import actions


class DualNetworkTrainer:
    """Trainer for dual-network continual learning system."""
    
    def __init__(self, vocab: Vocabulary, logger=None,
                 fast_lr: float = None, slow_lr: float = None):
        """Initialize dual-network trainer.
        
        Args:
            vocab: Vocabulary instance
            logger: Training logger
            fast_lr: Learning rate for fast learner (default: from config)
            slow_lr: Learning rate for slow learner (default: from config)
        """
        self.vocab = vocab
        self.logger = logger
        self.step_count = 0
        
        # Use provided learning rates or defaults from config
        self.fast_lr = fast_lr if fast_lr is not None else C.FAST_LEARNING_RATE
        self.slow_lr = slow_lr if slow_lr is not None else C.SLOW_LEARNING_RATE
        
        # Initialize co-occurrence tracker for generative replay
        self.cooccurrence_stats = TokenCooccurrenceStats(
            vocab_size=C.VOCAB_SIZE,
            num_actions=len(C.ACTION_TOKENS)
        )
        
        # Load pretrained embeddings
        from .embeddings import load_pretrained, build_embedding_matrix
        kv = load_pretrained()
        emb_matrix = build_embedding_matrix(self.vocab.tokens, kv).to(C.DEVICE)
        
        # Initialize dual network model
        self.model = DualNetworkModel(
            C.VOCAB_SIZE, 
            emb_matrix, 
            freeze_embeddings=False
        ).to(C.DEVICE)
        
        # Separate optimizers for fast and slow learners
        self.fast_opt = torch.optim.Adam(
            self.model.fast_learner.parameters(), 
            lr=self.fast_lr
        )
        self.slow_opt = torch.optim.Adam(
            self.model.slow_learner.parameters(), 
            lr=self.slow_lr
        )
        
        # Shared action proposal optimizer
        self.action_opt = torch.optim.Adam(
            list(self.model.action_concept_embed.parameters()) +
            list(self.model.action_pos_embed.parameters()) +
            list(self.model.action_fc1.parameters()) +
            list(self.model.action_fc2.parameters()),
            lr=C.LEARNING_RATE
        )
        
        print(f"\n{'='*80}")
        print("Dual-Network System Initialized")
        print(f"{'='*80}")
        stats = self.model.get_network_stats()
        print(f"Fast Learner (Hippocampus):")
        print(f"  - Parameters: {stats['fast_learner_params']:,}")
        print(f"  - Learning Rate: {self.fast_lr}")
        print(f"  - Embed Dim: {C.FAST_EMBED_DIM}, Hidden Dim: {C.FAST_HIDDEN_DIM}")
        print(f"Slow Learner (Cortex):")
        print(f"  - Parameters: {stats['slow_learner_params']:,}")
        print(f"  - Learning Rate: {self.slow_lr}")
        print(f"  - Embed Dim: {C.SLOW_EMBED_DIM}, Hidden Dim: {C.SLOW_HIDDEN_DIM}")
        print(f"\nParameter Ratio (Slow/Fast): {stats['param_ratio']:.2f}x")
        print(f"Consolidation Interval: {C.CONSOLIDATION_INTERVAL} steps")
        print(f"Replay from Training Logger: Yes")
        print(f"{'='*80}\n")
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def _sample_pair(self):
        """Sample a random pair of concepts."""
        return self.vocab.random_concept(), self.vocab.random_concept()
    
    def train_step(self, total_steps: int = None) -> Dict[str, any]:
        """Run one training step with the dual-network system.
        
        During interaction phase (daytime):
        - Fast learner adapts quickly to new experiences
        - Experiences are stored in replay buffer
        - Periodic consolidation transfers knowledge to slow learner
        
        Returns:
            Dictionary with training information
        """
        # Sample concepts
        a_id, b_id = self._sample_pair()
        
        # Propose action using shared action proposal network
        action_logits_prop = self.model.propose_action(
            torch.tensor([a_id], device=C.DEVICE),
            torch.tensor([b_id], device=C.DEVICE),
        )
        act_id = torch.multinomial(F.softmax(action_logits_prop, dim=-1), 1).item()
        act_tok = C.ACTION_TOKENS[act_id]
        
        # Get token strings
        a_tok, b_tok = self.vocab.decode(a_id), self.vocab.decode(b_id)
        
        # Check if this is a relation action
        is_relation = act_tok in C.RELATION_ACTIONS
        
        # Forward pass through slow learner (predictions shown to user/LLM)
        self.model.use_fast = False
        
        if is_relation:
            concept_logits = self.model(
                torch.tensor([a_id], device=C.DEVICE),
                torch.tensor([act_id], device=C.DEVICE),
                is_relation=True
            )
        else:
            concept_logits = self.model(
                torch.tensor([a_id], device=C.DEVICE),
                torch.tensor([act_id], device=C.DEVICE),
                torch.tensor([b_id], device=C.DEVICE),
                is_relation=False
            )
        
        # Prepare model outputs
        model_output = {
            "concept_logits": concept_logits,
            "action_logits_prop": action_logits_prop,
            "act_id": act_id,
        }
        
        # Get action handler
        handler = actions.get_handler(act_tok)
        
        # Increment step counter
        self.step_count += 1
        
        # Handle action-specific logic
        result = handler(
            model_output=model_output,
            concept_a=a_tok,
            concept_b=b_tok,
            vocab_decode_fn=self.vocab.decode,
            vocab_add_fn=self.vocab.add,
            logger=self.logger,
            step=self.step_count,
        )
        
        # Handle invalid questions
        if result.get("skip", False):
            if "action_loss" in result:
                action_loss = result["action_loss"]
                self.action_opt.zero_grad()
                action_loss.backward()
                self.action_opt.step()

                return {"skip": True, "action_loss": float(action_loss.item())}
            return {"skip": True}
        
        # Compute losses for both networks
        slow_loss = result["loss"]  # Standard cross-entropy loss
        
        # Get fast learner's prediction
        self.model.use_fast = True
        if is_relation:
            fast_concept_logits = self.model(
                torch.tensor([a_id], device=C.DEVICE),
                torch.tensor([act_id], device=C.DEVICE),
                is_relation=True
            )
        else:
            fast_concept_logits = self.model(
                torch.tensor([a_id], device=C.DEVICE),
                torch.tensor([act_id], device=C.DEVICE),
                torch.tensor([b_id], device=C.DEVICE),
                is_relation=False
            )
        
        # Fast learner loss: KL divergence from slow learner's output (knowledge distillation)
        slow_probs = F.softmax(concept_logits.detach(), dim=-1)
        fast_log_probs = F.log_softmax(fast_concept_logits, dim=-1)
        fast_loss = F.kl_div(fast_log_probs, slow_probs, reduction='batchmean')
        
        # Backprop through both networks
        self.slow_opt.zero_grad()
        self.fast_opt.zero_grad()
        slow_loss.backward()  # Slow learner learns from ground truth
        fast_loss.backward()  # Fast learner learns from slow learner
        self.slow_opt.step()
        self.fast_opt.step()
        
        total_loss = slow_loss.item() + fast_loss.item()
        
        # Increment interaction step counter
        self.model.increment_interaction_steps()
        
        # Periodic consolidation (sleep phase)
        consolidation_info = None
        # Use total_steps if provided, otherwise fall back to step_count
        steps_for_consolidation = total_steps if total_steps is not None else self.step_count
        if steps_for_consolidation % C.CONSOLIDATION_INTERVAL == 0:
            # Reload co-occurrence stats from log before consolidation
            if self.logger and hasattr(self.logger, 'log_file'):
                import os
                import shutil
                import time
                
                if os.path.exists(self.logger.log_file):
                    # Clear existing stats and reload from log
                    self.cooccurrence_stats = TokenCooccurrenceStats(
                        vocab_size=C.VOCAB_SIZE,
                        num_actions=len(C.ACTION_TOKENS)
                    )
                    entries = self.cooccurrence_stats.load_from_log(
                        self.logger.log_file,
                        vocab_encode_fn=self.vocab.encode,
                        action_tokens=C.ACTION_TOKENS
                    )
                    print(f"\n💤 Preparing for sleep: loaded {entries} interactions from log")
            
            # Perform consolidation (sleep)
            consolidation_info = self.consolidate(mode=C.CONSOLIDATION_MODE)
            
            # Archive log file after sleep and start fresh
            if self.logger and hasattr(self.logger, 'log_file'):
                if os.path.exists(self.logger.log_file):
                    # Create archive with timestamp
                    cycle_num = steps_for_consolidation // C.CONSOLIDATION_INTERVAL
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    archive_path = f"{self.logger.log_file}.cycle{cycle_num}_{timestamp}.archive"
                    
                    # Copy to archive
                    shutil.copy(self.logger.log_file, archive_path)
                    print(f"📁 Archived log to: {os.path.basename(archive_path)}")
                    
                    # Clear log file for next awake phase
                    open(self.logger.log_file, 'w').close()
                    print(f"🌅 Starting new awake phase with fresh log\n")
        
        # Prepare return info
        train_info = dict(
            concept_a=a_tok,
            concept_b=b_tok,
            action=act_tok,
            model_answer=result["model_answer"],
            is_correct=result["is_correct"],
            correct_answer=result["correct_answer"],
            loss=total_loss,
            step=self.step_count,
        )
        
        if consolidation_info:
            train_info["consolidation"] = consolidation_info
        
        return train_info
    
    def _perform_consolidation_if_needed(self, steps_for_consolidation: int) -> Dict[str, any]:
        """Helper method to perform consolidation if it's time.
        
        Returns:
            Consolidation info dict or None
        """
        if steps_for_consolidation % C.CONSOLIDATION_INTERVAL != 0:
            return None
        
        # Reload co-occurrence stats from log before consolidation
        if self.logger and hasattr(self.logger, 'log_file'):
            import os
            import shutil
            import time
            
            if os.path.exists(self.logger.log_file):
                # Clear existing stats and reload from log
                self.cooccurrence_stats = TokenCooccurrenceStats(
                    vocab_size=C.VOCAB_SIZE,
                    num_actions=len(C.ACTION_TOKENS)
                )
                print(self.logger.log_file)
                entries = self.cooccurrence_stats.load_from_log(
                    self.logger.log_file,
                    vocab_encode_fn=self.vocab.encode,
                    action_tokens=C.ACTION_TOKENS
                )
                print(f"\n💤 Preparing for sleep: loaded {entries} interactions from log")
        
        # Perform consolidation (sleep)
        consolidation_info = self.consolidate(mode=C.CONSOLIDATION_MODE)
        
        # Archive log file after sleep and start fresh
        if self.logger and hasattr(self.logger, 'log_file'):
            import os
            import shutil
            import time
            
            if os.path.exists(self.logger.log_file):
                # Create archive with timestamp
                cycle_num = steps_for_consolidation // C.CONSOLIDATION_INTERVAL
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                archive_path = f"{self.logger.log_file}.cycle{cycle_num}_{timestamp}.archive"
                
                # Copy to archive
                shutil.copy(self.logger.log_file, archive_path)
                print(f"📁 Archived log to: {os.path.basename(archive_path)}")
                
                # Clear log file for next awake phase
                open(self.logger.log_file, 'w').close()
                print(f"🌅 Starting new awake phase with fresh log\n")
        
        return consolidation_info
    
    def train_step_with_human_feedback(self, total_steps: int = None) -> Dict[str, any]:
        """Run one training step with human feedback instead of LLM.
        
        Strategy:
        1. Sample concepts and propose action
        2. Slow learner makes predictions (shown to user)
        3. Get human feedback through action handler
        4. Compute slow learner loss (standard cross-entropy)
        5. Compute fast learner loss (KL divergence from slow learner's output)
        6. Backprop through both networks
        
        Returns:
            Dictionary with training information
        """
        # Sample concepts
        a_id, b_id = self._sample_pair()
        
        # Use slow learner for predictions shown to user
        self.model.use_fast = False
        
        # Propose action using slow learner
        action_logits_prop = self.model.propose_action(
            torch.tensor([a_id], device=C.DEVICE),
            torch.tensor([b_id], device=C.DEVICE),
        )
        act_id = torch.multinomial(F.softmax(action_logits_prop, dim=-1), 1).item()
        act_tok = C.ACTION_TOKENS[act_id]
        
        # Get token strings
        a_tok, b_tok = self.vocab.decode(a_id), self.vocab.decode(b_id)
        
        # Check if this is a relation action
        is_relation = act_tok in C.RELATION_ACTIONS
        
        # Forward pass through slow learner (for user interaction)
        if is_relation:
            slow_concept_logits = self.model(
                torch.tensor([a_id], device=C.DEVICE),
                torch.tensor([act_id], device=C.DEVICE),
                is_relation=True
            )
        else:
            slow_concept_logits = self.model(
                torch.tensor([a_id], device=C.DEVICE),
                torch.tensor([act_id], device=C.DEVICE),
                torch.tensor([b_id], device=C.DEVICE),
                is_relation=False
            )
        
        # Prepare model outputs (using slow learner's predictions)
        model_output = {
            "concept_logits": slow_concept_logits,
            "action_logits_prop": action_logits_prop,
            "act_id": act_id,
        }
        
        # Get action handler with human feedback enabled
        handler = actions.get_handler(act_tok, human_feedback=True)
        
        # Increment step counter
        self.step_count += 1
        
        # Handle action-specific logic with human feedback
        result = handler(
            model_output=model_output,
            concept_a=a_tok,
            concept_b=b_tok,
            vocab_decode_fn=self.vocab.decode,
            vocab_add_fn=self.vocab.add,
            logger=self.logger,
            step=self.step_count,
        )
        
        # Check if we need to consolidate (before early return for skipped steps)
        steps_for_consolidation = total_steps if total_steps is not None else self.step_count
        
        # Handle invalid questions
        if result.get("skip", False):
            # Still do consolidation if it's time, even for skipped steps
            consolidation_info = self._perform_consolidation_if_needed(steps_for_consolidation)
            
            # Return skip result
            skip_result = {"skip": True}
            if "action_loss" in result:
                action_loss = result["action_loss"]
                # Update shared action proposal network
                self.action_opt.zero_grad()
                action_loss.backward()
                self.action_opt.step()
                skip_result["action_loss"] = float(action_loss.item())
            if consolidation_info:
                skip_result["consolidation"] = consolidation_info
            return skip_result
        
        # Compute losses for both networks
        slow_loss = result["loss"]  # Standard cross-entropy loss
        
        # Get fast learner's prediction
        self.model.use_fast = True
        if is_relation:
            fast_concept_logits = self.model(
                torch.tensor([a_id], device=C.DEVICE),
                torch.tensor([act_id], device=C.DEVICE),
                is_relation=True
            )
        else:
            fast_concept_logits = self.model(
                torch.tensor([a_id], device=C.DEVICE),
                torch.tensor([act_id], device=C.DEVICE),
                torch.tensor([b_id], device=C.DEVICE),
                is_relation=False
            )
        
        # Fast learner loss: KL divergence from slow learner's output (knowledge distillation)
        slow_probs = F.softmax(slow_concept_logits.detach(), dim=-1)
        fast_log_probs = F.log_softmax(fast_concept_logits, dim=-1)
        fast_loss = F.kl_div(fast_log_probs, slow_probs, reduction='batchmean')
        
        # Backprop through both networks
        self.slow_opt.zero_grad()
        self.fast_opt.zero_grad()
        slow_loss.backward()  # Slow learner learns from ground truth
        fast_loss.backward()  # Fast learner learns from slow learner
        self.slow_opt.step()
        self.fast_opt.step()
        
        total_loss = slow_loss.item() + fast_loss.item()
        
        # Increment interaction step counter
        self.model.increment_interaction_steps()
        
        # Periodic consolidation (sleep phase)
        consolidation_info = self._perform_consolidation_if_needed(steps_for_consolidation)
        
        # Note: Logging is handled by the action handler, not here
        
        train_info = dict(
            concept_a=a_tok,
            concept_b=b_tok,
            action=act_tok,
            model_answer=result["model_answer"],
            is_correct=result["is_correct"],
            correct_answer=result["correct_answer"],
            loss=total_loss,
        )
        
        if consolidation_info:
            train_info["consolidation"] = consolidation_info
        
        return train_info
    
    def consolidate_deep_sleep(self, num_replays: int = None, temperature: float = 2.0) -> Dict[str, any]:
        """Perform deep sleep (NREM) consolidation: sample from learned distribution.
        
        This generates synthetic training examples by:
        1. Sampling (concept_a, action, concept_b) from co-occurrence distribution
        2. Getting fast learner's prediction (teacher)
        3. Training slow learner to match fast learner (student)
        
        Args:
            num_replays: Number of synthetic examples to generate
            temperature: Temperature for knowledge distillation (higher = softer)
            
        Returns:
            Dictionary with consolidation statistics
        """
        if num_replays is None:
            num_replays = C.CONSOLIDATION_REPLAYS
        
        print(f"\n{'='*60}")
        print(f"😴 DEEP SLEEP (NREM) - Knowledge Consolidation")
        print(f"{'='*60}")
        
        # Check if we have enough data
        stats = self.cooccurrence_stats.get_stats()
        print(stats)
        if stats['tokens_with_data'] < C.CONSOLIDATION_MIN_DATA:
            print(f"Not enough data for deep sleep yet (need {C.CONSOLIDATION_MIN_DATA} tokens, have {stats['tokens_with_data']})")
            print(f"{'='*60}\n")
            return {"status": "insufficient_data", "replays": 0}
        
        print(f"Co-occurrence stats: {stats['tokens_with_data']} tokens, {stats['total_pairs']} pairs")
        
        # Set modes
        self.model.fast_learner.eval()  # Teacher in eval mode
        self.model.slow_learner.train()  # Student in train mode
        self.model.use_fast = False  # Use slow learner for training
        
        total_distill_loss = 0.0
        replayed = 0
        skipped = 0
        
        for _ in range(num_replays):
            # 1. Sample (concept_a, action, concept_b) from distribution
            concept_a_id, action_id, concept_b_id = self.cooccurrence_stats.sample_random_pair()
            
            if action_id is None:
                skipped += 1
                continue
            
            is_relation = C.ACTION_TOKENS[action_id] in C.RELATION_ACTIONS
            
            # 2. Get teacher (fast learner) prediction - NO GRAD
            with torch.no_grad():
                fast_logits = self.model.fast_learner(
                    torch.tensor([concept_a_id], device=C.DEVICE),
                    torch.tensor([action_id], device=C.DEVICE),
                    torch.tensor([concept_b_id], device=C.DEVICE) if not is_relation and concept_b_id is not None else None,
                    is_relation=is_relation
                )
            
            # 3. Get student (slow learner) prediction - WITH GRAD
            slow_logits = self.model.slow_learner(
                torch.tensor([concept_a_id], device=C.DEVICE),
                torch.tensor([action_id], device=C.DEVICE),
                torch.tensor([concept_b_id], device=C.DEVICE) if not is_relation and concept_b_id is not None else None,
                is_relation=is_relation
            )
            
            # 4. Distillation loss: match fast learner's distribution
            distill_loss = F.kl_div(
                F.log_softmax(slow_logits / temperature, dim=-1),
                F.softmax(fast_logits / temperature, dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            # 5. Backprop through slow learner ONLY
            self.slow_opt.zero_grad()
            distill_loss.backward()
            self.slow_opt.step()
            
            total_distill_loss += distill_loss.item()
            replayed += 1
        
        # Switch back to fast learner
        self.model.use_fast = True
        self.model.fast_learner.train()
        
        avg_loss = total_distill_loss / replayed if replayed > 0 else 0
        
        print(f"Deep Sleep Results:")
        print(f"  - Generated: {replayed} synthetic experiences")
        print(f"  - Skipped: {skipped}")
        print(f"  - Avg Distillation Loss: {avg_loss:.4f}")
        print(f"  - Temperature: {temperature}")
        print(f"{'='*60}\n")
        
        return {
            "status": "success",
            "replays": replayed,
            "skipped": skipped,
            "avg_distill_loss": avg_loss,
            "mode": "generative",
            "temperature": temperature
        }
    
    def consolidate(self, num_replays: int = None, mode: str = 'deep') -> Dict[str, any]:
        """Perform consolidation: replay experiences and train slow learner.
        
        Supports two sleep modes:
        - 'deep': Deep sleep (NREM) - generative replay with knowledge distillation
        - 'rem': REM sleep - imagination and prediction consistency training
        - 'full': Full sleep cycle - both deep and REM
        
        This mimics biological sleep phases:
        - Deep sleep: Consolidates learned patterns from hippocampus to cortex
        - REM sleep: Imagines novel scenarios and enforces consistency
        
        Args:
            num_replays: Number of experiences to replay (default: from config)
            mode: Sleep mode ('deep', 'rem', or 'full')
            
        Returns:
            Dictionary with consolidation statistics
        """
        # Route to appropriate consolidation method
        if mode == 'deep':
            return self.consolidate_deep_sleep(num_replays)
        elif mode == 'rem':
            return self.consolidate_rem_sleep(num_replays)
        elif mode == 'full':
            return self.consolidate_full_sleep(num_replays)
        else:
            return {"status": "unknown_mode", "mode": mode}
    
    def consolidate_rem_sleep(self, num_replays: int = None) -> Dict[str, any]:
        """Perform REM sleep consolidation: bidirectional self-consistency.
        
        REM sleep focuses on the slow learner's internal coherence:
        1. Imagine novel scenarios (sample random concepts)
        2. Slow learner makes prediction (forward)
        3. Reconstruct input from prediction (backward)
        4. Predict again from reconstructed input (forward)
        5. Train for consistency: forward → backward → forward
        
        This is pure self-consistency training - the slow learner
        should be able to reconstruct its own reasoning.
        
        Args:
            num_replays: Number of imagined scenarios
            
        Returns:
            Dictionary with consolidation statistics
        """
        if num_replays is None:
            num_replays = C.CONSOLIDATION_REPLAYS
        
        print(f"\n{'='*60}")
        print(f"💭 REM SLEEP - Bidirectional Self-Consistency")
        print(f"{'='*60}")
        
        # Check if we have enough data
        stats = self.cooccurrence_stats.get_stats()
        if stats['tokens_with_data'] < C.CONSOLIDATION_MIN_DATA:
            print(f"Not enough data for REM sleep yet (need {C.CONSOLIDATION_MIN_DATA} tokens, have {stats['tokens_with_data']})")
            print(f"{'='*60}\n")
            return {"status": "insufficient_data", "replays": 0}
        
        print(f"Imagining {num_replays} novel scenarios...")
        
        # Slow learner in training mode
        self.model.slow_learner.train()
        
        total_bidirectional_loss = 0.0
        total_distillation_loss = 0.0
        replayed = 0
        
        for _ in range(num_replays):
            # 1. Sample a target concept directly
            target_concept_id = torch.randint(0, self.vocab.size, (1,), device=C.DEVICE).item()
            
            # 2. Create a soft distribution around the target concept using embedding similarity
            # This represents an "imagined" prediction with some uncertainty
            target_emb = self.model.slow_learner.concept_embed.weight[target_concept_id:target_concept_id+1]
            all_embeddings = self.model.slow_learner.concept_embed.weight
            
            # Use cosine similarity as logits (already normalized by cosine)
            similarities = F.cosine_similarity(target_emb, all_embeddings, dim=-1)
            
            # Use similarities as logits (softer distribution, more realistic)
            # Don't scale too much - we want the slow learner to be able to match this
            slow_logits_1 = (similarities * 2.0).unsqueeze(0)  # [1, vocab_size]
            
            # 3. Dream: what concepts would produce this output?
            dream_results = self.model.slow_learner.dream_concept(slow_logits_1, top_k=2)
            reconstructed_a_id, _ = dream_results[0]
            
            # Use second-best concept as reconstructed_b (if available)
            reconstructed_b_id = dream_results[1][0] if len(dream_results) > 1 else reconstructed_a_id
            
            # 4. Use action proposal network to decide what action to take
            action_logits = self.model.propose_action(
                torch.tensor([reconstructed_a_id], device=C.DEVICE),
                torch.tensor([reconstructed_b_id], device=C.DEVICE)
            )
            action_id = torch.multinomial(F.softmax(action_logits, dim=-1), 1).item()
            is_relation = C.ACTION_TOKENS[action_id] in C.RELATION_ACTIONS
            
            # 5. Slow learner makes prediction on dreamed scenario
            slow_logits_2 = self.model.slow_learner(
                torch.tensor([reconstructed_a_id], device=C.DEVICE),
                torch.tensor([action_id], device=C.DEVICE),
                torch.tensor([reconstructed_b_id], device=C.DEVICE) if not is_relation else None,
                is_relation=is_relation
            )
            
            # 6. Bidirectional consistency: slow learner's predictions should be consistent
            bidirectional_loss = F.kl_div(
                F.log_softmax(slow_logits_2, dim=-1),
                F.softmax(slow_logits_1.detach(), dim=-1),
                reduction='batchmean'
            )
            
            # Backprop through slow learner
            self.slow_opt.zero_grad()
            bidirectional_loss.backward()
            self.slow_opt.step()
            
            # 7. Knowledge distillation: fast learner learns from slow learner's prediction
            # Switch to fast learner temporarily
            self.model.use_fast = True
            fast_logits = self.model.fast_learner(
                torch.tensor([reconstructed_a_id], device=C.DEVICE),
                torch.tensor([action_id], device=C.DEVICE),
                torch.tensor([reconstructed_b_id], device=C.DEVICE) if not is_relation else None,
                is_relation=is_relation
            )
            self.model.use_fast = False  # Back to slow learner
            
            # Knowledge distillation loss
            distillation_loss = F.kl_div(
                F.log_softmax(fast_logits, dim=-1),
                F.softmax(slow_logits_2.detach(), dim=-1),
                reduction='batchmean'
            )
            
            # Backprop through fast learner
            self.fast_opt.zero_grad()
            distillation_loss.backward()
            self.fast_opt.step()
            
            total_bidirectional_loss += bidirectional_loss.item()
            total_distillation_loss += distillation_loss.item()
            replayed += 1
        
        if replayed == 0:
            print("No scenarios could be imagined")
            print(f"{'='*60}\n")
            return {"status": "no_scenarios", "replays": 0}
        
        # Back to fast learner for interaction
        self.model.use_fast = True
        
        avg_bidirectional = total_bidirectional_loss / replayed
        avg_distillation = total_distillation_loss / replayed
        
        print(f"REM Sleep Results:")
        print(f"  - Imagined: {replayed} scenarios")
        print(f"  - Avg Bidirectional Loss (Slow): {avg_bidirectional:.4f}")
        print(f"  - Avg Distillation Loss (Fast): {avg_distillation:.4f}")
        print(f"{'='*60}\n")
        
        return {
            "status": "success",
            "replays": replayed,
            "avg_bidirectional_loss": avg_bidirectional,
            "avg_distillation_loss": avg_distillation,
            "mode": "rem"
        }
    
    def consolidate_full_sleep(self, num_replays: int = None) -> Dict[str, any]:
        """Perform full sleep cycle: both deep sleep and REM sleep.
        
        Mimics natural sleep architecture:
        - 70% deep sleep (NREM) - knowledge consolidation
        - 30% REM sleep - imagination and consistency
        
        Args:
            num_replays: Total number of replays (split 70/30)
            
        Returns:
            Dictionary with consolidation statistics
        """
        if num_replays is None:
            num_replays = C.CONSOLIDATION_REPLAYS
        
        print(f"\n{'='*80}")
        print(f"🌙 FULL SLEEP CYCLE")
        print(f"{'='*80}")
        
        # Split replays: 70% deep sleep, 30% REM
        deep_replays = int(num_replays * 0.7)
        rem_replays = num_replays - deep_replays
        
        # Deep sleep phase
        print(f"\nPhase 1: Deep Sleep ({deep_replays} replays)")
        deep_result = self.consolidate_deep_sleep(deep_replays)
        
        # REM sleep phase
        print(f"\nPhase 2: REM Sleep ({rem_replays} replays)")
        rem_result = self.consolidate_rem_sleep(rem_replays)
        
        print(f"\nFull Sleep Cycle Complete")
        print(f"{'='*80}\n")
        
        return {
            "status": "success",
            "mode": "full_sleep",
            "deep_sleep": deep_result,
            "rem_sleep": rem_result,
            "total_replays": deep_result.get('replays', 0) + rem_result.get('replays', 0)
        }
    
    def get_stats(self) -> Dict[str, any]:
        """Get training statistics."""
        stats = {
            "step_count": self.step_count,
        }
        
        stats.update(self.model.get_network_stats(self.logger))
        # Add co-occurrence stats
        cooc_stats = self.cooccurrence_stats.get_stats()
        stats['cooccurrence'] = cooc_stats
        
        return stats
    
    def evaluate_network(self, network: str = 'fast', num_samples: int = 10) -> Dict[str, any]:
        """Evaluate a specific network on random samples.
        
        Args:
            network: 'fast' or 'slow'
            num_samples: Number of samples to evaluate
            
        Returns:
            Evaluation statistics
        """
        # Set which network to use
        original_mode = self.model.use_fast
        self.model.use_fast = (network == 'fast')
        
        correct = 0
        total = 0
        
        for _ in range(num_samples):
            a_id, b_id = self._sample_pair()
            
            # Propose action
            action_logits = self.model.propose_action(
                torch.tensor([a_id], device=C.DEVICE),
                torch.tensor([b_id], device=C.DEVICE),
            )
            act_id = torch.argmax(action_logits, dim=-1).item()
            act_tok = C.ACTION_TOKENS[act_id]
            
            # Get predictions
            a_tok, b_tok = self.vocab.decode(a_id), self.vocab.decode(b_id)
            is_relation = act_tok in C.RELATION_ACTIONS
            
            if is_relation:
                logits = self.model(
                    torch.tensor([a_id], device=C.DEVICE),
                    torch.tensor([act_id], device=C.DEVICE),
                    is_relation=True
                )
            else:
                logits = self.model(
                    torch.tensor([a_id], device=C.DEVICE),
                    torch.tensor([act_id], device=C.DEVICE),
                    torch.tensor([b_id], device=C.DEVICE),
                    is_relation=False
                )
            
            # Get prediction
            pred_id = torch.argmax(logits, dim=-1).item()
            pred_tok = self.vocab.decode(pred_id)
            
            # Simple correctness check (would need LLM for real evaluation)
            total += 1
        
        # Restore original mode
        self.model.use_fast = original_mode
        
        return {
            "network": network,
            "samples": num_samples,
            "total": total,
        }


def demo(steps=10):
    """Demo function for dual-network training."""
    vocab = Vocabulary(["apple", "banana", "orange", "fruit"])
    tr = DualNetworkTrainer(vocab)
    
    for s in range(1, steps + 1):
        result = tr.train_step()
        print(json.dumps({"step": s, **result}, ensure_ascii=False, indent=2))
        time.sleep(0.5)
    
    # Print final stats
    print("\n" + "="*80)
    print("Final Statistics:")
    print("="*80)
    stats = tr.get_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    demo(20)
