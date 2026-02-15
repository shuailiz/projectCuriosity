import os
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque
from typing import List, Tuple, Dict, Optional

from . import config as C
from .encoder import VisualEncoder
from .curiosity_policy import CuriosityPolicy
from ..dual_network_model import ContinuousDualNetworkModel

class VisualTrainer:
    """
    Trainer for the Visual-Motor Continuous Learning System.
    
    Each model has a dedicated folder under MODELS_DIR/<model_name>/ containing:
        config.json    - Frozen config snapshot (LRs, schedule, architecture)
        checkpoint.pt  - Latest weights, optimizer states, counters
        replay.pt      - Replay buffer (embeddings + actions, no raw frames)
        training.log   - Append-only CSV training log
    """
    def __init__(self, model_name="default"):
        self.model_name = model_name
        self.model_dir = os.path.join(C.MODELS_DIR, model_name)
        self.device = C.DEVICE
        
        # Model folder paths
        self._checkpoint_path = os.path.join(self.model_dir, "checkpoint.pt")
        self._replay_path = os.path.join(self.model_dir, "replay.pt")
        self._config_path = os.path.join(self.model_dir, "config.json")
        self._log_path = os.path.join(self.model_dir, "training.log")
        
        # Determine if this is a new model or resuming
        is_new = not os.path.exists(self._checkpoint_path)
        
        # Create model folder
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Components
        self.encoder = VisualEncoder()
        
        # Initialize Dual Network with config parameters
        self.model = ContinuousDualNetworkModel(
            input_dim=C.ENCODED_DIM + C.ACTION_DIM,
            hidden_dim_fast=C.FAST_HIDDEN_DIM,
            hidden_dim_slow=C.SLOW_HIDDEN_DIM,
            output_dim=C.ENCODED_DIM
        )
        self.model.to(C.DEVICE)
        
        # Optimizers — separate LRs per phase
        self.fast_opt = torch.optim.Adam(
            self.model.fast_learner.parameters(), lr=C.FAST_LR
        )
        self.slow_wake_opt = torch.optim.Adam(
            self.model.slow_learner.parameters(), lr=C.SLOW_WAKE_LR
        )
        self.slow_sleep_opt = torch.optim.Adam(
            self.model.slow_learner.parameters(), lr=C.SLOW_SLEEP_LR
        )
        
        # Curiosity Policy
        self.policy = CuriosityPolicy()
        self.policy_opt = torch.optim.Adam(
            self.policy.parameters(), lr=C.POLICY_LEARNING_RATE
        )
        
        # Replay Buffer
        self.replay_buffer = deque(maxlen=C.REPLAY_BUFFER_SIZE)
        
        self.step_count = 0
        self.wake_steps_in_cycle = 0
        self.cycle_count = 0
        self.loss_fn = nn.MSELoss()
        
        if is_new:
            # Save config snapshot for new model
            self._save_config_snapshot()
            self._init_training_log()
            print(f"New model '{model_name}' created at {self.model_dir}")
        else:
            # Resume existing model
            self._load_checkpoint()
            self._load_replay()
            print(f"Model '{model_name}' resumed from {self.model_dir}")
            print(f"  Step: {self.step_count}, Cycle: {self.cycle_count}, "
                  f"Replay: {len(self.replay_buffer)} entries")

    def _save_config_snapshot(self):
        """Save current config values as JSON for reproducibility."""
        config_snapshot = {
            # Architecture
            "encoder_model": C.ENCODER_MODEL,
            "encoded_dim": C.ENCODED_DIM,
            "action_dim": C.ACTION_DIM,
            "fast_hidden_dim": C.FAST_HIDDEN_DIM,
            "slow_hidden_dim": C.SLOW_HIDDEN_DIM,
            "policy_hidden_dim": C.POLICY_HIDDEN_DIM,
            # Learning rates
            "fast_lr": C.FAST_LR,
            "slow_wake_lr": C.SLOW_WAKE_LR,
            "slow_sleep_lr": C.SLOW_SLEEP_LR,
            "policy_lr": C.POLICY_LEARNING_RATE,
            # Wake schedule
            "wake_steps_per_cycle": C.WAKE_STEPS_PER_CYCLE,
            "slow_wake_update_interval": C.SLOW_WAKE_UPDATE_INTERVAL,
            "slow_wake_w_distill": C.SLOW_WAKE_W_DISTILL,
            "slow_wake_w_raw": C.SLOW_WAKE_W_RAW,
            # SWS schedule
            "sws_steps": C.SWS_STEPS,
            "sws_alpha": C.SWS_ALPHA,
            "sws_beta": C.SWS_BETA,
            # REM schedule
            "rem_steps": C.REM_STEPS,
            "rem_k": C.REM_K,
            "rem_gamma": C.REM_GAMMA,
            # Tether
            "tether_steps": C.TETHER_STEPS,
            "tether_lr": C.TETHER_LR,
            # Other
            "batch_size": C.BATCH_SIZE,
            "replay_buffer_size": C.REPLAY_BUFFER_SIZE,
            "curiosity_warmup_steps": C.CURIOSITY_WARMUP_STEPS,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self._config_path, 'w') as f:
            json.dump(config_snapshot, f, indent=2)

    def _init_training_log(self):
        """Create training log CSV with header."""
        with open(self._log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "event", "step", "cycle",
                "fast_loss", "slow_distill", "slow_raw",
                "curiosity_reward", "sws_L_dyn", "rem_L_ms",
                "tether_loss", "replay_size"
            ])

    def log_event(self, event, **kwargs):
        """Append a row to the training log CSV."""
        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"), event,
            kwargs.get("step", self.step_count),
            kwargs.get("cycle", self.cycle_count),
            kwargs.get("fast_loss", ""),
            kwargs.get("slow_distill", ""),
            kwargs.get("slow_raw", ""),
            kwargs.get("curiosity_reward", ""),
            kwargs.get("sws_L_dyn", ""),
            kwargs.get("rem_L_ms", ""),
            kwargs.get("tether_loss", ""),
            kwargs.get("replay_size", len(self.replay_buffer)),
        ]
        with open(self._log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def encode_frame(self, frame):
        """Encode a raw image frame into an embedding."""
        return self.encoder.encode(frame)

    # ---- Save / Load (model folder) ----

    def save(self):
        """Save everything to the model folder: checkpoint + replay."""
        self._save_checkpoint()
        self._save_replay()
        self.log_event("save")
        print(f"Model '{self.model_name}' saved (step={self.step_count}, "
              f"cycle={self.cycle_count}, replay={len(self.replay_buffer)})")

    def _save_checkpoint(self):
        """Save model weights, optimizer states, and counters."""
        checkpoint = {
            'fast_learner': self.model.fast_learner.state_dict(),
            'slow_learner': self.model.slow_learner.state_dict(),
            'policy': self.policy.state_dict(),
            'fast_opt': self.fast_opt.state_dict(),
            'slow_wake_opt': self.slow_wake_opt.state_dict(),
            'slow_sleep_opt': self.slow_sleep_opt.state_dict(),
            'policy_opt': self.policy_opt.state_dict(),
            'step_count': self.step_count,
            'wake_steps_in_cycle': self.wake_steps_in_cycle,
            'cycle_count': self.cycle_count,
            'consolidation_count': self.model.consolidation_count,
            'interaction_steps': self.model.interaction_steps,
        }
        torch.save(checkpoint, self._checkpoint_path)

    def _load_checkpoint(self):
        """Load model weights, optimizer states, and counters."""
        checkpoint = torch.load(
            self._checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.fast_learner.load_state_dict(checkpoint['fast_learner'])
        self.model.slow_learner.load_state_dict(checkpoint['slow_learner'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.fast_opt.load_state_dict(checkpoint['fast_opt'])
        self.slow_wake_opt.load_state_dict(checkpoint['slow_wake_opt'])
        self.slow_sleep_opt.load_state_dict(checkpoint['slow_sleep_opt'])
        self.policy_opt.load_state_dict(checkpoint['policy_opt'])
        self.step_count = checkpoint['step_count']
        self.wake_steps_in_cycle = checkpoint['wake_steps_in_cycle']
        self.cycle_count = checkpoint['cycle_count']
        self.model.consolidation_count = checkpoint['consolidation_count']
        self.model.interaction_steps = checkpoint['interaction_steps']

    def _save_replay(self):
        """Save replay buffer (embeddings + actions + joint positions, no raw frames)."""
        replay_data = []
        for exp in self.replay_buffer:
            replay_data.append({
                'state_emb': exp['state_emb'].cpu(),
                'action': exp['action'].cpu(),
                'commanded_action': exp['commanded_action'].cpu(),
                'next_state_emb': exp['next_state_emb'].cpu(),
                'timestamp': exp['timestamp'],
                'joint_positions': exp['joint_positions'].cpu(),
            })
        torch.save(replay_data, self._replay_path)

    def _load_replay(self):
        """Load replay buffer from disk."""
        if not os.path.exists(self._replay_path):
            return
        replay_data = torch.load(
            self._replay_path, map_location=self.device, weights_only=False
        )
        self.replay_buffer.clear()
        for entry in replay_data:
            action = entry['action'].to(self.device)
            
            # Handle backward compatibility: old replays may not have joint_positions
            if 'joint_positions' in entry:
                joint_pos = entry['joint_positions'].to(self.device)
            else:
                joint_pos = torch.zeros(C.ACTION_DIM, dtype=torch.float32, device=self.device)
            
            # Handle backward compatibility: old replays may not have commanded_action
            if 'commanded_action' in entry:
                commanded = entry['commanded_action'].to(self.device)
            else:
                commanded = action.clone()  # Assume no clipping for old data
            
            self.replay_buffer.append({
                'frame': None,
                'action': action,
                'commanded_action': commanded,
                'next_frame': None,
                'timestamp': entry['timestamp'],
                'state_emb': entry['state_emb'].to(self.device),
                'next_state_emb': entry['next_state_emb'].to(self.device),
                'joint_positions': joint_pos,
            })

    @staticmethod
    def list_models():
        """List all available model names."""
        if not os.path.exists(C.MODELS_DIR):
            return []
        return sorted([
            d for d in os.listdir(C.MODELS_DIR)
            if os.path.isdir(os.path.join(C.MODELS_DIR, d))
        ])

    def store_experience(self, frame, actual_action, next_frame, state_emb, next_state_emb, 
                         joint_positions=None, commanded_action=None):
        """
        Store experience in replay buffer.
        Args:
            frame: Raw image frame (numpy array)
            actual_action: List or array [d1, d2] - action actually applied (after clipping)
            next_frame: Raw next image frame (numpy array)
            state_emb: Tensor (ENCODED_DIM) - Cached embedding
            next_state_emb: Tensor (ENCODED_DIM) - Cached embedding
            joint_positions: List [s1, s2] - joint angles at time of action (optional)
            commanded_action: List [d1, d2] - action commanded by policy (before clipping)
        """
        timestamp = time.time()
        action_tensor = torch.tensor(actual_action, dtype=torch.float32, device=self.device)
        
        # Store joint positions as tensor (default to zeros if not provided for backward compat)
        if joint_positions is not None:
            joint_tensor = torch.tensor(joint_positions, dtype=torch.float32, device=self.device)
        else:
            joint_tensor = torch.zeros(C.ACTION_DIM, dtype=torch.float32, device=self.device)
        
        # Store commanded action (for clipping penalty); default to actual if not provided
        if commanded_action is not None:
            commanded_tensor = torch.tensor(commanded_action, dtype=torch.float32, device=self.device)
        else:
            commanded_tensor = action_tensor.clone()
        
        # Store as dictionary
        experience = {
            'frame': frame,
            'action': action_tensor,
            'commanded_action': commanded_tensor,
            'next_frame': next_frame,
            'timestamp': timestamp,
            'state_emb': state_emb.detach(),
            'next_state_emb': next_state_emb.detach(),
            'joint_positions': joint_tensor,
        }
        self.replay_buffer.append(experience)

    def train_fast_learner(self, state, action, next_state):
        """
        Train the fast learner on a single recent experience (Online Learning).
        """
        self.model.use_fast_learner()
        self.model.fast_learner.train()
        
        # Prepare inputs (add batch dim)
        state_b = state.unsqueeze(0)
        action_b = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        target_b = next_state.unsqueeze(0)
        
        # Concatenate inputs for ContinuousDualNetworkModel
        inp = torch.cat([state_b, action_b], dim=-1)
        
        # Forward
        prediction = self.model.fast_learner(inp)
        
        # Loss
        loss = self.loss_fn(prediction, target_b)
        
        # Backward
        self.fast_opt.zero_grad()
        loss.backward()
        self.fast_opt.step()
        
        return loss.item()

    def _train_slow_wake(self, state_emb, action, next_state_emb):
        """
        Slow wake update: distillation + conservative raw transition learning.
        
        L_wake_slow = w_distill * L_distill + w_raw * L_raw
        
        L_distill = ||F_C(z_t, a_t) - stopgrad(F_H(z_t, a_t))||^2
            Primary: prevents Slow from diverging from Fast.
            
        L_raw = ||F_C(z_t, a_t) - stopgrad(z_{t+1})||^2
            Conservative: keeps Slow grounded in reality.
            
        Regularization: small LR (SLOW_WAKE_LR) acts as implicit regularization.
        """
        self.model.slow_learner.train()
        
        state_b = state_emb.unsqueeze(0)
        action_b = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        target_b = next_state_emb.unsqueeze(0)
        inp = torch.cat([state_b, action_b], dim=-1)
        
        pred_slow = self.model.slow_learner(inp)
        
        loss = torch.tensor(0.0, device=self.device)
        distill_val = 0.0
        raw_val = 0.0
        
        # L_distill: match Fast predictions (primary)
        if C.SLOW_WAKE_W_DISTILL > 0:
            with torch.no_grad():
                pred_fast = self.model.fast_learner(inp)
            L_distill = self.loss_fn(pred_slow, pred_fast)
            loss = loss + C.SLOW_WAKE_W_DISTILL * L_distill
            distill_val = L_distill.item()
        
        # L_raw: conservative direct transition learning
        if C.SLOW_WAKE_W_RAW > 0:
            L_raw = self.loss_fn(pred_slow, target_b.detach())
            loss = loss + C.SLOW_WAKE_W_RAW * L_raw
            raw_val = L_raw.item()
        
        self.slow_wake_opt.zero_grad()
        loss.backward()
        self.slow_wake_opt.step()
        
        return {"total": loss.item(), "distill": distill_val, "raw": raw_val}

    def train_step(self, current_frame, actual_action, next_frame, joint_positions=None, commanded_action=None):
        """
        WAKE phase step: Encode -> Store -> Train Fast -> Train Slow.
        
        Fast: trains on raw next-state targets (high LR).
        Slow: L_distill (match Fast) + L_raw (conservative raw targets), low LR.
        
        Args:
            current_frame: Raw image frame (numpy array)
            actual_action: List [d1, d2] - actual action applied (after clipping)
            next_frame: Raw next image frame (numpy array)
            joint_positions: List [s1, s2] - joint angles at time of action (for policy training)
            commanded_action: List [d1, d2] - action commanded by policy (before clipping)
        
        Returns:
            dict: Training stats including whether sleep should trigger.
        """
        # Encode
        with torch.no_grad():
            state_emb = self.encode_frame(current_frame)
            next_state_emb = self.encode_frame(next_frame)
            
        # Store
        self.store_experience(current_frame, actual_action, next_frame, state_emb, next_state_emb, 
                              joint_positions, commanded_action)
        
        # Compute curiosity reward (before training, reflects current model surprise)
        curiosity_reward = self.compute_curiosity_reward(state_emb, actual_action, next_state_emb)
        
        # Train Fast Learner (one-step, high LR)
        fast_loss = self.train_fast_learner(state_emb, actual_action, next_state_emb)
        
        # Train Slow Learner (distillation only, low LR)
        slow_wake_loss = None
        if self.step_count % C.SLOW_WAKE_UPDATE_INTERVAL == 0:
            slow_wake_loss = self._train_slow_wake(state_emb, actual_action, next_state_emb)
        
        self.step_count += 1
        self.wake_steps_in_cycle += 1
        
        # Train policy periodically
        policy_stats = None
        if (self.step_count >= C.CURIOSITY_WARMUP_STEPS and 
            self.step_count % C.POLICY_TRAIN_INTERVAL == 0):
            policy_stats = self.train_policy()
        
        # Check if sleep should trigger
        should_sleep = self.wake_steps_in_cycle >= C.WAKE_STEPS_PER_CYCLE
        
        return {
            "step": self.step_count,
            "fast_loss": fast_loss,
            "slow_wake_loss": slow_wake_loss,
            "curiosity_reward": curiosity_reward,
            "policy_stats": policy_stats,
            "buffer_size": len(self.replay_buffer),
            "should_sleep": should_sleep,
            "wake_steps": self.wake_steps_in_cycle,
        }

    def _sample_sequences(self, seq_len, batch_size):
        """
        Sample contiguous trajectory windows of length seq_len from the replay buffer.
        Returns list of sequences, each a list of experience dicts.
        """
        buf = list(self.replay_buffer)
        if len(buf) < seq_len:
            return []
        
        sequences = []
        max_start = len(buf) - seq_len
        for _ in range(batch_size):
            start = random.randint(0, max_start)
            seq = buf[start:start + seq_len]
            sequences.append(seq)
        return sequences

    def _freeze_fast(self):
        """Freeze Fast learner: eval mode, no gradients."""
        self.model.fast_learner.eval()
        for p in self.model.fast_learner.parameters():
            p.requires_grad_(False)

    def _unfreeze_fast(self):
        """Unfreeze Fast learner: re-enable gradients."""
        for p in self.model.fast_learner.parameters():
            p.requires_grad_(True)

    def sws(self, sws_steps=None):
        """
        SWS Phase (Slow-Wave Sleep): Fast teaches Slow via replay distillation.
        
        Fast frozen. Slow trains strongly (SLOW_SLEEP_LR).
        
        L_SWS = α * L_dyn + β * L_z
        
        L_dyn: ||F_C(z_t, a_t) - stopgrad(F_H(z_t, a_t))||^2
        L_z:   ||E_C(o_t) - stopgrad(E_H(o_t))||^2  (skip if single encoder)
        """
        if sws_steps is None:
            sws_steps = C.SWS_STEPS
        
        if len(self.replay_buffer) < C.BATCH_SIZE:
            return {"status": "skipped", "reason": "not enough data"}
        
        self._freeze_fast()
        self.model.slow_learner.train()
        
        log = {"L_dyn": [], "L_sws": []}
        
        for step in range(sws_steps):
            batch = random.sample(self.replay_buffer, C.BATCH_SIZE)
            
            states_b = torch.stack([x['state_emb'] for x in batch])
            actions_b = torch.stack([x['action'] for x in batch])
            inp = torch.cat([states_b, actions_b], dim=-1)
            
            L_sws = torch.tensor(0.0, device=self.device)
            
            # L_dyn: dynamics distillation
            if C.SWS_ALPHA > 0:
                pred_slow = self.model.slow_learner(inp)
                with torch.no_grad():
                    pred_fast = self.model.fast_learner(inp)
                L_dyn = self.loss_fn(pred_slow, pred_fast)
                L_sws = L_sws + C.SWS_ALPHA * L_dyn
                log["L_dyn"].append(L_dyn.item())
            
            # L_z: representation distillation (skip: single encoder, β=0)
            
            self.slow_sleep_opt.zero_grad()
            L_sws.backward()
            self.slow_sleep_opt.step()
            
            log["L_sws"].append(L_sws.item())
        
        n = len(log["L_sws"])
        return {
            "status": "completed",
            "steps": n,
            "avg_L_dyn": sum(log["L_dyn"]) / n if log["L_dyn"] else 0,
            "avg_L_sws": sum(log["L_sws"]) / n if n > 0 else 0,
            "log": log,
        }

    def rem(self, rem_steps=None):
        """
        REM Phase (Dreaming): Stabilize Slow via multi-step rollout consistency.
        
        Fast frozen. Slow trains (SLOW_SLEEP_LR).
        
        L_REM = γ * L_ms
        
        L_ms = (1/K) Σ_{i=1..K} ||z_hat_i - stopgrad(z_target_i)||^2
        
        z_hat is rolled forward autoregressively through F_C.
        z_target_i = stopgrad(cached embedding of o_{t+i}).
        """
        if rem_steps is None:
            rem_steps = C.REM_STEPS
        
        K = C.REM_K
        min_required = max(C.BATCH_SIZE, K + 1)
        
        if len(self.replay_buffer) < min_required:
            return {"status": "skipped", "reason": "not enough data"}
        
        self._freeze_fast()
        self.model.slow_learner.train()
        
        log = {"L_ms": [], "L_rem": []}
        
        for step in range(rem_steps):
            sequences = self._sample_sequences(K + 1, C.BATCH_SIZE)
            if not sequences:
                continue
            
            # Extract states and actions from trajectory window
            states = []
            actions_list = []
            for t in range(K + 1):
                states.append(torch.stack([seq[t]['state_emb'] for seq in sequences]))
            for t in range(K):
                actions_list.append(torch.stack([seq[t]['action'] for seq in sequences]))
            
            # Targets: stopgrad so encoder doesn't cheat by drifting targets
            z_targets = [s.detach() for s in states[1:]]
            
            # Autoregressive rollout through Slow dynamics
            z_hat = states[0]
            L_ms = torch.tensor(0.0, device=self.device)
            
            for t in range(K):
                inp = torch.cat([z_hat, actions_list[t]], dim=-1)
                z_hat = self.model.slow_learner(inp)
                L_ms = L_ms + self.loss_fn(z_hat, z_targets[t])
            
            L_ms = L_ms / K
            L_rem = C.REM_GAMMA * L_ms
            
            self.slow_sleep_opt.zero_grad()
            L_rem.backward()
            self.slow_sleep_opt.step()
            
            log["L_ms"].append(L_ms.item())
            log["L_rem"].append(L_rem.item())
        
        n = len(log["L_rem"])
        return {
            "status": "completed",
            "steps": n,
            "avg_L_ms": sum(log["L_ms"]) / n if n > 0 else 0,
            "avg_L_rem": sum(log["L_rem"]) / n if n > 0 else 0,
            "log": log,
        }

    def _tether_fast(self):
        """
        Post-sleep: Tether Fast toward Slow via output distillation.
        
        Prevents Fast from drifting too far from consolidated knowledge.
        Uses output-level distillation (architectures differ in size).
        """
        tether_steps = C.TETHER_STEPS
        if tether_steps <= 0 or len(self.replay_buffer) < C.BATCH_SIZE:
            return {"tether_steps": 0, "avg_tether_loss": 0}
        
        self._unfreeze_fast()
        self.model.fast_learner.train()
        self.model.slow_learner.eval()
        
        tether_opt = torch.optim.Adam(
            self.model.fast_learner.parameters(), lr=C.TETHER_LR
        )
        
        tether_losses = []
        for _ in range(tether_steps):
            batch = random.sample(self.replay_buffer, C.BATCH_SIZE)
            states_b = torch.stack([x['state_emb'] for x in batch])
            actions_b = torch.stack([x['action'] for x in batch])
            inp = torch.cat([states_b, actions_b], dim=-1)
            
            with torch.no_grad():
                slow_pred = self.model.slow_learner(inp)
            
            fast_pred = self.model.fast_learner(inp)
            t_loss = self.loss_fn(fast_pred, slow_pred)
            
            tether_opt.zero_grad()
            t_loss.backward()
            tether_opt.step()
            tether_losses.append(t_loss.item())
        
        return {
            "tether_steps": tether_steps,
            "avg_tether_loss": sum(tether_losses) / len(tether_losses),
        }

    def sleep_cycle(self):
        """
        Full sleep cycle: SWS → REM → Tether.
        
        WAKE → SWS → REM → WAKE → ...
        
        SWS: Fast teaches Slow (distillation, consolidation).
        REM: Slow stabilizes itself (multi-step rollout consistency).
        Tether: Pull Fast slightly toward consolidated Slow.
        """
        print(f"\n=== Sleep Cycle {self.cycle_count + 1} ===")
        
        # Phase 1: SWS
        print(f"  SWS: {C.SWS_STEPS} steps...")
        sws_stats = self.sws()
        if sws_stats["status"] == "completed":
            print(f"    avg L_dyn={sws_stats['avg_L_dyn']:.6f}")
        
        # Phase 2: REM
        print(f"  REM: {C.REM_STEPS} steps, K={C.REM_K}...")
        rem_stats = self.rem()
        if rem_stats["status"] == "completed":
            print(f"    avg L_ms={rem_stats['avg_L_ms']:.6f}")
        
        # Phase 3: Tether
        tether_stats = self._tether_fast()
        if tether_stats["tether_steps"] > 0:
            print(f"  Tether: {tether_stats['tether_steps']} steps, "
                  f"avg_loss={tether_stats['avg_tether_loss']:.6f}")
        
        # Unfreeze Fast for next wake phase
        self._unfreeze_fast()
        
        self.model.consolidation_count += 1
        self.cycle_count += 1
        self.wake_steps_in_cycle = 0
        
        # Log sleep cycle
        self.log_event("sleep",
            cycle=self.cycle_count,
            sws_L_dyn=sws_stats.get("avg_L_dyn", ""),
            rem_L_ms=rem_stats.get("avg_L_ms", ""),
            tether_loss=tether_stats.get("avg_tether_loss", ""),
        )
        
        # Auto-save after sleep
        if C.CHECKPOINT_SAVE_INTERVAL > 0 and self.cycle_count % C.CHECKPOINT_SAVE_INTERVAL == 0:
            self.save()
        
        return {
            "cycle": self.cycle_count,
            "sws": sws_stats,
            "rem": rem_stats,
            "tether": tether_stats,
        }

    def predict_next(self, frame, action):
        """
        Predict next state embedding using the currently active network.
        Useful for planning or visualization.
        """
        with torch.no_grad():
            state_emb = self.encode_frame(frame)
            state_b = state_emb.unsqueeze(0)
            action_b = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Concatenate inputs
            inp = torch.cat([state_b, action_b], dim=-1)
            
            pred_emb = self.model(inp)
            return pred_emb.squeeze(0)

    # ---- Curiosity-Driven Exploration ----

    def get_curiosity_action(self, frame, joint_positions, explore=True):
        """
        Use the curiosity policy to generate an action for the given frame.
        Falls back to random exploration during warmup.
        Args:
            frame: Raw image frame (numpy array)
            joint_positions: List [s1, s2] - current joint angles in degrees
            explore: Whether to add exploration noise
        Returns:
            list: [d1, d2] action values
        """
        if self.step_count < C.CURIOSITY_WARMUP_STEPS:
            # Random babbling during warmup
            d1 = random.uniform(-C.ACTION_SCALE, C.ACTION_SCALE)
            d2 = random.uniform(-C.ACTION_SCALE, C.ACTION_SCALE)
            return [d1, d2]
        
        with torch.no_grad():
            state_emb = self.encode_frame(frame)
        return self.policy.get_action(state_emb, joint_positions, explore=explore)

    def compute_curiosity_reward(self, state_emb, action, next_state_emb):
        """
        Compute curiosity reward = world model prediction error.
        Higher error means the world model is surprised, so the action was "interesting".
        Args:
            state_emb: Tensor (ENCODED_DIM,)
            action: list or Tensor [d1, d2]
            next_state_emb: Tensor (ENCODED_DIM,)
        Returns:
            float: Curiosity reward (prediction error)
        """
        with torch.no_grad():
            state_b = state_emb.unsqueeze(0)
            if isinstance(action, list):
                action_b = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                action_b = action.unsqueeze(0)
            target_b = next_state_emb.unsqueeze(0)
            
            inp = torch.cat([state_b, action_b], dim=-1)
            pred = self.model.fast_learner(inp)
            
            # MSE as reward signal
            reward = F.mse_loss(pred, target_b).item()
        return reward

    def train_policy(self, batch_size=None):
        """
        Train the curiosity policy to output actions that maximize prediction error.
        
        The key insight: we backprop through the world model (frozen) to update the policy.
        Policy outputs action -> world model predicts next state -> we MAXIMIZE the error
        between prediction and actual next state.
        
        Since we want to maximize error, we negate the loss (or equivalently, minimize -error).
        
        The policy also receives joint positions so it can learn to avoid commanding
        movements beyond joint limits.
        
        Additionally, we penalize actions that would be clipped at joint limits by computing
        the expected clipping amount based on the current joint positions.
        """
        if batch_size is None:
            batch_size = C.POLICY_BATCH_SIZE
            
        if len(self.replay_buffer) < batch_size:
            return None
        
        self.policy.train()
        # Freeze world model — we only update the policy
        self.model.fast_learner.eval()
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        
        states_b = torch.stack([x['state_emb'] for x in batch])
        next_states_b = torch.stack([x['next_state_emb'] for x in batch])
        joints_b = torch.stack([x['joint_positions'] for x in batch])
        
        # Policy generates actions from states and joint positions
        policy_actions = self.policy(states_b, joints_b)
        
        # World model predicts next state from (state, policy_action)
        inp = torch.cat([states_b, policy_actions], dim=-1)
        predicted_next = self.model.fast_learner(inp)
        
        # Curiosity reward = prediction error
        # We want to MAXIMIZE this, so we minimize the negative
        prediction_error = F.mse_loss(predicted_next, next_states_b.detach())
        
        # Clipping penalty: penalize when the policy outputs actions that would be clipped
        # We use the actual clipping that occurred (commanded - actual) as a supervision signal
        # to teach the policy to output actions closer to what actually gets applied
        clipping_penalty = torch.tensor(0.0, device=self.device)
        if C.POLICY_CLIPPING_PENALTY > 0:
            actual_b = torch.stack([x['action'] for x in batch])
            # Penalize policy for outputting actions different from what actually got applied
            # This creates gradient flow: policy learns to output actions that won't be clipped
            clipping_penalty = F.mse_loss(policy_actions, actual_b.detach())
        
        # Total loss: maximize curiosity, minimize clipping
        # policy_loss = -curiosity + penalty * clipping
        policy_loss = -prediction_error + C.POLICY_CLIPPING_PENALTY * clipping_penalty
        
        # Backward (only updates policy, world model is not in the optimizer)
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "curiosity_reward": prediction_error.item(),
            "clipping_penalty": clipping_penalty.item() if C.POLICY_CLIPPING_PENALTY > 0 else 0.0
        }
