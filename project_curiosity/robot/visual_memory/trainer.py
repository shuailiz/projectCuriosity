import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Optional

from . import config as C
from .encoder import VisualEncoder
from ...dual_network_model import ContinuousDualNetworkModel

class VisualTrainer:
    """
    Trainer for the Visual-Motor Continuous Learning System.
    Manages the Fast/Slow learners, Replay Buffer, and Training Loops.
    """
    def __init__(self):
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
        
        # Optimizers
        self.fast_opt = torch.optim.Adam(
            self.model.fast_learner.parameters(), 
            lr=C.FAST_LEARNING_RATE
        )
        self.slow_opt = torch.optim.Adam(
            self.model.slow_learner.parameters(), 
            lr=C.SLOW_LEARNING_RATE
        )
        
        # Replay Buffer
        # Stores tuples of (state_emb, action, next_state_emb)
        # We store embeddings to save memory and avoid re-encoding images constantly
        self.replay_buffer = deque(maxlen=C.REPLAY_BUFFER_SIZE)
        
        self.step_count = 0
        self.loss_fn = nn.MSELoss()
        
        self.device = C.DEVICE
        
        print("VisualTrainer Initialized.")

    def encode_frame(self, frame):
        """Encode a raw image frame into an embedding."""
        return self.encoder.encode(frame)

    def store_experience(self, state, action, next_state):
        """
        Store experience in replay buffer.
        Args:
            state: Tensor (ENCODED_DIM)
            action: List or array [d1, d2]
            next_state: Tensor (ENCODED_DIM)
        """
        # Move to CPU to save GPU memory if needed, or keep on device for speed
        # For 5000 items of 512 floats, GPU mem is negligible (5000*512*4 bytes = 10MB)
        # So we can keep them on device.
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.replay_buffer.append((state.detach(), action_tensor, next_state.detach()))

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

    def train_step(self, current_frame, action, next_frame):
        """
        Full interaction step: Encode -> Store -> Train Fast.
        Returns:
            dict: Training stats
        """
        # Encode
        with torch.no_grad():
            state_emb = self.encode_frame(current_frame)
            next_state_emb = self.encode_frame(next_frame)
            
        # Store
        self.store_experience(state_emb, action, next_state_emb)
        
        # Train Fast Learner immediately
        loss = self.train_fast_learner(state_emb, action, next_state_emb)
        
        self.step_count += 1
        
        return {
            "step": self.step_count,
            "fast_loss": loss,
            "buffer_size": len(self.replay_buffer)
        }

    def consolidate(self, num_replays=None):
        """
        Sleep phase: Consolidate memories from Fast to Slow learner.
        """
        if num_replays is None:
            num_replays = C.CONSOLIDATION_REPLAYS
            
        if len(self.replay_buffer) < C.BATCH_SIZE:
            return {"status": "skipped", "reason": "not enough data"}
            
        print(f"Consolidating... ({num_replays} batches)")
        
        self.model.slow_learner.train()
        self.model.fast_learner.eval() # Teacher mode
        
        total_slow_loss = 0
        total_distill_loss = 0
        
        for _ in range(num_replays):
            # Sample batch
            batch = random.sample(self.replay_buffer, C.BATCH_SIZE)
            
            # Unzip batch
            states, actions, next_states = zip(*batch)
            
            # Stack
            states_b = torch.stack(states)
            actions_b = torch.stack(actions)
            next_states_b = torch.stack(next_states)
            
            # Concatenate inputs
            inp = torch.cat([states_b, actions_b], dim=-1)
            
            # 1. Slow learner prediction
            slow_pred = self.model.slow_learner(inp)
            
            # 2. Fast learner prediction (Teacher)
            with torch.no_grad():
                fast_pred = self.model.fast_learner(inp)
                
            # 3. Ground Truth Loss (MSE with actual next state)
            gt_loss = self.loss_fn(slow_pred, next_states_b)
            
            # 4. Distillation Loss (MSE with Fast Learner's prediction)
            # In classification we use KL Div, here in regression MSE is fine.
            # We want Slow learner to mimic Fast learner's intuition.
            distill_loss = self.loss_fn(slow_pred, fast_pred)
            
            # Combined Loss
            # We can weight them. Distillation helps generalize from the fast learner's recent adaptation.
            # GT helps ground it in reality.
            loss = gt_loss + C.DISTILLATION_TEMPERATURE * distill_loss
            
            # Backward
            self.slow_opt.zero_grad()
            loss.backward()
            self.slow_opt.step()
            
            total_slow_loss += gt_loss.item()
            total_distill_loss += distill_loss.item()
            
        return {
            "status": "completed",
            "avg_gt_loss": total_slow_loss / num_replays,
            "avg_distill_loss": total_distill_loss / num_replays
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
