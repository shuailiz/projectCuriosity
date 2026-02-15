import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config as C


class CuriosityPolicy(nn.Module):
    """
    Policy network for curiosity-driven exploration.
    
    Takes the current state embedding and joint positions, outputs an action (servo deltas)
    that is trained to maximize the world model's prediction error.
    
    Joint positions are included so the policy can learn to avoid commanding
    movements beyond joint limits.
    
    Architecture: MLP with tanh output scaled to ACTION_SCALE.
    """
    def __init__(self):
        super().__init__()
        
        # Input: visual embedding + normalized joint positions
        self.input_dim = C.ENCODED_DIM + C.ACTION_DIM  # ACTION_DIM = number of joints
        self.hidden_dim = C.POLICY_HIDDEN_DIM
        self.output_dim = C.ACTION_DIM
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.action_scale = C.ACTION_SCALE
        self.noise_std = C.POLICY_NOISE_STD
        
        # Store servo limits for normalization
        self.servo_limits = C.SERVO_LIMITS
        
        self.to(C.DEVICE)

    def normalize_joints(self, joint_positions):
        """
        Normalize joint positions to [-1, 1] based on servo limits.
        Args:
            joint_positions: Tensor (B, ACTION_DIM) or (ACTION_DIM,) - raw joint angles in degrees
        Returns:
            Tensor: normalized joint positions in [-1, 1]
        """
        normalized = torch.zeros_like(joint_positions)
        for i, (j_min, j_max) in enumerate(self.servo_limits):
            # Map [j_min, j_max] to [-1, 1]
            normalized[..., i] = 2.0 * (joint_positions[..., i] - j_min) / (j_max - j_min) - 1.0
        return normalized

    def forward(self, state_emb, joint_positions):
        """
        Args:
            state_emb: Tensor (B, ENCODED_DIM)
            joint_positions: Tensor (B, ACTION_DIM) - current joint angles in degrees
        Returns:
            Tensor (B, ACTION_DIM) - Action in [-ACTION_SCALE, ACTION_SCALE]
        """
        # Normalize joint positions to [-1, 1]
        norm_joints = self.normalize_joints(joint_positions)
        
        # Concatenate visual embedding with normalized joint positions
        x = torch.cat([state_emb, norm_joints], dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        # tanh squashes to [-1, 1], then scale to [-ACTION_SCALE, ACTION_SCALE]
        action = torch.tanh(x) * self.action_scale
        return action

    def get_action(self, state_emb, joint_positions, explore=True):
        """
        Get an action for a single state, optionally with exploration noise.
        Args:
            state_emb: Tensor (ENCODED_DIM,)
            joint_positions: list or Tensor [s1, s2] - current joint angles in degrees
            explore: If True, add Gaussian noise for exploration
        Returns:
            list: [d1, d2] action values
        """
        with torch.no_grad():
            state_b = state_emb.unsqueeze(0)
            if isinstance(joint_positions, list):
                joint_b = torch.tensor(joint_positions, dtype=torch.float32, device=state_emb.device).unsqueeze(0)
            else:
                joint_b = joint_positions.unsqueeze(0)
            
            action = self.forward(state_b, joint_b).squeeze(0)
            
            if explore:
                noise = torch.randn_like(action) * self.noise_std * self.action_scale
                action = action + noise
                # Clamp to valid range
                action = action.clamp(-self.action_scale, self.action_scale)
            
            return action.cpu().tolist()
