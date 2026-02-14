import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config as C


class CuriosityPolicy(nn.Module):
    """
    Policy network for curiosity-driven exploration.
    
    Takes the current state embedding and outputs an action (servo deltas)
    that is trained to maximize the world model's prediction error.
    
    Architecture: MLP with tanh output scaled to ACTION_SCALE.
    """
    def __init__(self):
        super().__init__()
        
        self.input_dim = C.ENCODED_DIM
        self.hidden_dim = C.POLICY_HIDDEN_DIM
        self.output_dim = C.ACTION_DIM
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.action_scale = C.ACTION_SCALE
        self.noise_std = C.POLICY_NOISE_STD
        
        self.to(C.DEVICE)

    def forward(self, state_emb):
        """
        Args:
            state_emb: Tensor (B, ENCODED_DIM)
        Returns:
            Tensor (B, ACTION_DIM) - Action in [-ACTION_SCALE, ACTION_SCALE]
        """
        x = F.relu(self.fc1(state_emb))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        # tanh squashes to [-1, 1], then scale to [-ACTION_SCALE, ACTION_SCALE]
        action = torch.tanh(x) * self.action_scale
        return action

    def get_action(self, state_emb, explore=True):
        """
        Get an action for a single state, optionally with exploration noise.
        Args:
            state_emb: Tensor (ENCODED_DIM,)
            explore: If True, add Gaussian noise for exploration
        Returns:
            list: [d1, d2] action values
        """
        with torch.no_grad():
            state_b = state_emb.unsqueeze(0)
            action = self.forward(state_b).squeeze(0)
            
            if explore:
                noise = torch.randn_like(action) * self.noise_std * self.action_scale
                action = action + noise
                # Clamp to valid range
                action = action.clamp(-self.action_scale, self.action_scale)
            
            return action.cpu().tolist()
