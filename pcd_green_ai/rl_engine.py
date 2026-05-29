import torch
import torch.nn as nn
import torch.nn.functional as F

class PCDActorCriticAgent(nn.Module):
    """
    Advanced Machine Learning Core: Implements an Actor-Critic Policy 
    Optimization structure for autonomous molecular trajectory tracking.
    """
    def __init__(self, state_dim=4, action_dim=3):
        super(PCDActorCriticAgent, self).__init__()
        # Common Feature Extraction Layer
        self.shared_layer = nn.Linear(state_dim, 32)
        
        # Actor Head: Outputs mean action vectors for spatial stride adjustments
        self.actor_head = nn.Linear(32, action_dim)
        
        # Critic Head: Evaluates state value V(s) based on system energy state
        self.critic_head = nn.Linear(32, 1)
        
    def forward(self, state):
        x = F.relu(self.shared_layer(state))
        
        # Continuous action spatial translation shifts
        action_mean = torch.tanh(self.actor_head(x))
        state_value = self.critic_head(x)
        
        return action_mean, state_value
