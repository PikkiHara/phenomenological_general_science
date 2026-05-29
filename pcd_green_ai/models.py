import torch
import torch.nn as nn

class PCDAffinityPredictorNet(nn.Module):
    """
    Upgraded PCD Neural Network Predictor: Integrates dynamic 
    Min-Max feature scaling to neutralize vanishing gradients.
    """
    def __init__(self):
        super(PCDAffinityPredictorNet, self).__init__()
        # Input Vector: 4 normalized features
        self.network_backbone = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Prevent division by zero if variance is constant
        eps = 1e-6
        min_val = x.min(dim=0, keepdim=True)[0]
        max_val = x.max(dim=0, keepdim=True)[0]
        
        # Scale inputs cleanly between 0.0 and 1.0
        x_scaled = (x - min_val) / (max_val - min_val + eps)
        
        return self.network_backbone(x_scaled)
