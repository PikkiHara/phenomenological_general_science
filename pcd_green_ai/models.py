import torch
import torch.nn as nn

class PCDAffinityPredictorNet(nn.Module):
    """
    Deep Learning Layer for Phenomological Computational Dynamics:
    Predicts the structural system convergence lock stability percentage based on
    3D hardware metric profiles and molecular field state vectors.
    """
    def __init__(self):
        super(PCDAffinityPredictorNet, self).__init__()
        # Input Vector: 3 Matrix diagonals + 1 Field Variance = 4 Features
        self.network_backbone = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid() # Outputs continuous bounded match probability [0, 1]
        )
        
    def forward(self, x):
        return self.network_backbone(x)
