import torch
import torch.nn as nn
import math

class HyperbolicManifoldEmbeddingEngine(nn.Module):
    """
    Advanced Computer Science Core: Computes Non-Euclidean Hyperbolic 
    Graph Node Embeddings mapped onto a Poincaré ball, dynamically regularized
    by the macro hardware metric curvature scalar R.
    """
    def __init__(self, input_dim=3, embedding_dim=16):
        super(HyperbolicManifoldEmbeddingEngine, self).__init__()
        self.embedding_dim = embedding_dim
        self.linear_transform = nn.Linear(input_dim, embedding_dim)
        
    def mobius_addition(self, x, y, c=1.0):
        """
        Computes non-associative vector addition in the curved Poincaré 
        hyperbolic space framework.
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
        denom = 1.0 + 2.0 * c * xy + c**2 * x2 * y2
        return num / (denom + 1e-7)

    def forward(self, atomic_coordinates, hardware_curvature_r):
        """
        Embeds Euclidean PDB coordinates into a curved Hyperbolic space,
        using the hardware Riemann scalar R to dynamically alter the curvature parameter c.
        """
        # Linear projection to embedding feature space
        features = torch.tanh(self.linear_transform(atomic_coordinates))
        
        # Adapt Hyperbolic curvature 'c' based on systemic hardware friction R
        # Higher bottleneck constraints contract the available embedding volume
        c_adaptive = 1.0 / max(1.0, math.log(hardware_curvature_r + 1e-5))
        
        # Project Euclidean features directly onto the Poincaré disk boundary surface
        norm_x = torch.norm(features, p=2, dim=-1, keepdim=True)
        max_norm = 1.0 / math.sqrt(c_adaptive) - 1e-5
        hyperbolic_embeddings = features * (torch.clamp(norm_x, max=max_norm) / (norm_x + 1e-7))
        
        # Intrinsic geometric reduction
        manifold_centroid = torch.mean(hyperbolic_embeddings, dim=0, keepdim=True)
        
        return hyperbolic_embeddings, manifold_centroid, c_adaptive
