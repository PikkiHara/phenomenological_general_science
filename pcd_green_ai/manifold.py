import math
import torch

class MultiDimensionalManifoldEngine:
    """
    PCD General Science Engine: Computes a 3D Riemannian metric tensor 
    mapping the structural coupling between memory latency, instruction processing,
    and I/O bandwidth/disk stride fields.
    """
    def __init__(self):
        # Coordinates: [Sigma (Memory Log-Mass), Xi (Branch Entropy), Zeta (I/O Saturation)]
        self.metric_tensor = torch.zeros((3, 3))
        
    def compute_manifold_geometry(self, matrix_stride_latency, branch_miss_ratio, io_bus_utilization):
        """
        Calculates the localized curvature coefficients across three distinct hardware dimensions.
        """
        # Normalize inputs into continuous coordinate spaces
        sigma_coord = math.log(max(1.0, matrix_stride_latency))
        xi_coord = float(branch_miss_ratio)
        zeta_coord = float(io_bus_utilization) # New 3D Axis component
        
        # Diagonal elements: Pure structural variances along individual axes
        g_00 = max(0.01, sigma_coord ** 2)       # Memory field component
        g_11 = max(0.01, (xi_coord * 10.0) ** 2) # Instruction field component
        g_22 = max(0.01, (zeta_coord * 5.0) ** 2) # I/O field component
        
        # Off-diagonal elements: Cross-coupling variances
        g_01 = 0.5 * (sigma_coord * xi_coord)    # Memory <-> Instruction
        g_10 = g_01
        
        g_02 = 0.3 * (sigma_coord * zeta_coord)  # Memory <-> I/O
        g_20 = g_02
        
        g_12 = 0.2 * (xi_coord * zeta_coord)      # Instruction <-> I/O
        g_21 = g_12
        
        # Assemble the 3D Metric Tensor Matrix
        self.metric_tensor[0, 0] = g_00; self.metric_tensor[0, 1] = g_01; self.metric_tensor[0, 2] = g_02
        self.metric_tensor[1, 0] = g_10; self.metric_tensor[1, 1] = g_11; self.metric_tensor[1, 2] = g_12
        self.metric_tensor[2, 0] = g_20; self.metric_tensor[2, 1] = g_21; self.metric_tensor[2, 2] = g_22
        
        # Calculate the Determinant of the 3x3 Space Matrix using expansion by minors
        det_g = (
            g_00 * (g_11 * g_22 - g_12 * g_21) -
            g_01 * (g_10 * g_22 - g_12 * g_20) +
            g_02 * (g_10 * g_21 - g_11 * g_20)
        )
        
        # Invariant Curvature Scalar approximation in 3D Space
        trace_sum = g_00 + g_11 + g_22
        if det_g > 0:
            scalar_curvature_R = trace_sum / (det_g ** (1.0 / 3.0))
        else:
            scalar_curvature_R = 0.0
            
        return scalar_curvature_R, self.metric_tensor
