import math
import torch

class ThermodynamicFieldEngine:
    @staticmethod
    def evaluate_gradient_flux(grad):
        grad_mean = torch.mean(torch.abs(grad)).item()
        grad_std = torch.std(grad).item()
        field_strength_f = (grad_mean * grad_std) * 4e4
        exponent_x = max(0.1, grad_std * 40.0)
        ln_omega = exponent_x * math.log(2.0)
        entropy_s = 1.3806 * ln_omega
        delta_g = field_strength_f - (298.15 * 0.005 * entropy_s)
        return field_strength_f, delta_g
