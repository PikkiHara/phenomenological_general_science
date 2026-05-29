import torch
import torch.nn as nn
from .fields import ThermodynamicFieldEngine

class PhenomenologicalWrapper(nn.Module):
    def __init__(self, target_layer, layer_id, actuator):
        super(PhenomenologicalWrapper, self).__init__()
        self.target_layer = target_layer
        self.layer_id = layer_id
        self.actuator = actuator
        if hasattr(self.target_layer, 'weight') and self.target_layer.weight is not None:
            self.target_layer.weight.register_hook(self.resolve_layer_gauge_field)

    def resolve_layer_gauge_field(self, grad):
        F_mu_nu, delta_G = ThermodynamicFieldEngine.evaluate_gradient_flux(grad)
        action, savings = self.actuator.compute_power_actuation(F_mu_nu, delta_G)
        if F_mu_nu > 10.0:
            print(f"[LAYER {self.layer_id:02d} ACTUATION] F_μν: {F_mu_nu:9.2f} | Power Cap: {self.actuator.current_allocated_power:6.2f}W | -> {action}")
        throttling_ratio = self.actuator.current_allocated_power / self.actuator.base_gpu_power_limit_watts
        return grad * throttling_ratio

    def forward(self, x):
        return self.target_layer(x)
