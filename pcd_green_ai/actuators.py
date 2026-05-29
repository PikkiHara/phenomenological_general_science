import math

class CarbonMitigationActuator:
    def __init__(self, base_power_watts=250.0):
        self.base_gpu_power_limit_watts = base_power_watts
        self.current_allocated_power = base_power_watts
        self.total_watt_seconds_saved = 0.0
        self.co2_grams_per_watt_second = 0.385 / 3600.0

    def compute_power_actuation(self, field_strength_f, delta_g):
        if delta_g > 500.0:
            reduction_factor = max(0.4, 1.0 - (field_strength_f / 1e12))
            self.current_allocated_power = self.base_gpu_power_limit_watts * reduction_factor
            action = "SHIELD_ACTIVE"
        else:
            self.current_allocated_power = min(self.base_gpu_power_limit_watts, self.current_allocated_power + 25.0)
            action = "CLOCK_RESTORE"
        watts_saved = self.base_gpu_power_limit_watts - self.current_allocated_power
        self.total_watt_seconds_saved += watts_saved * 0.05
        return action, watts_saved

    def generate_dashboard_report(self):
        total_co2_offset_mg = (self.total_watt_seconds_saved * self.co2_grams_per_watt_second) * 1000.0
        efficiency_recovery = (self.total_watt_seconds_saved / (self.base_gpu_power_limit_watts * 0.6)) * 100.0
        print("\n" + "="*90)
        print("                         PCD GREEN-AI CORE SYSTEM REPORT")
        print("="*90)
        print(f" TOTAL RAW GPU ENVELOPE ENERGY PRESERVED : {self.total_watt_seconds_saved:.4f} JOULES")
        print(f" ESTIMATED CARBON OFFSETS GENERATED      : {total_co2_offset_mg:.4f} MG CO2 AVOIDED")
        print(f" RUNTIME EFFICIENCY COEFFICIENT RECOVERY : {efficiency_recovery:.3f}%")
        print("="*90)
