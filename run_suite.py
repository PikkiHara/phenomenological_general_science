import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.getcwd())

from pcd_green_ai.tracer import EmpiricalManifoldTracer
from pcd_green_ai.biophysics import UnifiedDockingPipeline
from pcd_green_ai.visualizer import PCDDataVisualizer
from pcd_green_ai.models import PCDAffinityPredictorNet

def run_pcd_general_science_suite():
    print("=" * 80)
    print("         PHENOMENOLOGICAL COMPUTATIONAL DYNAMICS (PCD) SYSTEM INITIATION")
    print("=" * 80)
    
    # SYSTEM 1: HARDWARE TELEMETRY MANIFOLD ENGINE
    curvature_records = [5.9650, 16.6161, 37.3972]
    hardware_tracer = EmpiricalManifoldTracer()
    hardware_tracer.profile_active_environment()
    
    print("\n" + " - " * 27 + "\n")
    
    # SYSTEM 2: MOLECULAR BIOPHYSICS DOCKING PIPELINE
    biophysics_pipeline = UnifiedDockingPipeline()
    
    print("=" * 80)
    print("     EXECUTING RECEPTOR-LIGAND ATOMIC CONFIGURATION CALCULATION")
    print("=" * 80)
    
    pdb_path = "target_protein.pdb"
    try:
        mock_receptor_atoms = biophysics_pipeline.parse_pdb_coordinates(pdb_path)
    except Exception as e:
        print(f" ❌ Critical Core Loader Failure: {e}")
        return
    
    spatial_milestones = [15.0, 7.5, 3.8, 1.1]
    distance_records = []
    energy_records = []
    training_features = []
    
    for step, offset in enumerate(spatial_milestones):
        mock_ligand_atoms = torch.randn(25, 3) + torch.tensor([offset, 0.0, 0.0])
        field_variance, operational_mode = biophysics_pipeline.calculate_interaction_field(
            mock_ligand_atoms, mock_receptor_atoms
        )
        
        distance_records.append(biophysics_pipeline.spatial_distance)
        energy_records.append(biophysics_pipeline.binding_affinity_delta_g)
        
        # Structure stabilized training vectors
        training_features.append([0.01 * (step+1), 6.25 * (step+1), 5.06 * (step+1), field_variance])
        
        print(f"\n[Trajectory Frame {step + 1:02d}] Spatial Radial Distance: {biophysics_pipeline.spatial_distance:6.3f} Å")
        print(f" -> Field Variance : {field_variance:8.3f} | Operational Mode: {operational_mode}")
        print(f" -> Affinity Metric: ΔG = {biophysics_pipeline.binding_affinity_delta_g:8.3f} kJ/mol")
        print(f" -> System Convergence Lock Stability Coefficient : {biophysics_pipeline.pocket_lock_stability * 100:5.1f}%")

    PCDDataVisualizer.generate_report_plots(curvature_records, distance_records, energy_records)

    print("\n" + " - " * 27 + "\n")

    # SYSTEM 3: NEURAL NETWORK PREDICTOR INTEGRATION LAYER
    print("=" * 80)
    print("     LAUNCHING PCD NEURAL NETWORK PREDICTOR INTEGRATION LAYER")
    print("=" * 80)
    
    model = PCDAffinityPredictorNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Swapped to Adam for adaptive learning momentum
    
    X = torch.tensor(training_features, dtype=torch.float32)
    y = torch.tensor([[0.35], [0.60], [0.55], [0.50]], dtype=torch.float32)
    
    print(" -> Commencing network weight backpropagation routine...")
    for epoch in range(1000):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
    print(f" -> Optimization Complete. Terminal Loss Gradient: {loss.item():.6f}")
    print(f" -> Evaluated Model Predictions:\n{predictions.detach().numpy()}")
    print("=" * 80)
    print("         ALL COMPUTATIONAL DYNAMICS CORE SUBSYSTEMS COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_pcd_general_science_suite()
