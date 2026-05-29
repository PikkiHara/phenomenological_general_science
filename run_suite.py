import sys
import os
import torch

# Ensure the local package directory is accessible to the path
sys.path.append(os.getcwd())

from pcd_green_ai.tracer import EmpiricalManifoldTracer
from pcd_green_ai.biophysics import UnifiedDockingPipeline

def run_pcd_general_science_suite():
    print("=" * 80)
    print("         PHENOMENOLOGICAL COMPUTATIONAL DYNAMICS (PCD) SYSTEM INITIATION")
    print("=" * 80)
    
    # SYSTEM 1: HARDWARE TELEMETRY MANIFOLD ENGINE (3D Space)
    hardware_tracer = EmpiricalManifoldTracer()
    hardware_tracer.profile_active_environment()
    
    print("\n" + " - " * 27 + "\n")
    
    # SYSTEM 2: MOLECULAR BIOPHYSICS DOCKING PIPELINE (PDB Core Driven)
    biophysics_pipeline = UnifiedDockingPipeline()
    
    print("=" * 80)
    print("     EXECUTING RECEPTOR-LIGAND ATOMIC CONFIGURATION CALCULATION")
    print("=" * 80)
    
    # Load authentic structural coordinates from our PDB asset file
    pdb_path = "target_protein.pdb"
    print(f" -> Parsing real structural dataset target from: {pdb_path}")
    
    try:
        mock_receptor_atoms = biophysics_pipeline.parse_pdb_coordinates(pdb_path)
        print(f" -> Successfully loaded {mock_receptor_atoms.shape[0]} base atomic coordinates from disk.")
    except Exception as e:
        print(f" ❌ Critical Core Loader Failure: {e}")
        return
    
    # Simulating continuous induced-fit descent trajectory relative to parsed matrix
    spatial_milestones = [15.0, 7.5, 3.8, 1.1]
    
    for step, offset in enumerate(spatial_milestones):
        # Translate the mock ligand matrix along the X-axis closer to the parsed protein center
        mock_ligand_atoms = torch.randn(25, 3) + torch.tensor([offset, 0.0, 0.0])
        
        # Resolve atomic interaction vectors based on parsed receptor layout
        field_variance, operational_mode = biophysics_pipeline.calculate_interaction_field(
            mock_ligand_atoms, mock_receptor_atoms
        )
        
        print(f"\n[Trajectory Frame {step + 1:02d}] Spatial Radial Distance: {biophysics_pipeline.spatial_distance:6.3f} Å")
        print(f" -> Field Variance : {field_variance:8.3f} | Operational Mode: {operational_mode}")
        print(f" -> Affinity Metric: ΔG = {biophysics_pipeline.binding_affinity_delta_g:8.3f} kJ/mol")
        print(f" -> System Convergence Lock Stability Coefficient : {biophysics_pipeline.pocket_lock_stability * 100:5.1f}%")

    print("=" * 80)
    print("         ALL COMPUTATIONAL DYNAMICS CORE SUBSYSTEMS COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_pcd_general_science_suite()
