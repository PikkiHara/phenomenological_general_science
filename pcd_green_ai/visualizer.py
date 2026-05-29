import matplotlib.pyplot as plt
import os

class PCDDataVisualizer:
    """
    Generates analytics graphics mapping 3D manifold curvatures and 
    biophysics affinity trajectories for structural profiling reports.
    """
    @staticmethod
    def generate_report_plots(curvature_records, distance_records, energy_records):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: 3D Manifold Space Curvature Growth
        ax1.plot(range(1, len(curvature_records) + 1), curvature_records, color='#2ca02c', marker='o', linewidth=2)
        ax1.set_title("PCD 3D Manifold Space Curvature Growth ($R$)")
        ax1.set_xlabel("Test Coordinate Points")
        ax1.set_ylabel("Scalar Curvature Coefficient ($R$)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Plot 2: Biophysics Affinity vs Coordinate Frame
        color = '#1f77b4'
        ax2.set_xlabel('Trajectory Frames')
        ax2.set_ylabel('Spatial Distance (Å)', color=color)
        ax2.plot(range(1, len(distance_records) + 1), distance_records, color=color, marker='s', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax3 = ax2.twinx()  
        color = '#d62728'
        ax3.set_ylabel('Free Energy $\Delta$G (kJ/mol)', color=color)
        ax3.plot(range(1, len(energy_records) + 1), energy_records, color=color, marker='^', linestyle='--', linewidth=2)
        ax3.tick_params(axis='y', labelcolor=color)
        
        ax2.set_title("Biophysics Thermodynamic Flight Profile")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        # Save output image
        output_image = "pcd_runtime_analytics.png"
        plt.savefig(output_image, dpi=150)
        plt.close()
        print(f"\n[Visualizer] Analytics chart cleanly rendered and saved to disk as: {output_image}")
