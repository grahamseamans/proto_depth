"""
Training script for the Proto-Depth model.
Memory-optimized version for 64 prototypes and 64 slots.
"""

from src.train import train

if __name__ == "__main__":
    # Training configuration
    config = {
        "data_path": "data/SYNTHIA-SF/SEQ1/DepthLeft",
        "batch_size": 2,  # Small batch size due to memory constraints with 64x64
        "num_epochs": 1000,
        "lr": 1e-4,
        "num_prototypes": 64,  # Full 64 prototypes
        "num_slots": 64,  # Full 64 slots
        "viz_interval": 10,  # Visualize less frequently due to memory constraints
        "num_final_samples": 8,  # Generate 8 side-by-side visualizations at the end
        # Memory optimization parameters
        "samples_per_slot": 200,  # Reduced from 5000 to save memory
        "min_distance": 0.5,  # Minimum distance between slot centroids (in meters)
        # Loss weights
        "global_weight": 0.7,  # Weight for global chamfer component
        "slot_weight": 0.2,  # Weight for per-slot proximity component
        "repulsion_weight": 0.1,  # Weight for centroid repulsion component
        # Regularization weights
        "w_chamfer": 1.0,  # Weight for chamfer distance loss
        "w_edge": 0.3,  # Weight for edge length regularization
        "w_normal": 0.3,  # Weight for normal consistency
        "w_laplacian": 0.3,  # Weight for laplacian smoothing
        # Visualization options
        "use_interactive_viz": True,  # Export data for interactive visualization
    }

    # Start training
    train(**config)
