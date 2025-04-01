"""
Training script for the Proto-Depth model.
"""

from src.train import train

if __name__ == "__main__":
    # Training configuration
    config = {
        "data_path": "data/SYNTHIA-SF/SEQ1/DepthLeft",
        "batch_size": 5,  # Smaller batch size for testing
        "num_epochs": 1000,  # Just 2 epochs for testing
        "lr": 1e-4,
        "num_prototypes": 4,  # Fewer prototypes for testing
        "num_slots": 4,  # Fewer slots for testing
        "viz_interval": 5,  # Visualize more frequently
        "num_final_samples": 8,  # Generate 8 side-by-side visualizations at the end
        # Regularization weights
        "w_chamfer": 1.0,  # Weight for chamfer distance loss
        "w_edge": 0.3,  # Weight for edge length regularization (balanced)
        "w_normal": 0.3,  # Weight for normal consistency (balanced)
        "w_laplacian": 0.3,  # Weight for laplacian smoothing (balanced)
        # Visualization options
        "use_interactive_viz": True,  # Export data for interactive visualization
    }

    # Start training
    train(**config)
