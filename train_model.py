"""
Training script for the Proto-Depth model.
"""

from src.train import train

if __name__ == "__main__":
    # Training configuration
    config = {
        "data_path": "data/SYNTHIA-SF/SEQ1/DepthDebugLeft",
        "batch_size": 4,  # Smaller batch size for testing
        "num_epochs": 2,  # Just 2 epochs for testing
        "lr": 1e-4,
        "num_prototypes": 5,  # Fewer prototypes for testing
        "num_slots": 3,  # Fewer slots for testing
        "viz_interval": 5,  # Visualize more frequently
    }

    # Start training
    train(**config)
