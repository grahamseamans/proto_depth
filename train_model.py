"""
Training script for the Proto-Depth model.
Performance-optimized version with vectorized operations and improved Chamfer distance.

This file serves as the central configuration point for all training parameters.

PERFORMANCE OPTIMIZATIONS:
--------------------------
1. Vectorized mesh transformation - eliminating loops in favor of batch operations
   * All slots processed simultaneously using tensor reshaping and broadcasting
   * Rotation matrices computed in batches rather than per-prototype
   * Expected speedup: 5-10x for the mesh transform step

2. Efficient Chamfer distance using KNN
   * Using K-nearest neighbors (K=3) approach instead of all-pairs comparison
   * Reduces complexity from O(n²) to O(n log n)
   * Vectorized slot-to-target distance calculations
   * Expected speedup: 2-5x for the Chamfer loss computation

3. Improved parallelization
   * Increased worker threads for data loading
   * Better CPU utilization during training

To run the optimized training:
   python train_model.py
"""

from src.train import train

if __name__ == "__main__":
    # =====================================================================
    # DATASET AND TRAINING CONFIGURATION
    # =====================================================================

    # Data and batch configuration
    data_path = "data/SYNTHIA-SF/SEQ1/DepthLeft"  # Path to depth images
    batch_size = 2  # Small batch size due to memory constraints with 32x32
    num_epochs = 1000  # Total training epochs
    lr = 1e-4  # Learning rate for Adam optimizer

    # Model complexity parameters
    num_prototypes = 32  # Number of prototype shapes (32 recommended)
    num_slots = 32  # Number of object slots (32 recommended)

    # =====================================================================
    # MEMORY OPTIMIZATION PARAMETERS
    # =====================================================================

    # Point sampling parameters
    samples_per_slot = 200  # Points sampled per slot (reduced from 5000 to save memory)

    # Slot positioning
    min_distance = 0.5  # Minimum distance between slot centroids (in meters)

    # =====================================================================
    # LOSS WEIGHTS AND COMPONENTS
    # =====================================================================

    # Chamfer loss components
    global_weight = 0.7  # Weight for global chamfer component
    slot_weight = 0.2  # Weight for per-slot proximity component
    repulsion_weight = 0.1  # Weight for centroid repulsion component
    k_nearest = 3  # Number of nearest neighbors for KNN-based Chamfer distance

    # Regularization weights
    w_chamfer = 1.0  # Overall weight for chamfer distance loss
    w_edge = 0.3  # Weight for edge length regularization
    w_normal = 0.3  # Weight for normal consistency
    w_laplacian = 0.3  # Weight for laplacian smoothing

    # =====================================================================
    # VISUALIZATION AND MONITORING
    # =====================================================================

    # Visualization frequency
    viz_interval = 10  # Visualize every N batches
    num_final_samples = 8  # Number of samples to visualize at the end
    use_interactive_viz = True  # Export data for interactive visualization

    # =====================================================================
    # PERFORMANCE OPTIMIZATION
    # =====================================================================

    # Parallelization
    num_workers = 8  # Number of worker threads for data loading

    # =====================================================================
    # ASSEMBLE CONFIG DICTIONARY
    # =====================================================================

    # Assemble all parameters into a single configuration dictionary
    config = {
        # Dataset and training
        "data_path": data_path,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "num_prototypes": num_prototypes,
        "num_slots": num_slots,
        # Memory optimization
        "samples_per_slot": samples_per_slot,
        "min_distance": min_distance,
        # Loss weights
        "global_weight": global_weight,
        "slot_weight": slot_weight,
        "repulsion_weight": repulsion_weight,
        "k_nearest": k_nearest,
        # Regularization
        "w_chamfer": w_chamfer,
        "w_edge": w_edge,
        "w_normal": w_normal,
        "w_laplacian": w_laplacian,
        # Visualization
        "viz_interval": viz_interval,
        "num_final_samples": num_final_samples,
        "use_interactive_viz": use_interactive_viz,
        # Performance
        "num_workers": num_workers,
    }

    # Start training with the configured parameters
    train(**config)
