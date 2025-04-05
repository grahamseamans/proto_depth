"""
Run Energy-Based Scene Optimization with GPU Acceleration

This script uses the GPU-accelerated implementation of the energy-based scene optimizer,
which leverages Kaolin's fast nearest neighbor search operations to significantly
speed up the optimization process compared to the CPU-based KD-tree approach.
"""

import torch
import numpy as np
from energy_optimizer_native import EnergyBasedSceneOptimizer
from viz_exporter import VizExporter
from dataloader_minimal import process_depth_image
from pathlib import Path
import argparse
from glob import glob
import os
import sys
from PIL import Image
import random


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run Energy-Based Scene Optimization (Native Implementation)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/SYNTHIA-SF/SEQ1/DepthDebugLeft",
        help="Path to depth image data",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1000,
        help="Number of optimization iterations",
    )
    parser.add_argument(
        "--num_slots",
        type=int,
        default=64,
        help="Number of slots (objects) in the scene",
    )
    parser.add_argument(
        "--num_prototypes", type=int, default=10, help="Number of prototype archetypes"
    )
    parser.add_argument(
        "--slot_lr", type=float, default=0.01, help="Learning rate for slot parameters"
    )
    parser.add_argument(
        "--prototype_lr",
        type=float,
        default=0.001,
        help="Learning rate for prototype parameters",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.0,
        help="Standard deviation of noise for energy-based model",
    )
    parser.add_argument(
        "--viz_interval",
        type=int,
        default=50,
        help="Interval for visualization updates",
    )
    parser.add_argument(
        "--image_index",
        type=int,
        default=-1,
        help="Index of the image to use (-1 for random)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)",
    )
    parser.add_argument(
        "--ico_level",
        type=int,
        default=4,
        help="Subdivision level for icosphere (higher = more detailed mesh)",
    )

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Setup visualization exporter
    viz_exporter = VizExporter(local_mode=True)

    # Get all PNG files in the directory
    depth_files = glob(os.path.join(args.data_path, "*.png"))
    if not depth_files:
        print(f"No PNG files found in {args.data_path}")
        sys.exit(1)

    print(f"Found {len(depth_files)} depth images")

    # Select an image
    if args.image_index == -1:
        # Random selection
        image_path = random.choice(depth_files)
        print(f"Randomly selected: {image_path}")
    else:
        # Specific index
        if args.image_index >= len(depth_files):
            print(
                f"Image index {args.image_index} out of range (0-{len(depth_files) - 1})"
            )
            sys.exit(1)
        image_path = depth_files[args.image_index]
        print(f"Selected image at index {args.image_index}: {image_path}")

    # Load and process the depth image
    try:
        # Load depth image
        depth_img = Image.open(image_path)

        # Process depth image
        depth_img_tensor, point_cloud, original_depth = process_depth_image(image_path)

        # Convert to device
        point_cloud = point_cloud.to(device)

        print(f"Point cloud shape: {point_cloud.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Create energy-based optimizer (native version)
    optimizer = EnergyBasedSceneOptimizer(
        num_slots=args.num_slots,
        num_prototypes=args.num_prototypes,
        device=device,
        slot_lr=args.slot_lr,
        prototype_lr=args.prototype_lr,
        noise_std=args.noise_std,
        ico_level=args.ico_level,
    )

    # Define visualization callback
    def viz_callback(optimizer, point_cloud, iteration):
        if iteration % args.viz_interval == 0:
            # Get all slot meshes
            slot_meshes, _, _ = optimizer.get_slots()

            # Get the actual learned prototype weights - using the proper weights rather than dummy ones
            prototype_weights = optimizer.get_prototype_weights()

            print(
                f"Exporting visualization with {optimizer.num_slots} slots and {optimizer.num_prototypes} prototypes"
            )
            print(f"Prototype weights shape: {prototype_weights.shape}")

            # Export for visualization
            viz_exporter.export_visualization_data(
                epoch=0,  # No epochs in this approach
                batch=iteration,
                depth_img=depth_img_tensor.unsqueeze(0).to(device),  # Add batch dim
                points_list=[point_cloud],  # Wrap in list for compatibility
                slots=slot_meshes,
                prototype_offsets=optimizer.prototype_offsets,
                prototype_weights=prototype_weights,  # Using actual learned weights
                scales=optimizer.slot_params[:, 6:].unsqueeze(0),  # Add batch dim
                transforms=optimizer.convert_transforms_for_viz(),
                loss=optimizer.loss_history[-1] if optimizer.loss_history else 0.0,
                global_chamfer=optimizer.loss_history[-1]
                if optimizer.loss_history
                else 0.0,
                per_slot_chamfer=0.0,  # Not computed in this approach
            )
            print(f"Iteration {iteration}: Loss = {optimizer.loss_history[-1]:.6f}")

    # Run optimization
    print(f"\nStarting native energy-based optimization with:")
    print(f"  Num slots: {args.num_slots}")
    print(f"  Num prototypes: {args.num_prototypes}")
    print(f"  Ico sphere level: {args.ico_level}")
    print(f"  Slot learning rate: {args.slot_lr}")
    print(f"  Prototype learning rate: {args.prototype_lr}")
    print(f"  Noise std (for EBM): {args.noise_std}")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Using PyTorch3D: No (native implementation)")

    optimizer.optimize(
        point_cloud, num_iterations=args.num_iterations, callback=viz_callback
    )

    print("\nOptimization complete!")
    print(f"Final loss: {optimizer.loss_history[-1]:.6f}")

    # Final visualization
    viz_callback(optimizer, point_cloud, args.num_iterations)

    print("\nVisualizations have been exported to the viz_server.")
    print("Run the visualization server to see the results:")
    print("  sh run_viz_server.sh")


if __name__ == "__main__":
    main()
