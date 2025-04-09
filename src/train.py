"""
Main training script for 4D reality learning system.
"""

import torch
import argparse
from pathlib import Path

from core import SceneState, EnergyOptimizer, SceneDataset
from viz_exporter import VizExporter


def train(args):
    """Main training function"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize dataset
    dataset = SceneDataset(
        num_scenes=args.num_scenes,
        num_frames=args.num_frames,
        num_objects=args.num_objects,
        models_dir=args.models_dir,
        device=device,
    )
    print(f"Dataset initialized with {len(dataset)} scenes")

    # Initialize visualization
    viz = VizExporter(local_mode=True)

    def visualization_callback(scene_state, energy, iteration):
        """Callback for visualization during optimization"""
        if iteration % args.viz_interval == 0:
            # TODO: Implement visualization
            # This will need to:
            # 1. Get current scene state
            # 2. Render depth maps
            # 3. Export to visualization server
            pass

    # Training loop
    for scene_idx in range(len(dataset)):
        print(f"\nOptimizing scene {scene_idx + 1}/{len(dataset)}")

        # Get scene data
        scene_data = dataset[scene_idx]
        depth_maps = scene_data["depth_maps"]
        camera_positions = scene_data["camera_positions"]
        camera_rotations = scene_data["camera_rotations"]

        # Initialize scene state
        scene_state = SceneState(num_objects=args.num_objects, device=device)

        # Initialize optimizer
        optimizer = EnergyOptimizer(
            scene_state=scene_state, learning_rate=args.learning_rate
        )

        # Optimize scene
        loss_history = optimizer.optimize(
            point_cloud=depth_maps,  # TODO: Convert depth maps to point clouds
            num_iterations=args.num_iterations,
            callback=visualization_callback if not args.no_viz else None,
        )

        print(f"Final energy: {loss_history[-1]:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train 4D reality learning system")

    # Dataset parameters
    parser.add_argument(
        "--num-scenes", type=int, default=100, help="Number of scenes to generate"
    )
    parser.add_argument(
        "--num-frames", type=int, default=30, help="Number of frames per scene"
    )
    parser.add_argument(
        "--num-objects", type=int, default=2, help="Number of objects per scene"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="3d_models",
        help="Directory containing 3D models",
    )

    # Optimization parameters
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of optimization iterations per scene",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for optimization",
    )

    # Visualization parameters
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument(
        "--viz-interval",
        type=int,
        default=10,
        help="Visualization interval (iterations)",
    )

    args = parser.parse_args()

    # Create models directory if it doesn't exist
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
