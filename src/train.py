"""
Main training script for 4D reality learning system.
"""

import torch
import argparse
from pathlib import Path

from core.scene import Scene
from viz_exporter import VizExporter


def train(args):
    """Main training function"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize visualization
    viz = VizExporter(local_mode=True)

    def visualization_callback(scene, loss, iteration):
        """Callback for visualization during optimization"""
        if iteration % args.viz_interval == 0:
            # Get current scene data
            batch = scene.get_batch()

            # Export visualization data
            viz.export_visualization_data(
                epoch=0,  # Single epoch for now
                batch=iteration,
                depth_img=batch["depth_maps"][0].unsqueeze(0),  # First frame only
                points_list=[
                    target["point_clouds"][0],
                    batch["point_clouds"][0],
                ],  # First frame
                slots=None,  # Not using slots yet
                prototype_offsets=None,  # Not using prototypes yet
                prototype_weights=None,
                scales=scene.scales,
                transforms=scene.positions,
                loss=loss,
            )

    # Training loop
    for scene_idx in range(args.num_scenes):
        print(f"\nOptimizing scene {scene_idx + 1}/{args.num_scenes}")

        # Create target scene
        target_scene = Scene(num_objects=args.num_objects, device=device)
        target = target_scene.get_batch()

        # Create scene to optimize
        scene = Scene(num_objects=args.num_objects, device=device)

        # Optimize scene
        loss_history = scene.optimize(
            target_points=target["point_clouds"],
            num_iterations=args.num_iterations,
            callback=visualization_callback if not args.no_viz else None,
        )

        print(f"Final loss: {loss_history[-1]:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train 4D reality learning system")

    # Dataset parameters
    parser.add_argument(
        "--num-scenes", type=int, default=100, help="Number of scenes to generate"
    )
    parser.add_argument(
        "--num-objects", type=int, default=2, help="Number of objects per scene"
    )

    # Optimization parameters
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of optimization iterations per scene",
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

    train(args)


if __name__ == "__main__":
    main()
