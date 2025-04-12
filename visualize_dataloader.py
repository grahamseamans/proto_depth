"""
Visualize scene generation and optimization.
Generates visualizations of the scene layout and point clouds.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src import Scene


def save_scene_data(scene, output_dir):
    """Save depth maps and point clouds from each camera view"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get point clouds from true scene state
    point_clouds = scene.get_ground_truth_clouds()

    # Save point clouds
    for i, points in enumerate(point_clouds):
        points = points.squeeze().cpu()
        print(f"\nCamera {i} points shape: {points.shape}")
        torch.save(points, output_dir / f"points_{i}.pt")

    # Save scene layout visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot true camera positions
    camera_positions = torch.stack(
        [cam.extrinsics.cam_pos().squeeze() for cam in scene.true_cameras]
    )
    ax.scatter(
        camera_positions[:, 0].cpu(),
        camera_positions[:, 2].cpu(),
        camera_positions[:, 1].cpu(),
        c="blue",
        marker="^",
        s=100,
        label="Cameras",
    )

    # Plot true object positions
    positions = scene.true_positions
    ax.scatter(
        positions[:, 0].cpu(),
        positions[:, 2].cpu(),
        positions[:, 1].cpu(),
        c="red",
        marker="o",
        s=100,
        label="Objects",
    )

    # Add labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.legend()
    plt.savefig(output_dir / "scene_layout.png")
    plt.close()

    return point_clouds


def main():
    """Main visualization script"""
    print("Starting visualization...")

    # Create test scene
    print("Creating test scene...")
    scene = Scene(
        num_objects=2,  # Two objects
        device="cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
    )

    # Save scene data (depth maps and point clouds)
    print("Saving scene data...")
    save_scene_data(scene, "tests/output")

    print("Done! Results saved to tests/output/")


if __name__ == "__main__":
    main()
