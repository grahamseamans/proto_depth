"""
Test script for visualizing depth maps and scene layout.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.core import SceneDataset


def visualize_scene():
    """Generate and visualize a test scene"""
    print("Initializing dataset...")
    dataset = SceneDataset(
        num_scenes=1,  # One scene
        num_frames=8,  # 8 views around it
        num_objects=2,  # Two dragons
        models_dir=str(Path(__file__).parent.parent / "3d_models"),
    )

    print("Getting scene data...")
    scene_data = dataset[0]
    depth_maps = scene_data["depth_maps"]

    # Plot depth maps in a grid
    print("Plotting depth maps...")
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for i, depth_map in enumerate(depth_maps):
        ax = axes[i // 4, i % 4]
        # Convert to numpy and squeeze if needed
        depth_np = depth_map.detach().cpu().numpy().squeeze()
        im = ax.imshow(depth_np)
        ax.set_title(f"View {i}")
        ax.axis("off")

    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.suptitle("Depth Maps from Different Views")
    plt.tight_layout()

    # Save depth map figure
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "depth_maps.png")
    print(f"Saved depth maps to {output_dir / 'depth_maps.png'}")
    plt.close()

    # Show camera positions and dragons in 3D
    print("Plotting scene layout...")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot camera positions
    positions = scene_data["camera_positions"].detach().cpu().numpy()
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c="blue",
        label="Camera Positions",
    )

    # Plot camera path
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], c="blue", alpha=0.3)

    # Plot dragons
    state = scene_data["scene_state"]
    dragon_positions = state.get_object_positions().detach().cpu().numpy()
    ax.scatter(
        dragon_positions[:, 0],
        dragon_positions[:, 1],
        dragon_positions[:, 2],
        c="red",
        s=100,
        label="Dragons",
    )

    # Add arrows for dragon orientations
    rotations = state.get_object_rotations().detach().cpu().numpy()
    for pos, rot in zip(dragon_positions, rotations):
        # Just show yaw for now with a small arrow
        yaw = rot[1]  # Assuming [pitch, yaw, roll]
        direction = np.array([np.cos(yaw), 0, np.sin(yaw)])
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            direction[0],
            direction[1],
            direction[2],
            color="red",
            alpha=0.5,
            length=2.0,
        )

    ax.set_title("Scene Layout")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Save scene layout figure
    plt.savefig(output_dir / "scene_layout.png")
    print(f"Saved scene layout to {output_dir / 'scene_layout.png'}")
    plt.close()


if __name__ == "__main__":
    print("Starting visualization test...")
    visualize_scene()
    print("Done!")
