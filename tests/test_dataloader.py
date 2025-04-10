"""
Test script for visualizing scene generation.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from core import Scene


def visualize_scene():
    """Generate and visualize a test scene"""
    print("Initializing scene...")
    scene = Scene(
        num_objects=2,  # Two objects
    )

    print("Getting scene data...")
    batch = scene.get_batch()
    depth_maps = batch["depth_maps"]

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

    # Show camera positions and objects in 3D
    print("Plotting scene layout...")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot camera positions
    positions = []
    for camera in scene.cameras:
        pos = camera.extrinsics.cam_pos().squeeze()
        positions.append(pos)
    positions = torch.stack(positions).detach().cpu().numpy()

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c="blue",
        label="Camera Positions",
    )

    # Plot camera path
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], c="blue", alpha=0.3)

    # Plot objects
    object_positions = scene.positions.detach().cpu().numpy()
    ax.scatter(
        object_positions[:, 0],
        object_positions[:, 1],
        object_positions[:, 2],
        c="red",
        s=100,
        label="Objects",
    )

    # Add arrows for object orientations
    rotations = scene.rotations.detach().cpu().numpy()
    for pos, rot in zip(object_positions, rotations):
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
