"""
Visualize point clouds using Open3D.
"""

import torch
import open3d as o3d
import numpy as np
from pathlib import Path

from core import Scene


def visualize_scene(scene):
    """Visualize point clouds from scene using Open3D"""
    # Get point clouds from scene
    batch = scene.get_batch()
    point_clouds = batch["point_clouds"]

    # Create Open3D point clouds with different colors for each view
    o3d_clouds = []
    for i, points in enumerate(point_clouds):
        points = points.detach().cpu().numpy()

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Color points by view index
        colors = np.zeros((len(points), 3))
        colors[:, i % 3] = 1.0  # Cycle through RGB
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d_clouds.append(pcd)

    # Combine all point clouds
    combined = o3d.geometry.PointCloud()
    for pcd in o3d_clouds:
        combined += pcd

    # Add coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    # Visualize
    o3d.visualization.draw_geometries([combined, frame])


def main():
    """Main visualization script"""
    print("Creating test scene...")
    scene = Scene(num_objects=2, device="cuda")

    print("Visualizing point clouds...")
    visualize_scene(scene)


if __name__ == "__main__":
    main()
