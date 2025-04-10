"""
Visualize point clouds using Open3D.
"""

import torch
import open3d as o3d
import numpy as np
from pathlib import Path


def visualize_point_clouds(points_dir):
    """Visualize point clouds from multiple views using Open3D"""
    points_dir = Path(points_dir)

    # Load all point clouds
    point_clouds = []
    for i in range(8):  # 8 views
        points = torch.load(points_dir / f"points_{i}.pt").numpy()

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Color points by view index
        colors = np.zeros((len(points), 3))
        colors[:, i % 3] = 1.0  # Cycle through RGB
        pcd.colors = o3d.utility.Vector3dVector(colors)

        point_clouds.append(pcd)

    # Combine all point clouds
    combined = o3d.geometry.PointCloud()
    for pcd in point_clouds:
        combined += pcd

    # Add coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    # Visualize
    o3d.visualization.draw_geometries([combined, frame])


if __name__ == "__main__":
    visualize_point_clouds("tests/output")
