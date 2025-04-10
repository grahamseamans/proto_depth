"""
Visualize point clouds using Open3D.
"""

import torch
import open3d as o3d
import numpy as np
from pathlib import Path


def visualize_point_clouds():
    """Visualize point clouds from saved depth maps"""
    output_dir = Path("tests/output")

    # Load depth maps and convert to point clouds
    o3d_clouds = []
    for i in range(8):  # 8 views
        # Load raw depth map
        depth = torch.load(
            output_dir / f"depth_{i}_raw.pt", map_location=torch.device("cpu")
        )
        # Remove batch dimension and get valid points
        depth = depth.squeeze()  # [H, W]
        mask = depth < depth.max()
        valid_depths = depth[mask].detach().cpu().numpy()

        # Create pixel coordinates
        h, w = depth.shape
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        x = x[mask].cpu().numpy()
        y = y[mask].cpu().numpy()

        # Calculate grid position (3x3 grid)
        grid_row = i // 3  # 3 rows
        grid_col = i % 3  # 3 columns

        # Scale points to local space and offset to grid position
        points = np.stack(
            [
                (x / w) + (grid_col * 1.5),  # Scale to [0,1] and offset by column
                -(y / h) + (-grid_row * 1.5),  # Scale to [0,1], flip, and offset by row
                -valid_depths * 0.1,  # Scale depth to be smaller
            ],
            axis=1,
        )

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
    print("Loading point clouds...")
    visualize_point_clouds()


if __name__ == "__main__":
    main()
