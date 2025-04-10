"""
Visualize point clouds using Open3D.
"""

import torch
import open3d as o3d
import numpy as np
from pathlib import Path


def create_point_cloud(points, color=[1, 0, 0]):
    """Create Open3D point cloud with depth coloring"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Color points by depth (Z coordinate)
    z = points[:, 2]  # Depth values
    colors = np.zeros((len(points), 3))
    # Map depth to color (red=near, blue=far)
    colors[:, 0] = (z - z.min()) / (z.max() - z.min())  # Red channel
    colors[:, 2] = 1 - colors[:, 0]  # Blue channel
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize_point_clouds():
    """Visualize point clouds from each camera"""
    output_dir = Path("tests/output")

    # Load point clouds
    clouds = []
    for i in range(8):
        # Load points
        points = torch.load(
            output_dir / f"points_{i}.pt", map_location=torch.device("cpu")
        )
        points = points.squeeze().numpy()  # [N, 3]

        # Calculate grid position (2x4 grid)
        grid_row = i // 4  # 2 rows
        grid_col = i % 4  # 4 columns
        offset = np.array(
            [
                grid_col * 10,  # X offset (10 units between columns)
                -grid_row * 10,  # Y offset (10 units between rows)
                0,  # No Z offset
            ]
        )

        # Offset points to grid position
        points = points + offset

        # Create point cloud
        pcd = create_point_cloud(points)

        # Add coordinate frame at grid position
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        frame_points = np.asarray(frame.vertices) + offset
        frame.vertices = o3d.utility.Vector3dVector(frame_points)

        # Group point cloud and frame
        clouds.extend([pcd, frame])

        # Print camera info
        if i < 4:
            # First diagonal: (-5,3,-5) to (5,3,5)
            t = -5 + (10 / 3) * i
            print(f"Camera {i}: ({t:.1f}, 3, {t:.1f})")
        else:
            # Second diagonal: (-5,3,5) to (5,3,-5)
            t = -5 + (10 / 3) * (i - 4)
            print(f"Camera {i}: ({t:.1f}, 3, {-t:.1f})")

    # Visualize all point clouds
    o3d.visualization.draw_geometries(
        clouds,
        window_name="Camera Views",
        width=1600,
        height=900,
        left=50,
        top=50,
    )


def main():
    """Main visualization script"""
    print("Loading point clouds...")
    visualize_point_clouds()


if __name__ == "__main__":
    main()
