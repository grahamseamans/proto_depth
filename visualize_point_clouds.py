"""
Visualize point clouds using Open3D.
Press keys 1-8 to switch between camera views.
"""

import torch
import open3d as o3d
import numpy as np
from pathlib import Path


def create_point_cloud(points):
    """Create Open3D point cloud"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def visualize_point_clouds():
    """Visualize point clouds with keyboard controls"""
    output_dir = Path("tests/output")

    # Load point clouds
    point_clouds = []  # List of point clouds
    frames = []  # List of coordinate frames
    for i in range(8):
        # Load points
        points = torch.load(
            output_dir / f"points_{i}.pt", map_location=torch.device("cpu")
        )
        points = points.squeeze().numpy()  # [N, 3]

        # Create point cloud
        pcd = create_point_cloud(points)
        point_clouds.append(pcd)

        # Add coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        frames.append(frame)

        # Print camera info
        if i < 4:
            # First diagonal: (-5,3,-5) to (5,3,5)
            t = -5 + (10 / 3) * i
            print(f"Camera {i}: ({t:.1f}, 3, {t:.1f})")
        else:
            # Second diagonal: (-5,3,5) to (5,3,-5)
            t = -5 + (10 / 3) * (i - 4)
            print(f"Camera {i}: ({t:.1f}, 3, {-t:.1f})")

    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Camera Views", width=1600, height=900)

    # Show first view
    active_idx = 0
    vis.add_geometry(point_clouds[active_idx])
    vis.add_geometry(frames[active_idx])

    # Get view control
    view_control = vis.get_view_control()

    def set_camera_view(view_control):
        """Position camera at origin"""
        view_control.set_lookat([0, 0, -10])  # Look at point in front (-Z)
        view_control.camera_local_translate(0, 0, 0)  # Stay at origin

    # Set initial view
    set_camera_view(view_control)

    # Register key callbacks (1-8 for cameras)
    def make_callback(camera_idx):
        def callback(vis):
            nonlocal active_idx
            print(f"Switching to Camera {camera_idx} view")

            # Clear geometries
            vis.clear_geometries()

            # Show new view
            active_idx = camera_idx
            vis.add_geometry(point_clouds[active_idx])
            vis.add_geometry(frames[active_idx])

            # Reset camera view
            set_camera_view(view_control)

            return False

        return callback

    for i in range(8):
        # Register keys 1-8
        vis.register_key_callback(ord(str(i + 1)), make_callback(i))

    # Run visualizer
    vis.run()
    vis.destroy_window()


def main():
    """Main visualization script"""
    print("Loading point clouds...")
    print("Press keys 1-8 to switch between camera views")
    visualize_point_clouds()


if __name__ == "__main__":
    main()
