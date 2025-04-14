"""
Visualize point clouds using Open3D.
"""

import json
import glob
import os
import copy
import open3d as o3d
import numpy as np


def find_latest_run():
    """Find the most recent run directory."""
    runs = glob.glob("viz_server/data/run_*")
    if not runs:
        return None

    def get_timestamp(run_path):
        # Extract run ID from path and convert to int
        run_id = os.path.basename(run_path).split("run_")[1]
        try:
            return int(run_id)
        except ValueError:
            # If parsing fails, return creation time as fallback
            return int(os.path.getctime(run_path))

    # Sort by timestamp to get the most recent
    latest = max(runs, key=get_timestamp)
    print(f"Selected latest run: {latest}")
    return latest


def create_point_cloud(points, color=None):
    """Create Open3D point cloud with optional color"""
    # Convert points to numpy array
    points = np.array(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        # Set the same color for all points
        colors = np.tile(color, (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_point_clouds():
    """Visualize point clouds"""
    # Find latest run
    latest_run = find_latest_run()
    if not latest_run:
        print("No run directories found in viz_server/data/")
        return

    # Construct path to first frame
    json_path = os.path.join(latest_run, "iter_0000", "frame_0000.json")
    if not os.path.exists(json_path):
        print(f"No frame found at {json_path}")
        return

    print(f"Loading frame from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Get camera transforms and point clouds
    camera_transforms = data["true"]["camera"]["transforms"]
    point_clouds = data["point_clouds"]

    print("\nLoaded from JSON:")
    print(f"Number of cameras: {len(camera_transforms)}")
    print(f"Number of point clouds: {len(point_clouds)}")

    # Convert to numpy arrays
    camera_transforms = [np.array(transform) for transform in camera_transforms]
    point_clouds = [np.array(pc) for pc in point_clouds]

    # # Print camera transforms
    # print("\nCamera transforms (camera to world):")
    # for i, transform in enumerate(camera_transforms):
    #     print(f"\nCamera {i}:")
    #     print(
    #         f"Position: [{transform[0, 3]:.3f}, {transform[1, 3]:.3f}, {transform[2, 3]:.3f}]"
    #     )
    #     print(
    #         f"Right: [{transform[0, 0]:.3f}, {transform[1, 0]:.3f}, {transform[2, 0]:.3f}]"
    #     )
    #     print(
    #         f"Up: [{transform[0, 1]:.3f}, {transform[1, 1]:.3f}, {transform[2, 1]:.3f}]"
    #     )
    #     print(
    #         f"Forward: [{transform[0, 2]:.3f}, {transform[1, 2]:.3f}, {transform[2, 2]:.3f}]"
    #     )
    #     # Print inverse (world to camera) matrix
    #     world2cam = np.linalg.inv(transform)
    #     print("\nWorld to camera matrix:")
    #     print(world2cam)

    # Transform each point cloud to world space
    world_clouds = []
    for i, (points, transform) in enumerate(zip(point_clouds, camera_transforms)):
        print(f"\nCamera {i}:")
        print(f"Point cloud shape: {points.shape}")
        print(f"Transform matrix:\n{transform}")

        # Add homogeneous coordinate (w=1)
        points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)  # [N, 4]

        # Transform points from camera to world space using cam2world matrix
        points_world_h = transform @ points_h.T  # [4, N]
        points_world_h = points_world_h.T  # [N, 4]

        # Convert back to 3D
        points_world = points_world_h[:, :3]  # [N, 3]
        world_clouds.append(points_world)

        print(f"Transformed shape: {points_world.shape}")

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Clouds", width=1600, height=900)

    # Generate colors for each camera's point cloud
    num_cameras = len(point_clouds)
    colors = []
    for i in range(num_cameras):
        # Generate evenly spaced hues, convert to RGB
        hue = i / num_cameras
        # Simple hue to RGB conversion (you could use a more sophisticated method)
        if hue < 1 / 3:
            colors.append([1 - 3 * hue, 3 * hue, 0])  # Red to Green
        elif hue < 2 / 3:
            colors.append(
                [0, 1 - 3 * (hue - 1 / 3), 3 * (hue - 1 / 3)]
            )  # Green to Blue
        else:
            colors.append([3 * (hue - 2 / 3), 0, 1 - 3 * (hue - 2 / 3)])  # Blue to Red

    # Add each point cloud in world space
    # for i, points in enumerate(world_clouds):
    for i, points in enumerate(point_clouds):
        # Create and add colored point cloud
        pcd = create_point_cloud(points, colors[i])
        vis.add_geometry(pcd)

    # Add coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(frame)

    # Create base arrow at origin pointing in -z
    base_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02,
        cone_radius=0.04,
        cylinder_height=0.1,
        cone_height=0.05,
    )
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    base_arrow.rotate(R, center=[0, 0, 0])
    base_arrow.paint_uniform_color([1, 1, 0])  # Yellow
    vis.add_geometry(base_arrow)

    # Create transformed arrows for each camera
    for i, transform in enumerate(camera_transforms):
        arrow = copy.deepcopy(base_arrow)

        # Print camera info
        pos = transform[:3, 3]  # Fourth column is position
        view_dir = transform[:3, 2]  # Third column is forward direction
        at_point = pos + view_dir  # Point camera is looking at
        print(f"\nCamera {i}:")
        print(f"Position: {pos}")
        print(f"View direction: {view_dir}")
        print(f"Looking at point: {at_point}")

        # Get arrow's direction before transform
        vertices = np.asarray(arrow.vertices)
        tip_vertex = vertices[vertices[:, 1].argmax()]  # Highest y-coord is tip
        base_vertex = vertices[vertices[:, 1].argmin()]  # Lowest y-coord is base
        direction = tip_vertex - base_vertex
        direction = direction / np.linalg.norm(direction)
        print(f"Arrow {i} before transform:")
        print(f"Direction: {direction}")

        # Transform arrow
        arrow.transform(transform)

        # Get arrow's direction after transform
        vertices = np.asarray(arrow.vertices)
        tip_vertex = vertices[vertices[:, 1].argmax()]
        base_vertex = vertices[vertices[:, 1].argmin()]
        direction = tip_vertex - base_vertex
        direction = direction / np.linalg.norm(direction)
        print(f"Arrow {i} after transform:")
        print(f"Direction: {direction}")
        print(f"Should match view direction: {view_dir}")

        arrow.paint_uniform_color(colors[i])  # Same color as point cloud
        vis.add_geometry(arrow)

    # Set view
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])  # Look along -Z axis
    view_control.set_lookat([0, 0, 0])  # Look at origin
    view_control.set_up([0, 1, 0])  # Y is up
    view_control.set_zoom(0.7)

    # Run visualizer
    vis.run()
    vis.destroy_window()


def main():
    """Main visualization script"""
    print("Loading and visualizing point clouds...")
    visualize_point_clouds()


if __name__ == "__main__":
    main()
