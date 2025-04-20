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
    # create idenity camera transform
    camera_transforms = [np.eye(4) for _ in range(len(data["point_clouds"]))]

    point_clouds = data["point_clouds"]

    print("\nLoaded from JSON:")
    print(f"Number of cameras: {len(camera_transforms)}")
    print(f"Number of point clouds: {len(point_clouds)}")

    # Convert to numpy arrays
    camera_transforms = [np.array(transform) for transform in camera_transforms]
    point_clouds = [np.array(pc) for pc in point_clouds]

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
    for i, points in enumerate(world_clouds):
        # for i, points in enumerate(point_clouds):
        # Create and add colored point cloud
        pcd = create_point_cloud(points, colors[i])
        vis.add_geometry(pcd)

    # Add coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(frame)

    # --- Add ground truth bunny mesh(es) at true locations ---
    import copy as _copy
    import math

    bunny_mesh = o3d.io.read_triangle_mesh("3d_models/bunny.obj")
    bunny_mesh.compute_vertex_normals()
    bunny_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # light gray

    # Get ground truth object transforms from JSON
    gt_positions = np.array(data["true"]["objects"]["positions"])  # [num_objects, 3]
    gt_rotations = np.array(
        data["true"]["objects"]["rotations"]
    )  # [num_objects, 3] (Euler XYZ, radians)
    gt_scales = np.array(data["true"]["objects"]["scales"])  # [num_objects, 1]

    for i in range(gt_positions.shape[0]):
        mesh = _copy.deepcopy(bunny_mesh)
        # Apply scale
        scale = float(gt_scales[i][0])
        mesh.scale(scale, center=[0, 0, 0])
        # Apply rotation (XYZ Euler, radians)
        rx, ry, rz = gt_rotations[i]
        # R = o3d.geometry.get_rotation_matrix_from_xyz([rx, ry, rz])
        # mesh.rotate(R, center=[0, 0, 0])
        # Apply translation
        mesh.translate(gt_positions[i])
        vis.add_geometry(mesh)
    # --- End ground truth mesh addition ---

    # For each camera, add an arrow at the camera position, pointing along +Z in camera space
    for i, transform in enumerate(camera_transforms):
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.02,
            cone_radius=0.04,
            cylinder_height=0.1,
            cone_height=0.05,
        )
        # Flip arrow direction: rotate 180 degrees around Y axis before applying camera transform
        R_flip = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.pi, 0])
        arrow.rotate(R_flip, center=[0, 0, 0])
        arrow.paint_uniform_color(colors[i])
        arrow.transform(transform)
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
