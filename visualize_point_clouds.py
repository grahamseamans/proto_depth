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

    gt_point_clouds = data["true"].get("point_clouds", [])
    pred_point_clouds = data["pred"].get("point_clouds", [])

    print("\nLoaded from JSON:")
    print(f"Number of cameras: {len(camera_transforms)}")
    print(f"Number of ground truth point clouds: {len(gt_point_clouds)}")
    print(f"Number of predicted point clouds: {len(pred_point_clouds)}")

    # Convert to numpy arrays
    gt_point_clouds = [np.array(pc) for pc in gt_point_clouds]
    pred_point_clouds = [np.array(pc) for pc in pred_point_clouds]

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Clouds", width=1600, height=900)

    # Generate colors for each camera's point cloud
    num_cameras = len(gt_point_clouds)
    gt_colors = []
    pred_colors = []
    for i in range(num_cameras):
        # Generate evenly spaced hues, convert to RGB
        hue = i / num_cameras
        # Ground truth: solid, bright
        if hue < 1 / 3:
            gt_colors.append([1 - 3 * hue, 3 * hue, 0])  # Red to Green
        elif hue < 2 / 3:
            gt_colors.append(
                [0, 1 - 3 * (hue - 1 / 3), 3 * (hue - 1 / 3)]
            )  # Green to Blue
        else:
            gt_colors.append(
                [3 * (hue - 2 / 3), 0, 1 - 3 * (hue - 2 / 3)]
            )  # Blue to Red
        # Predicted: faded version of same color
        pred_colors.append([c * 0.5 + 0.5 for c in gt_colors[-1]])

    # Add ground truth point clouds (solid)
    for i, points in enumerate(gt_point_clouds):
        pcd = create_point_cloud(points, gt_colors[i])
        vis.add_geometry(pcd)

    # Add predicted point clouds (semi-transparent, if possible)
    for i, points in enumerate(pred_point_clouds):
        pcd = create_point_cloud(points, pred_colors[i])
        # Open3D doesn't support alpha, but we can use lighter color
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
    gt_rotmats = np.array(data["true"]["objects"]["rot_mats"])  # [num_objects, 3, 3]
    gt_scales = np.array(data["true"]["objects"]["scales"])  # [num_objects, 1]

    for i in range(gt_positions.shape[0]):
        mesh = _copy.deepcopy(bunny_mesh)
        # Apply scale
        scale = float(gt_scales[i][0])
        mesh.scale(scale, center=[0, 0, 0])
        # Apply rotation (rotation matrix)
        rot_matrix = gt_rotmats[i]
        mesh.rotate(rot_matrix, center=[0, 0, 0])
        # Apply translation
        mesh.translate(gt_positions[i])
        vis.add_geometry(mesh)
    # --- End ground truth mesh addition ---

    # --- Add predicted bunny mesh(es) at predicted locations ---
    pred_positions = np.array(data["pred"]["objects"]["positions"])  # [num_objects, 3]
    pred_rotmats = np.array(data["pred"]["objects"]["rot_mats"])  # [num_objects, 3, 3]
    pred_scales = np.array(data["pred"]["objects"]["scales"])  # [num_objects, 1]

    for i in range(pred_positions.shape[0]):
        mesh = _copy.deepcopy(bunny_mesh)
        # Apply scale
        scale = float(pred_scales[i][0])
        mesh.scale(scale, center=[0, 0, 0])
        # Apply rotation (rotation matrix)
        rot_matrix = pred_rotmats[i]
        mesh.rotate(rot_matrix, center=[0, 0, 0])
        # Apply translation
        mesh.translate(pred_positions[i])
        mesh.paint_uniform_color([0.2, 0.4, 1.0])  # blue for predicted
        vis.add_geometry(mesh)
    # --- End predicted mesh addition ---

    # For each camera, add an arrow at the camera position, pointing along +Z in camera space (ground truth)
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
        arrow.paint_uniform_color(gt_colors[i])
        arrow.transform(transform)
        vis.add_geometry(arrow)

    # Add arrows for predicted cameras (use black)
    pred_camera_transforms = data["pred"]["camera"]["transforms"]
    for i, transform in enumerate(pred_camera_transforms):
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.02,
            cone_radius=0.04,
            cylinder_height=0.1,
            cone_height=0.05,
        )
        R_flip = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.pi, 0])
        arrow.rotate(R_flip, center=[0, 0, 0])
        arrow.paint_uniform_color(pred_colors[i])
        arrow.transform(np.array(transform))
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
