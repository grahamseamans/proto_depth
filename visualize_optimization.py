"""
Visualize optimization progress using saved data.
Uses Open3D for interactive visualization.
"""

import torch
import open3d as o3d
import numpy as np
from pathlib import Path


def create_point_cloud(points, color=[1, 0, 0]):
    """Create Open3D point cloud with given color"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
    return pcd


def load_iteration(iter_dir, num_views):
    """Load point clouds for a specific iteration"""
    clouds = []
    for i in range(num_views):
        points = torch.load(iter_dir / f"pred_points_{i:02d}.pt")
        clouds.append(points)
    return clouds


def visualize_iteration(ground_truth, pred_points):
    """Show ground truth vs predicted point clouds"""
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(frame)

    # Add ground truth points (red)
    for points in ground_truth:
        gt_cloud = create_point_cloud(points, color=[1, 0, 0])  # Red
        vis.add_geometry(gt_cloud)

    # Add predicted points (blue)
    for points in pred_points:
        pred_cloud = create_point_cloud(points, color=[0, 0, 1])  # Blue
        vis.add_geometry(pred_cloud)

    # Set default camera viewpoint
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])

    # Run visualization
    vis.run()
    vis.destroy_window()


def main():
    """Main visualization script"""
    data_dir = Path("optimization_data")
    if not data_dir.exists():
        print("No optimization data found. Run run_optimization.py first.")
        return

    # Load ground truth
    print("Loading ground truth...")
    ground_truth = []
    i = 0
    while (data_dir / f"ground_truth_{i:02d}.pt").exists():
        points = torch.load(data_dir / f"ground_truth_{i:02d}.pt")
        ground_truth.append(points)
        i += 1
    num_views = i

    # Find all iteration directories
    iter_dirs = sorted([d for d in data_dir.iterdir() if d.name.startswith("iter_")])
    if not iter_dirs:
        print("No iteration data found.")
        return

    print(f"Found {len(iter_dirs)} iterations")
    print("Press 'n' to view next iteration, 'p' for previous, 'q' to quit")

    # Visualize each iteration
    current_iter = 0
    while True:
        print(f"\nViewing iteration {current_iter}")
        pred_points = load_iteration(iter_dirs[current_iter], num_views)
        visualize_iteration(ground_truth, pred_points)

        # Get user input
        cmd = input("Command (n/p/q): ").lower()
        if cmd == "q":
            break
        elif cmd == "n" and current_iter < len(iter_dirs) - 1:
            current_iter += 1
        elif cmd == "p" and current_iter > 0:
            current_iter -= 1


if __name__ == "__main__":
    main()
