"""
Optimize scene state and visualize progress.
Shows ground truth vs current state at each iteration.
"""

import torch
import open3d as o3d
import numpy as np
from pathlib import Path

from src import Scene


def create_point_cloud(points, color=[1, 0, 0]):
    """Create Open3D point cloud with given color"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
    return pcd


def visualize_iteration(scene, ground_truth_points, iteration):
    """
    Visualize current state vs ground truth.

    Args:
        scene: Current scene state
        ground_truth_points: List of point clouds from cameras
        iteration: Current iteration number
    """
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(f"Iteration {iteration}")

    # Add coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(frame)

    # Add ground truth points (red)
    for points in ground_truth_points:
        gt_cloud = create_point_cloud(points, color=[1, 0, 0])  # Red
        vis.add_geometry(gt_cloud)

    # Get current state's prediction
    pred_points = scene.get_scene_points()

    # Add predicted points (blue)
    for points in pred_points:
        pred_cloud = create_point_cloud(points, color=[0, 0, 1])  # Blue
        vis.add_geometry(pred_cloud)

    # Update view
    vis.poll_events()
    vis.update_renderer()

    # Save screenshot
    vis.capture_screen_image(f"iteration_{iteration:04d}.png")
    vis.destroy_window()


def main():
    """Main optimization script"""
    print("Creating scene...")
    scene = Scene(
        num_objects=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Get ground truth point clouds
    print("Getting ground truth...")
    ground_truth = scene.get_scene_points()

    # Randomize scene state to start from scratch
    print("Randomizing initial state...")
    scene.positions.data = torch.randn_like(scene.positions) * 2.0
    scene.rotations.data = torch.randn_like(scene.rotations) * np.pi
    scene.scales.data = torch.ones_like(scene.scales) * 2.0

    # Optimize with visualization
    print("Starting optimization...")

    def vis_callback(scene, loss, iter_num):
        if iter_num % 10 == 0:  # Visualize every 10 iterations
            print(f"Iteration {iter_num}: loss = {loss:.6f}")
            visualize_iteration(scene, ground_truth, iter_num)

    scene.optimize(ground_truth, num_iterations=100, callback=vis_callback)

    print("Done! Check iteration_*.png files for visualization.")


if __name__ == "__main__":
    main()
