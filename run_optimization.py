"""
Run scene optimization on RunPod and save state at each iteration.
"""

import torch
import numpy as np
from pathlib import Path

from src import Scene


def save_iteration_data(scene, ground_truth, iteration, output_dir):
    """Save scene state and point clouds for this iteration"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    iter_dir = output_dir / f"iter_{iteration:04d}"
    iter_dir.mkdir(exist_ok=True)

    # Save scene parameters
    torch.save(
        {
            "positions": scene.positions.detach().cpu(),
            "rotations": scene.rotations.detach().cpu(),
            "scales": scene.scales.detach().cpu(),
        },
        iter_dir / "scene_state.pt",
    )

    # Save ground truth point clouds (only on first iteration)
    if iteration == 0:
        for i, points in enumerate(ground_truth):
            torch.save(points.detach().cpu(), output_dir / f"ground_truth_{i:02d}.pt")

    # Save current predicted point clouds
    pred_points = scene.get_scene_points()
    for i, points in enumerate(pred_points):
        torch.save(points.detach().cpu(), iter_dir / f"pred_points_{i:02d}.pt")


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

    # Create output directory
    output_dir = Path("optimization_data")
    output_dir.mkdir(exist_ok=True)

    # Optimize and save data
    print("Starting optimization...")

    def save_callback(scene, loss, iter_num):
        if iter_num % 10 == 0:  # Save every 10 iterations
            print(f"Iteration {iter_num}: loss = {loss:.6f}")
            save_iteration_data(scene, ground_truth, iter_num, output_dir)

    scene.optimize(ground_truth, num_iterations=100, callback=save_callback)

    print("Done! Data saved to optimization_data/")


if __name__ == "__main__":
    main()
