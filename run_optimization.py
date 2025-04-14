"""
Run scene optimization on RunPod and save state at each iteration.
Saves data in viz server format for visualization.
"""

import json
import os
import time
import torch
from pathlib import Path
from src import Scene


def save_frame_json(
    out_path,
    true_cam2world,
    pred_cam2world,
    true_object_positions,
    true_object_rotations,
    true_object_scales,
    pred_object_positions,
    pred_object_rotations,
    pred_object_scales,
    point_clouds,
):
    """Write a single frame's data to a JSON file."""
    data = {
        "true": {
            "camera": {"transforms": true_cam2world},
            "objects": {
                "positions": true_object_positions,
                "rotations": true_object_rotations,
                "scales": true_object_scales,
            },
        },
        "pred": {
            "camera": {"transforms": pred_cam2world},
            "objects": {
                "positions": pred_object_positions,
                "rotations": pred_object_rotations,
                "scales": pred_object_scales,
            },
        },
        "point_clouds": point_clouds,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def save_iteration_data(scene: Scene, iteration, output_dir):
    """Save scene state and point clouds for this iteration in the new unified format"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    iter_dir = output_dir / f"iter_{iteration:04d}"
    iter_dir.mkdir(exist_ok=True)

    # Save data for each frame
    for frame in range(scene.num_frames):
        # Gather required data for the new spec
        # True camera/object params
        # Get camera transforms - ensure single list level for 4x4 matrices
        true_cam2world = [
            camera.extrinsics.inv_view_matrix().squeeze().cpu().tolist()
            for camera in scene.true_cameras
        ]
        pred_cam2world = [
            camera.extrinsics.inv_view_matrix().squeeze().cpu().tolist()
            for camera in scene.pred_cameras
        ]

        # Get object positions, rotations, scales
        true_object_positions = scene.true_positions[frame].cpu().tolist()
        true_object_rotations = scene.true_rotations[frame].cpu().tolist()
        true_object_scales = scene.true_scales[frame].cpu().tolist()
        pred_object_positions = scene.pred_positions[frame].cpu().tolist()
        pred_object_rotations = scene.pred_rotations[frame].cpu().tolist()
        pred_object_scales = scene.pred_scales[frame].cpu().tolist()

        # Get point clouds (still in camera-local space)
        raw_clouds = scene.get_ground_truth_clouds(frame)

        point_clouds = [pc.cpu().tolist() for pc in raw_clouds]

        # Save frame data
        save_frame_json(
            str(iter_dir / f"frame_{frame:04d}.json"),
            true_cam2world,
            pred_cam2world,
            true_object_positions,
            true_object_rotations,
            true_object_scales,
            pred_object_positions,
            pred_object_rotations,
            pred_object_scales,
            point_clouds,
        )

    # Save metadata
    metadata = {
        "iteration": iteration,
        "num_frames": scene.num_frames,
        "timestamp": time.time(),
        "loss": [float(scene.compute_energy(f)) for f in range(scene.num_frames)],
    }
    with open(iter_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)


def main():
    """Main optimization script"""
    print("Creating scene...")
    scene = Scene(
        num_objects=2,
        num_frames=30,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Create output directory with timestamp
    timestamp = int(time.time())
    output_dir = Path("viz_server/data") / f"run_{timestamp}"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save run metadata
    run_metadata = {
        "timestamp": timestamp,
        "num_frames": scene.num_frames,
        "num_objects": scene.num_objects,
        "description": "Time-varying scene optimization",
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f)

    # Save initial state
    print("Saving initial state...")
    save_iteration_data(scene, 0, output_dir)

    # TODO: Add optimization loop here
    # For now just save a few iterations with random changes
    print("Simulating optimization...")
    for i in range(1, 2):
        # Add random changes to predicted positions
        scene.pred_positions += torch.randn_like(scene.pred_positions) * 0.01

        # Save this iteration
        print(f"Saving iteration {i}...")
        save_iteration_data(scene, i, output_dir)

    print(f"Done! Data saved to {output_dir}")
    print("Run the viz server and open http://localhost:5000 to view results")


if __name__ == "__main__":
    main()
