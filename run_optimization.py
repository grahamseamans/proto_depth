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
from tqdm import tqdm


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
    ground_truth_point_clouds,
    predicted_point_clouds,
    true_object_rotmats,
    pred_object_rotmats,
):
    """Write a single frame's data to a JSON file."""
    data = {
        "true": {
            "camera": {"transforms": true_cam2world},
            "objects": {
                "positions": true_object_positions,
                "rotations": true_object_rotations,
                "rot_mats": true_object_rotmats,
                "scales": true_object_scales,
            },
        },
        "pred": {
            "camera": {"transforms": pred_cam2world},
            "objects": {
                "positions": pred_object_positions,
                "rotations": pred_object_rotations,
                "rot_mats": pred_object_rotmats,
                "scales": pred_object_scales,
            },
        },
        "ground_truth_point_clouds": ground_truth_point_clouds,
        "predicted_point_clouds": predicted_point_clouds,
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

    from kaolin.math.quat import rot33_from_quat

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

        # Convert quaternions to rotation matrices and save as lists
        true_object_rotmats = (
            rot33_from_quat(scene.true_rotations[frame].cpu()).detach().numpy().tolist()
        )
        pred_object_rotmats = (
            rot33_from_quat(scene.pred_rotations[frame].cpu()).detach().numpy().tolist()
        )

        # Get per-camera point clouds in world space
        ground_truth_point_clouds = []
        predicted_point_clouds = []
        num_cameras = len(scene.true_cameras)
        for camera_idx in range(num_cameras):
            gt_points_world, pred_points_world = scene.get_point_cloud_pair(
                camera_idx, frame
            )
            ground_truth_point_clouds.append(gt_points_world.cpu().tolist())
            predicted_point_clouds.append(pred_points_world.cpu().tolist())

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
            ground_truth_point_clouds,
            predicted_point_clouds,
            true_object_rotmats,
            pred_object_rotmats,
        )

    # Save metadata
    metadata = {
        "iteration": iteration,
        "num_frames": scene.num_frames,
        "timestamp": time.time(),
        "loss": float(scene.compute_energy()),
    }
    with open(iter_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)


def main():
    """Main optimization script"""
    import torch

    print("Enabling PyTorch anomaly detection...")
    torch.autograd.set_detect_anomaly(True)

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

    # Optimization loop
    num_iters = 100
    lr = 1e-2
    optimizer = torch.optim.Adam(
        [scene.pred_positions, scene.pred_rotations, scene.pred_scales], lr=lr
    )

    print("Starting optimization...")
    pbar = tqdm(range(1, num_iters + 1), desc="Optimizing", ncols=100)
    for i in pbar:
        optimizer.zero_grad()
        loss = scene.compute_energy()
        if torch.isnan(loss):
            print(
                f"[DEBUG] NaN detected in loss at iter {i}, skipping backward/step and save."
            )
            break
        loss.backward()
        optimizer.step()

        # Normalize pred_rotations to unit quaternions after each step
        with torch.no_grad():
            scene.pred_rotations.data = torch.nn.functional.normalize(
                scene.pred_rotations.data, dim=-1
            )
            # Debug: print min/max, check for NaNs/Infs
            min_val = scene.pred_rotations.data.min().item()
            max_val = scene.pred_rotations.data.max().item()
            num_nans = torch.isnan(scene.pred_rotations.data).sum().item()
            num_infs = torch.isinf(scene.pred_rotations.data).sum().item()
            print(
                f"[DEBUG] pred_rotations normalized: min={min_val}, max={max_val}, NaNs={num_nans}, Infs={num_infs}"
            )
            if num_nans > 0 or num_infs > 0:
                print(
                    "[ERROR] NaNs or Infs detected in pred_rotations after normalization!"
                )

        pbar.set_description(f"Iter {i} | Loss: {loss.item():.6f}")
        save_iteration_data(scene, i, output_dir)

    print(f"Done! Data saved to {output_dir}")
    print("Run the viz server and open http://localhost:5000 to view results")


if __name__ == "__main__":
    main()
