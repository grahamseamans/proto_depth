"""
Optimize scene state over time and save visualization data.
"""

import torch
import json
import time
from pathlib import Path
import numpy as np
from src import Scene


def save_iteration_data(scene, iteration, output_dir):
    """Save scene state for all frames at this iteration"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    iter_dir = output_dir / f"iter_{iteration:04d}"
    iter_dir.mkdir(exist_ok=True)

    # Save data for each frame
    for frame in range(scene.num_frames):
        frame_data = scene.get_visualization_data(frame)

        # Save frame data
        with open(iter_dir / f"frame_{frame:04d}.json", "w") as f:
            json.dump(frame_data, f)

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
    for i in range(1, 5):
        # Add random changes to predicted positions
        scene.pred_positions += torch.randn_like(scene.pred_positions) * 0.01

        # Save this iteration
        print(f"Saving iteration {i}...")
        save_iteration_data(scene, i, output_dir)

    print(f"Done! Data saved to {output_dir}")
    print("Run the viz server and open http://localhost:5000 to view results")


if __name__ == "__main__":
    main()
