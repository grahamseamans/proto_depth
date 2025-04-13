"""Validate frame data against the specification."""

import glob
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel


class CameraTransform(BaseModel):
    transforms: List[List[List[float]]]


class Objects(BaseModel):
    positions: List[List[float]]
    rotations: List[List[float]]
    scales: List[List[float]]


class SceneState(BaseModel):
    camera: CameraTransform
    objects: Objects


class FrameData(BaseModel):
    true: SceneState
    pred: SceneState
    point_clouds: List[List[List[float]]]


def find_latest_run() -> Optional[str]:
    """Find the most recent run directory."""
    runs = glob.glob("viz_server/data/run_*")
    if not runs:
        return None

    # Sort by creation time to get the most recent
    latest = max(runs, key=os.path.getctime)
    return latest


def validate_frame(path: str) -> Tuple[bool, str, Optional[FrameData]]:
    """Validate frame data at path."""
    try:
        with open(path) as f:
            data = FrameData.model_validate_json(f.read())
        return True, "Valid", data
    except Exception as e:
        return False, str(e), None


def main():
    """Find and validate the latest frame."""
    latest_run = find_latest_run()

    if not latest_run:
        print("No run directories found in viz_server/data/")
        return

    print(f"\nFound latest run: {latest_run}")
    frame_path = os.path.join(latest_run, "iter_0000", "frame_0000.json")

    if not os.path.exists(frame_path):
        print(f"No frame found at {frame_path}")
        return

    print(f"Validating frame: {frame_path}")
    is_valid, message, data = validate_frame(frame_path)

    if is_valid:
        print("\n✅ Frame data is valid!")
        # Print some basic info about the data
        n_cameras = len(data.true.camera.transforms)
        n_objects = len(data.true.objects.positions)
        n_points = [len(pc) for pc in data.point_clouds]
        print(f"\nSummary:")
        print(f"- Number of cameras: {n_cameras}")
        print(f"- Number of objects: {n_objects}")
        print(f"- Points per camera: {n_points}")
    else:
        print(f"\n❌ Validation failed: {message}")


if __name__ == "__main__":
    main()
