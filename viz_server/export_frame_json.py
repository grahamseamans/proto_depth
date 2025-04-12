"""
Export a single frame's data in the new unified JSON format for visualization.

This script is a template for refactoring the exporter to match viz_server/DATA_FORMAT.md.
"""

import json
import os


def export_frame_json(
    out_path,
    true_camera_positions,
    true_camera_rotations,
    pred_camera_positions,
    pred_camera_rotations,
    true_object_positions,
    true_object_rotations,
    true_object_scales,
    pred_object_positions,
    pred_object_rotations,
    pred_object_scales,
    point_clouds,  # List of N arrays, each Kx3, in camera-local space
):
    """
    Write a single frame's data to a JSON file in the new spec.

    Args:
        out_path: Path to write the JSON file.
        *_positions, *_rotations, *_scales: Lists/arrays as described in DATA_FORMAT.md.
        point_clouds: List of N arrays (camera-local point clouds).
    """
    data = {
        "true": {
            "camera": {
                "positions": true_camera_positions,
                "rotations": true_camera_rotations,
            },
            "objects": {
                "positions": true_object_positions,
                "rotations": true_object_rotations,
                "scales": true_object_scales,
            },
        },
        "pred": {
            "camera": {
                "positions": pred_camera_positions,
                "rotations": pred_camera_rotations,
            },
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
