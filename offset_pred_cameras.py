#!/usr/bin/env python3
"""
Script to add a Y offset to predicted camera positions in frame JSON files.
This helps visualize the difference between true and predicted cameras by preventing overlap.
"""

import json
import glob
from pathlib import Path
import sys

# Amount to offset pred cameras up in Y direction
OFFSET_Y = 0.1


def offset_pred_cameras(run_dir):
    """Add Y offset to all pred camera positions in a run directory."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist")
        return False

    # Find all frame JSON files
    frame_files = list(run_dir.glob("**/frame_*.json"))
    if not frame_files:
        print(f"Error: No frame files found in {run_dir}")
        return False

    print(f"Found {len(frame_files)} frame files")
    modified = 0

    for frame_path in frame_files:
        try:
            # Load frame data
            with open(frame_path) as f:
                data = json.load(f)

            # Add offset to pred camera positions
            if (
                "pred" in data
                and "camera" in data["pred"]
                and "positions" in data["pred"]["camera"]
            ):
                for pos in data["pred"]["camera"]["positions"]:
                    pos[1] += OFFSET_Y  # Add to Y coordinate
                modified += 1

                # Save modified data
                with open(frame_path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"Modified {frame_path.relative_to(run_dir)}")
            else:
                print(
                    f"Skipping {frame_path.relative_to(run_dir)} - missing pred camera data"
                )

        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            continue

    print(f"\nDone! Modified {modified} files")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python offset_pred_cameras.py <run_directory>")
        print("Example: python offset_pred_cameras.py viz_server/data/run_1744492226")
        sys.exit(1)

    run_dir = sys.argv[1]
    success = offset_pred_cameras(run_dir)
    sys.exit(0 if success else 1)
