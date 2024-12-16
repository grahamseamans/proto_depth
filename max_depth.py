import os
import glob
import numpy as np
from PIL import Image


def load_depth(depth_path):
    """
    Load depth from a SYNTHIA 24-bit depth png and return a float array of shape (H,W) in meters.
    depth = ((R + G*256 + B*65536) / (256^3 - 1)) * 1000
    """
    im = Image.open(depth_path)
    im = np.array(im, dtype=np.int32)

    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    depth_int24 = R + G * 256 + B * (256 * 256)
    depth = depth_int24.astype(np.float64) / ((256**3) - 1) * 1000.0  # meters
    return depth


def compute_global_max_depth(root_dir, sequences=[1, 2, 3, 4, 5, 6]):
    depth_files = []
    for seq in sequences:
        seq_dir = os.path.join(root_dir, f"SEQ{seq}", "DepthLeft")
        depth_files += glob.glob(os.path.join(seq_dir, "*.png"))

    if len(depth_files) == 0:
        raise RuntimeError(
            "No depth files found. Check directory and sequence numbers."
        )

    max_depth = 0.0
    for i, dfile in enumerate(depth_files):
        depth = load_depth(dfile)
        m = depth.max()
        if m > max_depth:
            max_depth = m
        # Print progress occasionally
        if (i + 1) % 100 == 0:
            print(
                f"Processed {i+1}/{len(depth_files)} images. Current max depth: {max_depth:.2f} m"
            )

    return max_depth


if __name__ == "__main__":
    data_dir = "/Users/cibo/code/proto_depth/data/SYNTHIA-SF"
    max_depth = compute_global_max_depth(data_dir, sequences=[1, 2, 3, 4, 5, 6])
    print(f"Global maximum depth across all sequences: {max_depth:.2f} meters")

    # You can now store this max_depth in a file or hardcode it elsewhere for faster dataset initialization.
    # For example:
    # with open("max_depth.txt", "w") as f:
    #     f.write(str(max_depth))
