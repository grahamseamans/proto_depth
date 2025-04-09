"""
Minimal dataloader for energy-based optimization

This standalone module provides basic functionality for loading and processing
depth images without extra dependencies.
"""

import numpy as np
import torch
from PIL import Image

# Constants
FOCAL = 847.630211643
MAX_DEPTH = 1000.0
CUTOFF_DISTANCE = 300.0
STANDARD_SIZE = (640, 480)  # (width, height)


def load_depth_image(depth_path):
    """Load a depth image from a PNG file."""
    im = Image.open(depth_path)
    im = np.array(im, dtype=np.int32)
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]
    depth_int24 = R + G * 256 + B * (256**2)
    depth = depth_int24.astype(np.float64) / ((256**3) - 1) * 1000.0  # meters
    return depth


def normalize_depth(depth):
    """Normalize depth using sqrt scaling."""
    ratio = depth / CUTOFF_DISTANCE
    normalized = np.sqrt(np.clip(ratio, 0, 1))
    return normalized


def depth_to_pointcloud(depth, focal=FOCAL, max_depth=MAX_DEPTH):
    """Convert a depth image to a point cloud."""
    H, W = depth.shape
    cx = W / 2.0
    cy = H / 2.0
    fx = focal
    fy = focal

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Convert to Three.js coordinate system:
    # - Z is negated (Three.js uses -Z as forward)
    Z = -depth  # Negate Z to match Three.js -Z forward convention
    X = -(u - cx) / fx * Z
    Y = (v - cy) / fy * Z

    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    valid_mask = points[:, 2] > -max_depth
    points = points[valid_mask]
    return points


def stratified_depth_sampling(points, bins=5, samples_per_bin=2000):
    """
    Sample points from different depth bins to ensure even coverage.

    Note: This function is no longer used in process_depth_image as we're now using
    the full resolution point cloud with all valid points for maximum detail.
    """
    z_vals = points[:, 2]
    min_z, max_z = z_vals.min(), z_vals.max()
    bin_edges = np.linspace(min_z, max_z, bins + 1)
    sampled_points = []

    for i in range(bins):
        bin_mask = (z_vals >= bin_edges[i]) & (z_vals < bin_edges[i + 1])
        bin_points = points[bin_mask]
        if len(bin_points) == 0:
            continue

        if len(bin_points) > samples_per_bin:
            idxs = np.random.choice(len(bin_points), samples_per_bin, replace=False)
            bin_points = bin_points[idxs]
        sampled_points.append(bin_points)

    if len(sampled_points) > 0:
        sampled_points = np.concatenate(sampled_points, axis=0)
    else:
        sampled_points = points

    return sampled_points


def process_depth_image(depth_path):
    """
    Process a depth image and convert it to a point cloud.

    Args:
        depth_path: Path to depth image

    Returns:
        depth_img_tensor: Tensor of shape [3, H, W] - normalized depth image
        point_cloud: Tensor of shape [N, 3] - point cloud
        original_depth: Tensor of shape [1, H, W] - original depth in meters
    """
    # Load the depth image
    depth = load_depth_image(depth_path)

    # Generate point cloud from original size depth - FULL RESOLUTION (no downsampling)
    points = depth_to_pointcloud(depth)

    # Convert to tensor - using all valid points (no stratified sampling)
    point_cloud = torch.from_numpy(points.astype(np.float32))  # (N,3)

    # Print the actual point cloud size
    print(f"Using full resolution point cloud with {len(points)} points")

    # Normalize the depth
    depth_normalized = normalize_depth(depth)

    # Convert to tensor
    depth_img = torch.from_numpy(depth_normalized[None].astype(np.float32))  # (1,H,W)

    # Also keep the original depth in meters for visualization
    depth_meters = torch.from_numpy(depth[None].astype(np.float32))  # (1,H,W)

    # Expand to 3 channels
    depth_img_3ch = depth_img.expand(3, -1, -1)  # (3,H,W)

    return depth_img_3ch, point_cloud, depth_meters
