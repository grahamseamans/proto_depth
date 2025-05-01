"""
Pure Chamfer distance implementation using cdist.
Simple and memory-efficient due to sparse gradients from min operation.
"""

import time
import torch


def chamfer_distance(
    points: torch.Tensor,  # [num_pairs, 2, num_points, xyz=3]
    masks: torch.Tensor,  # [num_pairs, 2, num_points]
    squared: bool = True,  # Whether to return squared distances
    debug: bool = False,  # Keep debug for timing info
) -> torch.Tensor:  # [num_pairs] Mean distance per pair
    """Pure Chamfer distance using cdist.

    Computes exact Chamfer distance between pairs of point clouds
    using torch.cdist for all pairwise distances.

    Memory usage:
    - Temporary distance matrix: O(num_pairs * num_points * num_points)
    - Final storage: O(num_pairs * num_points) due to min operation
    - Gradients: O(num_pairs * num_points) since only min distances kept

    Example for 120 pairs of 1500-point clouds:
    - Distance matrix: ~1.08 GB
    - Final storage: ~720 KB
    - Total with overhead: ~2 GB

    Args:
        points: Point cloud pairs [num_pairs, 2, num_points, xyz=3]
        masks: Valid point masks [num_pairs, 2, num_points]
        squared: Whether to return squared distances (default: True)
        debug: Whether to print timing info

    Returns:
        chamfer_dist: Mean (squared) distance per pair [num_pairs]
    """
    t0 = time.time()

    # Compute all pairwise distances for each pair
    # For each pair:
    #   For each point in cloud 0:
    #     Distance to each point in cloud 1
    # Use squared distances by default for better gradients
    dists = torch.cdist(
        points[:, 0],  # [pairs, points, xyz]
        points[:, 1],  # [pairs, points, xyz],
    )

    if squared:
        dists = dists.pow(2)

    # Get minimum distance to any point in other cloud
    # Only keeps gradient for min point
    min_dists_0 = dists.min(dim=-1)[0]  # [pairs, points]
    min_dists_1 = dists.min(dim=-2)[0]  # [pairs, points]

    # Mask out invalid points
    masked_dists_0 = min_dists_0 * masks[:, 0]  # [pairs, points]
    masked_dists_1 = min_dists_1 * masks[:, 1]  # [pairs, points]

    # Average over valid points
    mean_dists_0 = masked_dists_0.sum(dim=-1) / (
        masks[:, 0].sum(dim=-1) + 1e-10
    )  # [pairs]
    mean_dists_1 = masked_dists_1.sum(dim=-1) / (
        masks[:, 1].sum(dim=-1) + 1e-10
    )  # [pairs]

    # Average both directions
    chamfer_dist = (mean_dists_0 + mean_dists_1) / 2  # [pairs]

    if debug:
        t1 = time.time()
        print(f"Chamfer distance: {t1 - t0:.3f}s")

    return chamfer_dist
