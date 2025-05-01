"""
Memory-efficient Chamfer distance implementation.
Uses einsum/einops for clarity and smart batching for memory efficiency.
Properly handles padding for variable-sized point clouds.
"""

import time
import torch
from einops import rearrange, reduce


def chamfer_distance(
    points: torch.Tensor,  # [pairs, clouds=2, points=60k, xyz=3]
    valid_idx: torch.Tensor,  # [pairs, clouds=2, points=60k]
    squared: bool = True,  # Whether to return squared distances
    debug: bool = False,
) -> torch.Tensor:  # [pairs] Mean distance per pair
    """Memory-efficient Chamfer distance using smart batching.

    Processes pairs of point clouds in batches that fit in GPU memory.
    Uses einsum/einops for clearer tensor operations.
    Results are accumulated and returned at the end.

    Args:
        points: Point clouds [pairs, clouds=2, points=60k, xyz=3]
        valid_idx: Boolean mask of valid points [pairs, clouds=2, points=60k]
        squared: Whether to return squared distances (default: True)
        debug: Whether to print timing info

    Returns:
        chamfer_dist: Mean distance per pair [pairs]
    """
    device = points.device
    t0 = time.time()

    # Count valid points per cloud
    valid_counts = reduce(valid_idx, "p c n -> p c", "sum")  # [pairs, clouds]
    max_points = reduce(valid_counts, "p c -> p", "max")  # [pairs]

    # Sort pairs by point count (largest first)
    sorted_indices = torch.argsort(max_points, descending=True)
    remaining_pairs = sorted_indices

    # Get GPU memory info
    free_memory = torch.cuda.get_device_properties(device).total_memory
    free_memory -= torch.cuda.memory_allocated()
    bytes_per_float = 4
    memory_headroom = 0.8  # Leave 20% for overhead

    # Process pairs in memory-efficient batches
    all_dists = []
    while len(remaining_pairs) > 0:
        first_count = max_points[remaining_pairs[0]].item()

        # Memory per batch:
        # - Two point clouds: 2 * points * 3 * 4 bytes
        # - Distance matrix: points * points * 4 bytes
        bytes_per_batch = (
            2 * first_count * 3 * bytes_per_float  # Point clouds
            + first_count * first_count * bytes_per_float  # Distance matrix
        )
        batch_size = int(memory_headroom * free_memory / bytes_per_batch)
        if batch_size == 0:
            raise RuntimeError(
                f"Cannot process pair with {first_count} points "
                f"(would need {bytes_per_batch / 1e9:.2f}GB, have {free_memory / 1e9:.2f}GB)"
            )
        batch_pairs = remaining_pairs[:batch_size]

        if debug:
            print(
                f"Processing batch of {batch_size} pairs with {first_count} points each"
            )

        # Create fixed-size tensors for this batch
        batch_points = torch.zeros(
            batch_size, 2, first_count, 3, device=device
        )  # [batch, clouds=2, points, xyz]
        batch_masks = torch.zeros(
            batch_size, 2, first_count, device=device, dtype=torch.bool
        )  # [batch, clouds=2, points]

        # Fill with valid points
        for b, pair_idx in enumerate(batch_pairs):
            # Cloud A
            valid_a = valid_idx[pair_idx, 0]  # [60k]
            points_a = points[pair_idx, 0][valid_a]  # [num_valid, xyz]
            batch_points[b, 0, : len(points_a)] = points_a
            batch_masks[b, 0, : len(points_a)] = True

            # Cloud B
            valid_b = valid_idx[pair_idx, 1]  # [60k]
            points_b = points[pair_idx, 1][valid_b]  # [num_valid, xyz]
            batch_points[b, 1, : len(points_b)] = points_b
            batch_masks[b, 1, : len(points_b)] = True

        # Compute pairwise distances (squared by default)
        # Force matrix multiplication mode since we have large point clouds
        power = 2 if squared else 1
        dists = torch.cdist(
            batch_points[:, 0],  # [batch, points, xyz]
            batch_points[:, 1],  # [batch, points, xyz]
            p=power,
            compute_mode="use_mm_for_euclid_dist",
        )  # [batch, points_a, points_b]

        # Mask invalid points (both padding and original invalids)
        # Add singleton dimensions for broadcasting
        mask_a = batch_masks[:, 0, :, None]  # [batch, points_a, 1]
        mask_b = batch_masks[:, 1, None, :]  # [batch, 1, points_b]
        masked_dists = dists * mask_a * mask_b  # [batch, points_a, points_b]

        # Get min distances in both directions
        # Replace 0s with inf so they're not selected by min
        inf_dists = torch.where(masked_dists > 0, masked_dists, float("inf"))
        min_ab = reduce(inf_dists, "b i j -> b i", "min")  # [batch, points_a]
        min_ba = reduce(inf_dists, "b i j -> b j", "min")  # [batch, points_b]

        # Average over valid points
        mean_ab = reduce(min_ab * batch_masks[:, 0], "b i -> b", "sum") / (
            reduce(batch_masks[:, 0], "b i -> b", "sum") + 1e-10
        )  # [batch]
        mean_ba = reduce(min_ba * batch_masks[:, 1], "b i -> b", "sum") / (
            reduce(batch_masks[:, 1], "b i -> b", "sum") + 1e-10
        )  # [batch]

        # Average both directions
        batch_dists = (mean_ab + mean_ba) / 2  # [batch]
        all_dists.append(batch_dists)

        # Remove processed pairs
        remaining_pairs = remaining_pairs[batch_size:]

    # Combine all results
    all_dists = torch.cat(all_dists)  # [pairs]

    # Unsort to match input pair order
    chamfer_dist = torch.empty_like(all_dists)
    chamfer_dist[sorted_indices] = all_dists

    if debug:
        t1 = time.time()
        print(f"Chamfer distance: {t1 - t0:.3f}s")

    return chamfer_dist
