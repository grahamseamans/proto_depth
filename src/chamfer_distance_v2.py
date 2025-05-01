"""
Memory-efficient Chamfer distance implementation.
Uses smart batching to handle large point clouds without OOM errors.
"""

import time
import torch


def chamfer_distance(
    points: torch.Tensor,  # [batch, H*W, xyz=3] Full point clouds
    valid_idx: torch.Tensor,  # [batch, H*W] Boolean mask of valid points
    pair_idx: torch.Tensor,  # [num_pairs, 2] Which clouds to compare
    debug: bool = False,
) -> torch.Tensor:  # [num_pairs] Mean distance per pair
    """Memory-efficient Chamfer distance using smart batching.

    Processes pairs of point clouds in batches that fit in GPU memory.
    Pairs are sorted by max points to minimize padding waste.
    Results are accumulated and returned at the end.

    Args:
        points: Point clouds [batch, H*W, xyz=3]
        valid_idx: Boolean mask of valid points [batch, H*W]
        pair_idx: Indices of clouds to compare [num_pairs, 2]
        debug: Whether to print timing info

    Returns:
        chamfer_dist: Mean distance per pair [num_pairs]
    """
    device = points.device
    t0 = time.time()

    # Count valid points for each cloud
    valid_counts = valid_idx.sum(dim=1)  # [batch]

    # Get max points for each pair
    pair_max_counts = torch.maximum(
        valid_counts[pair_idx[:, 0]], valid_counts[pair_idx[:, 1]]
    )  # [num_pairs]

    # Sort pairs by max count (largest first)
    sorted_indices = pair_max_counts.argsort(descending=True)
    remaining_pairs = pair_idx[sorted_indices]

    # Get GPU memory info
    free_memory = torch.cuda.get_device_properties(device).total_memory
    free_memory -= torch.cuda.memory_allocated()
    bytes_per_float = 4
    memory_headroom = 0.8  # Leave 20% for overhead

    # Process pairs in memory-efficient batches
    all_dists = []
    while len(remaining_pairs) > 0:
        batch_pairs = []
        total_points = 0

        # Keep adding pairs until we're near capacity
        while len(remaining_pairs) > 0:
            next_pair = remaining_pairs[0]
            next_size = max(
                valid_counts[next_pair[0]].item(), valid_counts[next_pair[1]].item()
            )

            # Check if adding this pair would exceed memory
            # Each pair needs N×M matrix of floats
            next_memory = total_points * next_size * bytes_per_float
            if next_memory > free_memory * memory_headroom:
                break

            batch_pairs.append(next_pair)
            total_points += next_size
            remaining_pairs = remaining_pairs[1:]

            if debug:
                print(f"Added pair with {next_size} points, total {total_points}")

        if len(batch_pairs) == 0:
            raise RuntimeError(
                f"Cannot process pair with {next_size} points "
                f"(would need {next_memory / 1e9:.2f}GB, have {free_memory / 1e9:.2f}GB)"
            )

        # Convert to tensor
        batch_pairs = torch.stack(batch_pairs)

        # Process this batch
        batch_dists = []
        for idx_a, idx_b in batch_pairs:
            # Extract valid points
            points_a = points[idx_a][valid_idx[idx_a]]  # [N, 3]
            points_b = points[idx_b][valid_idx[idx_b]]  # [M, 3]

            # Compute distances
            dists = torch.cdist(points_a, points_b)  # [N, M]
            if debug:
                print(f"Distance matrix shape: {dists.shape}")

            # Get min distances in both directions
            min_ab = dists.min(dim=1)[0].mean()  # [1]
            min_ba = dists.min(dim=0)[0].mean()  # [1]

            # Average both directions
            batch_dists.append((min_ab + min_ba) / 2)

        # Stack batch results
        all_dists.append(torch.stack(batch_dists))

    # Combine all results
    all_dists = torch.cat(all_dists)  # [num_pairs]

    # Unsort to match input pair order
    chamfer_dist = torch.empty_like(all_dists)
    chamfer_dist[sorted_indices] = all_dists

    if debug:
        t1 = time.time()
        print(f"Chamfer distance: {t1 - t0:.3f}s")

    return chamfer_dist
