"""
Memory-efficient Chamfer distance implementation.
Uses smart batching and index-based point selection without padding.
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
    Uses index-based point selection to avoid padding.
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

        # Create indices for each pair's valid points
        batch_idx = torch.arange(len(batch_pairs), device=device)[:, None]  # [B, 1]
        valid_a = valid_idx[batch_pairs[:, 0]]  # [B, H*W]
        valid_b = valid_idx[batch_pairs[:, 1]]  # [B, H*W]

        # Get points directly using indexing
        points_a = points[batch_pairs[:, 0]][valid_a]  # [total_valid_a, 3]
        points_b = points[batch_pairs[:, 1]][valid_b]  # [total_valid_b, 3]

        # Keep track of which points belong to which pair
        pair_idx_a = batch_idx.expand(-1, points.shape[1])[valid_a]  # [total_valid_a]
        pair_idx_b = batch_idx.expand(-1, points.shape[1])[valid_b]  # [total_valid_b]

        # Compute distances between all valid points
        dists = torch.cdist(points_a, points_b)  # [total_valid_a, total_valid_b]

        # Get min distances for each point
        min_ab = dists.min(dim=1)[0]  # [total_valid_a]
        min_ba = dists.min(dim=0)[0]  # [total_valid_b]

        # Average per pair
        mean_ab = torch.zeros(len(batch_pairs), device=device)
        mean_ab.index_add_(0, pair_idx_a, min_ab)
        mean_ab /= valid_a.sum(dim=1)  # Divide by number of points in each cloud

        mean_ba = torch.zeros(len(batch_pairs), device=device)
        mean_ba.index_add_(0, pair_idx_b, min_ba)
        mean_ba /= valid_b.sum(dim=1)

        # Average both directions
        batch_dists = (mean_ab + mean_ba) / 2
        all_dists.append(batch_dists)

    # Combine all results
    all_dists = torch.cat(all_dists)  # [num_pairs]

    # Unsort to match input pair order
    chamfer_dist = torch.empty_like(all_dists)
    chamfer_dist[sorted_indices] = all_dists

    if debug:
        t1 = time.time()
        print(f"Chamfer distance: {t1 - t0:.3f}s")

    return chamfer_dist
