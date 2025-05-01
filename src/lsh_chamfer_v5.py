"""
LSH-based Chamfer distance implementation (v5).
Uses hierarchical locality-sensitive hashing for efficient point cloud comparison.

Key ideas:
1. All operations are symmetric and handle both directions at once
2. Direct binary code comparison using einsum
3. Level-weighted sampling without intermediate reductions
4. Pure tensor operations, no loops or lists
"""

import time
import torch
from einops import rearrange


def lsh_chamfer_distance(
    points: torch.Tensor,  # [num_pairs, 2, num_points, xyz=3]
    masks: torch.Tensor,  # [num_pairs, 2, num_points]
    levels: int = 3,  # Number of LSH levels (coarse to fine)
    bits_per_level: int = 4,  # Number of bits per level
    num_samples: int = 20,  # Number of points to sample per bucket
    debug: bool = False,
) -> torch.Tensor:  # [num_pairs] Mean distance per pair
    """LSH-based Chamfer distance between pairs of point clouds.

    Args:
        points: Point cloud pairs [num_pairs, 2, num_points, xyz=3]
        masks: Valid point masks [num_pairs, 2, num_points]
        levels: Number of LSH levels (coarse to fine)
        bits_per_level: Number of bits per level
        num_samples: Number of points to sample per bucket
        debug: Whether to print timing info

    Returns:
        chamfer_dist: Mean distance per pair [num_pairs]
    """
    device = points.device
    num_pairs, _, num_points, _ = points.shape

    # Stage 1: Project points to binary codes
    # Input:  points [num_pairs, 2, num_points, xyz=3]
    # Output: binary_codes [num_pairs, 2, points, levels, bits]
    t0 = time.time()

    # Generate random unit vectors for all levels
    total_vectors = bits_per_level * (levels * (levels + 1)) // 2
    vectors = torch.randn(total_vectors, 3, device=device)
    vectors = vectors / vectors.norm(dim=-1, keepdim=True)

    # Project points onto vectors
    flat_points = rearrange(points, "p c n d -> (p c) n d")
    projs = torch.matmul(flat_points, vectors.T)

    # Convert to binary codes
    level_projs = rearrange(
        projs,
        "(p c) n (l h) -> p c n l h",
        p=num_pairs,
        c=2,
        l=levels,
    )
    binary_codes = level_projs > 0

    if debug:
        t1 = time.time()
        print(f"Stage 1 (Binary codes): {t1 - t0:.3f}s")

    # Stage 2: Match points and weight by level
    # Input:  binary_codes [num_pairs, 2, points, levels, bits]
    # Output: weighted [num_pairs, cloud_a, cloud_b, points_m, points_n, levels]
    t0 = time.time()

    # Compare all points with all points at once
    # True where codes match at each bit
    matches = torch.einsum(
        "panlb,pbmlb->pabmnl",
        binary_codes,  # [pairs, cloud_a, points_n, levels, bits]
        binary_codes,  # [pairs, cloud_b, points_m, levels, bits]
    )  # [pairs, cloud_a, cloud_b, points_m, points_n, levels]

    # Points must match all bits at a level
    matches = matches.all(
        dim=-1
    )  # [pairs, cloud_a, cloud_b, points_m, points_n, levels]

    # Weight matches by level priority
    # Level 0 (coarse): weight = 0.001 - rarely sample
    # Level 1 (medium): weight = 0.01  - sometimes sample
    # Level 2 (fine):   weight = 1.0   - prefer to sample
    level_weights = torch.tensor([0.001, 0.01, 1.0], device=device)[-levels:]
    weighted = matches * level_weights[None, None, None, None, None, :]

    # Mask invalid points from both clouds
    weighted = weighted * (
        masks[:, :, None, :, None, None]  # Mask points_m
        * masks[:, None, :, None, :, None]  # Mask points_n
    )

    if debug:
        t1 = time.time()
        print(f"Stage 2 (Match & weight): {t1 - t0:.3f}s")

    # Stage 3: Sample points and compute distances
    # Input:  weighted [num_pairs, cloud_a, cloud_b, points_m, points_n, levels]
    #         points [num_pairs, 2, points, xyz]
    # Output: chamfer_dist [num_pairs]
    t0 = time.time()

    # Sample points according to weighted matches
    # Normalize weights to probabilities
    probs = weighted / (
        weighted.sum(dim=-2, keepdim=True) + 1e-10
    )  # Normalize over target points

    # Reshape to combine all dimensions except target points
    # [pairs, cloud_a, cloud_b, points_m, points_n, levels] -> [(pairs*clouds*clouds*points*levels) points_n]
    flat_probs = probs.reshape(-1, num_points)

    # Sample target points
    sample_idx = torch.multinomial(
        flat_probs,
        num_samples=num_samples,
        replacement=True,
    )

    # Reshape back to separate dimensions
    sample_idx = sample_idx.reshape(num_pairs, 2, 2, num_points, -1, num_samples).sum(
        dim=-2
    )  # Sum over levels

    # Compute distances for both directions at once
    # Gather sampled points
    sampled_points = points[
        torch.arange(num_pairs, device=device)[:, None, None, None],  # [pairs, 1, 1, 1]
        torch.arange(2, device=device)[None, None, :, None],  # [1, 1, clouds, 1]
        sample_idx,  # [pairs, clouds, clouds, points, samples]
    ]  # [pairs, clouds, clouds, points, samples, xyz]

    # Compute distances between each point and its samples
    source_points = points[:, :, None, :, None, :]  # [pairs, clouds, 1, points, 1, xyz]
    dists = torch.norm(
        source_points - sampled_points, dim=-1
    )  # [pairs, clouds, clouds, points, samples]

    # Get minimum distance to any sample
    min_dists = dists.min(dim=-1)[0]  # [pairs, clouds, clouds, points]

    # Average over valid points
    masked_dists = min_dists * masks[:, :, None, :] * masks[:, None, :, None]
    mean_dists = masked_dists.sum(dim=-1) / (
        masks[:, :, None] * masks[:, None, :] + 1e-10
    )  # [pairs, clouds, clouds]

    # Average both directions
    chamfer_dist = mean_dists.mean(dim=(1, 2))  # [pairs]

    if debug:
        t1 = time.time()
        print(f"Stage 3 (Sample & compute): {t1 - t0:.3f}s")

    return chamfer_dist
