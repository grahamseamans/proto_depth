"""
LSH-based Chamfer distance implementation (v4).
Uses hierarchical locality-sensitive hashing for efficient point cloud comparison.

Key ideas:
1. Keep everything in binary form - no conversion to integers
2. Direct comparison of binary codes between points using einsum
3. Level-weighted sampling:
   - Level 0 (coarse): weight = 0.001 - rarely sample
   - Level 1 (medium): weight = 0.01  - sometimes sample
   - Level 2 (fine):   weight = 1.0   - prefer to sample
4. Simple tensor operations, minimal code
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

    # Stage 1: Get binary codes
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

    # Stage 2: Find matching points
    # Input:  binary_codes [num_pairs, 2, points, levels, bits]
    # Output: sampling_probs [num_pairs, 2, points, points]
    t0 = time.time()

    # Compare binary codes between points
    # For each point, find points in other cloud with matching codes at each level
    # Then weight by level priority to prefer finer matches
    level_weights = torch.tensor([0.001, 0.01, 1.0], device=device)[
        -levels:
    ]  # [levels]

    # Create sampling probabilities for both directions
    sampling_probs = []
    for c in range(2):
        other_c = 1 - c
        # Compare codes using einsum
        # True where codes match at each bit
        matches = torch.einsum(
            "pnlb,pmlb->pnml",
            binary_codes[:, c],  # [pairs, points1, levels, bits]
            binary_codes[:, other_c],  # [pairs, points2, levels, bits]
        )  # [pairs, points1, points2, levels]

        # Points must match all bits at a level
        matches = matches.all(dim=-1)  # [pairs, points1, points2, levels]

        # Weight matches by level and mask invalid points
        probs = matches * level_weights[None, None, None, :]  # Weight by level
        probs = probs.sum(dim=-1)  # Sum over levels
        probs = (
            probs * masks[:, c, :, None] * masks[:, other_c, None, :]
        )  # Mask invalids
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)  # Normalize
        sampling_probs.append(probs)

    if debug:
        t1 = time.time()
        print(f"Stage 2 (Find matches): {t1 - t0:.3f}s")

    # Stage 3: Sample points and compute distances
    # Input:  sampling_probs [num_pairs, 2, points, points]
    #         points [num_pairs, 2, points, xyz]
    # Output: chamfer_dist [num_pairs]
    t0 = time.time()

    # Sample target points
    sample_idx = []
    for c in range(2):
        # Sample points according to probabilities
        idx = torch.multinomial(
            sampling_probs[c].reshape(-1, num_points),  # [(pairs*points) points]
            num_samples=num_samples,
            replacement=True,
        ).reshape(num_pairs, num_points, num_samples)  # [pairs, points, samples]
        sample_idx.append(idx)

    # Compute distances for both directions
    mean_dists = []
    for c in range(2):
        other_c = 1 - c
        # Get sampled points
        sampled_points = points[
            torch.arange(num_pairs, device=device)[:, None, None],  # [pairs, 1, 1]
            other_c,  # Get points from other cloud
            sample_idx[c],  # [pairs, points, samples]
        ]  # [pairs, points, samples, xyz]

        # Compute distances
        source_points = points[:, c, :, None, :]  # [pairs, points, 1, xyz]
        dists = torch.norm(
            source_points - sampled_points, dim=-1
        )  # [pairs, points, samples]

        # Get minimum distance to any sample
        min_dists = dists.min(dim=-1)[0]  # [pairs, points]

        # Average over valid points
        masked_dists = min_dists * masks[:, c]  # [pairs, points]
        mean_dist = masked_dists.sum(dim=-1) / (
            masks[:, c].sum(dim=-1) + 1e-10
        )  # [pairs]
        mean_dists.append(mean_dist)

    # Average both directions
    chamfer_dist = (mean_dists[0] + mean_dists[1]) / 2  # [pairs]

    if debug:
        t1 = time.time()
        print(f"Stage 3 (Sample & compute): {t1 - t0:.3f}s")

    return chamfer_dist
