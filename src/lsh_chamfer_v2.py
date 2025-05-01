"""
LSH-based Chamfer distance implementation (v2).
Uses hierarchical locality-sensitive hashing for efficient point cloud comparison.

Important Design Principles:
1. All operations work on pairs of point clouds with shape [num_pairs, 2, ...].
   This makes the symmetry of Chamfer distance explicit and allows for vectorized
   operations across both clouds in each pair simultaneously.

2. Invalid points are only masked out at three specific points:
   - When counting points in buckets
   - When sampling points from buckets
   - When computing final mean distances

Algorithm Overview:
1. Hash points into buckets at multiple levels:
   - Generate random hyperplanes for each level
   - Project points onto hyperplanes to get binary codes
   - Convert binary codes to bucket assignments
   - Count points per bucket

2. Find points in same bucket:
   - For each point, count how many points from other cloud are in its bucket
   - Find finest level where bucket has points

3. Sample points from buckets:
   - Create distribution over valid target points
   - Sample target points using multinomial
   - Gather sampled points from original clouds

4. Compute distances:
   - Find closest sampled point for each point
   - Average over valid points in each cloud
   - Average both directions for final distance
"""

import time
import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple


def lsh_chamfer_distance(
    points: torch.Tensor,  # [num_pairs, 2, num_points, xyz=3]
    masks: torch.Tensor,  # [num_pairs, 2, num_points]
    levels: int = 3,  # Number of LSH levels (coarse to fine)
    bits_per_level: int = 4,  # Number of bits per level (2^bits buckets)
    num_samples: int = 20,  # Number of points to sample per bucket
    debug: bool = False,
) -> torch.Tensor:  # [num_pairs] Mean distance per pair
    """LSH-based Chamfer distance between pairs of point clouds.
    Uses hierarchical LSH to efficiently sample points for comparison.

    Args:
        points: Point cloud pairs [num_pairs, 2, num_points, xyz=3]
        masks: Valid point masks [num_pairs, 2, num_points]
        levels: Number of LSH levels (coarse to fine)
        bits_per_level: Number of bits per level (2^bits buckets)
        num_samples: Number of points to sample per bucket
        debug: Whether to print timing info

    Returns:
        chamfer_dist: Mean distance per pair [num_pairs]

    Implementation:
    1. Hash points into buckets at multiple levels:
       - Level 0: 4 planes -> 16 buckets (coarse matching)
       - Level 1: 8 planes -> 256 buckets (medium matching)
       - Level 2: 12 planes -> 4096 buckets (fine matching)
       Points that are close together will share buckets at multiple levels.

    2. Find points in same bucket:
       For each point, find the finest level where its bucket contains
       points from the other cloud. This gives adaptive granularity.

    3. Sample points from buckets:
       Create probability distribution over points in same bucket,
       then sample target points for comparison.

    4. Compute distances:
       Find closest sampled point for each point, average over valid
       points, then average both directions for final distance.
    """
    device = points.device
    num_pairs, _, num_points, _ = points.shape
    max_buckets = 2**bits_per_level

    # Stage 1: Hash all points
    # Input:  points [num_pairs, 2, num_points, xyz=3]
    #         masks [num_pairs, 2, num_points]
    # Output: bucket_onehot [num_pairs, 2, points, levels, buckets] One-hot bucket assignments
    #         bucket_counts [num_pairs, 2, levels, buckets] Points per bucket per cloud
    t0 = time.time()

    # Step 1.1: Generate hyperplanes
    # Need more planes at each level: [4, 8, 12] for bits_per_level=4
    # total_vectors = bits_per_level * (1 + 2 + 3) for 3 levels
    # Output: vectors [total_vectors xyz=3] - Unit vectors for projection
    total_vectors = bits_per_level * (levels * (levels + 1)) // 2
    vectors = torch.randn(total_vectors, 3, device=device)
    vectors = vectors / vectors.norm(dim=-1, keepdim=True)

    # Step 1.2: Project points onto hyperplanes
    # Input:  points [num_pairs, 2, points, xyz]
    # Output: projs [num_pairs, 2, points, total_vectors]
    flat_points = rearrange(points, "p c n d -> (p c) n d")
    projs = torch.matmul(flat_points, vectors.T)

    # Step 1.3: Convert to binary codes
    # Input:  projs [num_pairs, 2, points, total_vectors]
    # Output: binary [num_pairs, 2, points, levels, bits_per_level]
    level_projs = rearrange(
        projs,
        "(p c) n (l h) -> p c n l h",
        p=num_pairs,
        c=2,
        l=levels,
    )
    binary = level_projs > 0

    # Step 1.4: Convert to bucket assignments
    # Input:  binary [num_pairs, 2, points, levels, bits_per_level]
    # Output: bucket_onehot [num_pairs, 2, points, levels, buckets]
    bit_weights = 2 ** torch.arange(bits_per_level, device=device)
    bucket_onehot = F.one_hot(
        (binary * bit_weights).sum(dim=-1),
        num_classes=max_buckets,
    ).float()

    # Step 1.5: Count points per bucket
    # Input:  bucket_onehot [num_pairs, 2, points, levels, buckets]
    #         masks [num_pairs, 2, points]
    # Output: bucket_counts [num_pairs, 2, levels, buckets]
    bucket_counts = (bucket_onehot * rearrange(masks, "p c n -> p c n 1 1")).sum(dim=2)

    if debug:
        t1 = time.time()
        print(f"Stage 1 (Hash points): {t1 - t0:.3f}s")

    # Stage 2: Find points in same bucket
    # Input:  bucket_onehot [num_pairs, 2, points, levels, buckets]
    #         bucket_counts [num_pairs, 2, levels, buckets]
    # Output: points_in_bucket [num_pairs, 2, points, levels] Points in each point's bucket
    #         finest_level [num_pairs, 2, points] Finest level with points
    t0 = time.time()

    # Step 2.1: Count points in same bucket
    # Input:  bucket_onehot [num_pairs, 2, points, levels, buckets]
    #         bucket_counts [num_pairs, 2, levels, buckets]
    # Output: points_in_bucket [num_pairs, 2, points, levels]
    points_in_bucket = torch.einsum(
        "pcnlb,pclb->pcnl",
        bucket_onehot,
        torch.roll(bucket_counts, shifts=1, dims=1),
    )

    # Step 2.2: Find finest level with points
    # Input:  points_in_bucket [num_pairs, 2, points, levels]
    # Output: finest_level [num_pairs, 2, points]
    level_weights = torch.arange(levels, device=device)
    finest_level = ((points_in_bucket > 0) * level_weights).argmax(dim=-1)

    if debug:
        t1 = time.time()
        print(f"Stage 2 (Find buckets): {t1 - t0:.3f}s")

    # Stage 3: Sample points from buckets
    # Input:  bucket_onehot [num_pairs, 2, points, levels, buckets]
    #         points [num_pairs, 2, points, xyz]
    #         masks [num_pairs, 2, points]
    # Output: sampled_points [num_pairs, 2, points, samples, xyz]
    t0 = time.time()

    # Step 3.1: Create distribution over valid target points
    # Input:  bucket_onehot [num_pairs, 2, points, levels, buckets]
    # Output: probs [num_pairs, 2, points, clouds=2, points]
    same_bucket = torch.einsum(
        "pcnlb,pqmlb->pcnqm",
        bucket_onehot,
        rearrange(bucket_onehot, "p c n l b -> p c l n b"),
    )
    other_cloud = torch.roll(same_bucket, shifts=1, dims=3)
    valid_targets = other_cloud & rearrange(masks, "p c n -> p c n 1 1")
    probs = valid_targets.float()
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

    # Step 3.2: Sample target points
    # Input:  probs [num_pairs, 2, points, clouds=2, points]
    # Output: sample_idx [num_pairs, 2, points, samples]
    sample_idx = torch.multinomial(
        rearrange(probs, "p c n q m -> (p c n) m", p=num_pairs),
        num_samples=num_samples,
        replacement=True,
    ).reshape(num_pairs, 2, num_points, num_samples)

    # Step 3.3: Gather sampled points
    # Input:  points [num_pairs, 2, points, xyz]
    #         sample_idx [num_pairs, 2, points, samples]
    # Output: sampled_points [num_pairs, 2, points, samples, xyz]
    sampled_points = points[
        torch.arange(num_pairs, device=device)[:, None, None, None],
        torch.roll(torch.arange(2, device=device), shifts=1),
        sample_idx,
    ]

    if debug:
        t1 = time.time()
        print(f"Stage 3 (Sample points): {t1 - t0:.3f}s")

    # Stage 4: Compute distances
    # Input:  sampled_points [num_pairs, 2, points, samples, xyz]
    #         points [num_pairs, 2, points, xyz]
    #         masks [num_pairs, 2, points]
    # Output: chamfer_dist [num_pairs] Mean distance per pair
    t0 = time.time()

    # Step 4.1: Compute point-to-sample distances
    # Input:  points [num_pairs, 2, points, xyz]
    #         sampled_points [num_pairs, 2, points, samples, xyz]
    # Output: dists [num_pairs, 2, points, samples]
    expanded_points = rearrange(points, "p c n d -> p c n 1 d")
    dists = torch.norm(expanded_points - sampled_points, dim=-1)

    # Step 4.2: Find closest sample for each point
    # Input:  dists [num_pairs, 2, points, samples]
    # Output: min_dists [num_pairs, 2, points]
    min_dists = dists.min(dim=-1)[0]

    # Step 4.3: Average over valid points
    # Input:  min_dists [num_pairs, 2, points]
    #         masks [num_pairs, 2, points]
    # Output: mean_dists [num_pairs, 2]
    masked_dists = min_dists * masks
    mean_dists = masked_dists.sum(dim=-1) / (masks.sum(dim=-1) + 1e-6)

    # Step 4.4: Average both directions
    # Input:  mean_dists [num_pairs, 2]
    # Output: chamfer_dist [num_pairs]
    chamfer_dist = mean_dists.mean(dim=-1)

    if debug:
        t1 = time.time()
        print(f"Stage 4 (Compute distances): {t1 - t0:.3f}s")

    return chamfer_dist
