"""
LSH-based Chamfer distance implementation (v3).
Uses hierarchical locality-sensitive hashing for efficient point cloud comparison.

Important Design Principles:
1. All operations work on pairs of point clouds with shape [num_pairs, 2, ...].
   This makes the symmetry of Chamfer distance explicit and allows for vectorized
   operations across both clouds in each pair simultaneously.

2. Keep binary LSH codes throughout computation:
   - No conversion to integers or one-hot vectors
   - Direct comparison of binary codes for bucket matching
   - More memory efficient, especially for fine-grained levels

3. Invalid points are only masked out at three specific points:
   - When counting points in buckets
   - When sampling points from buckets
   - When computing final mean distances
"""

import time
import torch
from einops import rearrange
from typing import Tuple


def lsh_chamfer_distance(
    points: torch.Tensor,  # [num_pairs, 2, num_points, xyz=3]
    masks: torch.Tensor,  # [num_pairs, 2, num_points]
    levels: int = 3,  # Number of LSH levels (coarse to fine)
    bits_per_level: int = 4,  # Number of bits per level
    num_samples: int = 20,  # Number of points to sample per bucket
    debug: bool = False,
) -> torch.Tensor:  # [num_pairs] Mean distance per pair
    """LSH-based Chamfer distance between pairs of point clouds.
    Uses hierarchical LSH to efficiently sample points for comparison.

    Args:
        points: Point cloud pairs [num_pairs, 2, num_points, xyz=3]
        masks: Valid point masks [num_pairs, 2, num_points]
        levels: Number of LSH levels (coarse to fine)
        bits_per_level: Number of bits per level
        num_samples: Number of points to sample per bucket
        debug: Whether to print timing info

    Returns:
        chamfer_dist: Mean distance per pair [num_pairs]

    Implementation:
    1. Hash points into buckets at multiple levels:
       - Level 0: 4 bits for coarse matching
       - Level 1: 8 bits for medium matching
       - Level 2: 12 bits for fine matching
       Points that are close together will share binary codes.

    2. Find points in same bucket:
       For each point, find the finest level where its binary code
       matches points from the other cloud.

    3. Sample points from buckets:
       Create probability distribution over points with matching codes,
       then sample target points for comparison.

    4. Compute distances:
       Find closest sampled point for each point, average over valid
       points, then average both directions for final distance.
    """
    device = points.device
    num_pairs, _, num_points, _ = points.shape

    # Stage 1: Hash all points
    # Input:  points [num_pairs, 2, num_points, xyz=3]
    #         masks [num_pairs, 2, num_points]
    # Output: binary_codes [num_pairs, 2, points, levels, bits_per_level]
    #         bucket_counts [num_pairs, 2, levels, 2^bits] Points with each code
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
    # Output: binary_codes [num_pairs, 2, points, levels, bits_per_level]
    level_projs = rearrange(
        projs,
        "(p c) n (l h) -> p c n l h",
        p=num_pairs,
        c=2,
        l=levels,
    )
    binary_codes = level_projs > 0

    # Step 1.4: Count points with each code
    # Input:  binary_codes [num_pairs, 2, points, levels, bits]
    #         masks [num_pairs, 2, points]
    # Output: bucket_counts [num_pairs, 2, levels, 2^bits]
    # For each level, count points that share the same binary code
    bucket_counts = []
    for l in range(levels):
        # Get codes for this level
        level_codes = binary_codes[..., l, :]  # [pairs, clouds, points, bits]

        # Convert to integers for unique counting
        bit_weights = 2 ** torch.arange(bits_per_level, device=device)
        code_ints = (level_codes * bit_weights).sum(dim=-1)  # [pairs, clouds, points]

        # Count occurrences of each code
        max_codes = 2**bits_per_level
        level_counts = torch.zeros(
            num_pairs, 2, max_codes, device=device
        )  # [pairs, clouds, codes]

        # Only count valid points
        for p in range(num_pairs):
            for c in range(2):
                valid_codes = code_ints[p, c][masks[p, c]]
                unique, counts = torch.unique(valid_codes, return_counts=True)
                level_counts[p, c, unique] = counts

        bucket_counts.append(level_counts)

    bucket_counts = torch.stack(bucket_counts, dim=2)  # [pairs, clouds, levels, codes]

    if debug:
        t1 = time.time()
        print(f"Stage 1 (Hash points): {t1 - t0:.3f}s")

    # Stage 2: Find points in same bucket
    # Input:  binary_codes [num_pairs, 2, points, levels, bits]
    #         bucket_counts [num_pairs, 2, levels, codes]
    # Output: points_in_bucket [num_pairs, 2, points, levels]
    #         finest_level [num_pairs, 2, points]
    t0 = time.time()

    # Step 2.1: Count points with matching codes
    # For each point in cloud 0, count points in cloud 1 with same code
    # For each point in cloud 1, count points in cloud 0 with same code
    points_in_bucket = []
    for l in range(levels):
        # Get codes for this level
        level_codes = binary_codes[..., l, :]  # [pairs, clouds, points, bits]

        # Convert to integers for comparison
        bit_weights = 2 ** torch.arange(bits_per_level, device=device)
        code_ints = (level_codes * bit_weights).sum(dim=-1)  # [pairs, clouds, points]

        # For each point, look up how many points from other cloud have same code
        level_counts = []
        for c in range(2):
            other_c = 1 - c
            # Get counts from other cloud
            other_counts = bucket_counts[:, other_c, l]  # [pairs, codes]
            # Look up counts for each point's code
            point_counts = other_counts[
                torch.arange(num_pairs, device=device)[:, None],
                code_ints[:, c],
            ]  # [pairs, points]
            level_counts.append(point_counts)

        level_counts = torch.stack(level_counts, dim=1)  # [pairs, clouds, points]
        points_in_bucket.append(level_counts)

    points_in_bucket = torch.stack(
        points_in_bucket, dim=3
    )  # [pairs, clouds, points, levels]

    # Step 2.2: Find finest level with points
    # Input:  points_in_bucket [num_pairs, 2, points, levels]
    # Output: finest_level [num_pairs, 2, points]
    level_weights = torch.arange(levels, device=device)
    finest_level = ((points_in_bucket > 0) * level_weights).argmax(dim=-1)

    if debug:
        t1 = time.time()
        print(f"Stage 2 (Find buckets): {t1 - t0:.3f}s")

    # Stage 3: Sample points from buckets
    # Input:  binary_codes [num_pairs, 2, points, levels, bits]
    #         points [num_pairs, 2, points, xyz]
    #         masks [num_pairs, 2, points]
    # Output: sampled_points [num_pairs, 2, points, samples, xyz]
    t0 = time.time()

    # Step 3.1: Create distribution over valid target points
    # For each point, find points from other cloud with matching code
    # at its finest valid level
    probs = torch.zeros(
        num_pairs, 2, num_points, num_points, device=device
    )  # [pairs, clouds, points, points]

    for p in range(num_pairs):
        for c in range(2):
            other_c = 1 - c
            for n in range(num_points):
                if masks[p, c, n]:  # Only process valid points
                    # Get point's code at its finest level
                    level = finest_level[p, c, n]
                    code = binary_codes[p, c, n, level]  # [bits]

                    # Find points in other cloud with matching code
                    other_codes = binary_codes[p, other_c, :, level]  # [points, bits]
                    matches = (other_codes == code).all(dim=-1)

                    # Only include valid points
                    matches = matches & masks[p, other_c]

                    # Convert to probabilities
                    if matches.any():
                        probs[p, c, n, matches] = 1.0 / matches.sum()

    # Step 3.2: Sample target points
    # Input:  probs [num_pairs, 2, points, points]
    # Output: sample_idx [num_pairs, 2, points, samples]
    sample_idx = torch.multinomial(
        probs.reshape(-1, num_points),  # [(pairs*clouds*points) points]
        num_samples=num_samples,
        replacement=True,
    ).reshape(num_pairs, 2, num_points, num_samples)

    # Step 3.3: Gather sampled points
    # Input:  points [num_pairs, 2, points, xyz]
    #         sample_idx [num_pairs, 2, points, samples]
    # Output: sampled_points [num_pairs, 2, points, samples, xyz]
    sampled_points = points[
        torch.arange(num_pairs, device=device)[:, None, None, None],
        1 - torch.arange(2, device=device),  # Get points from other cloud
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
