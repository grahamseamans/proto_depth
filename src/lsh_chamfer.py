"""
LSH-based Chamfer distance implementation.
Uses hierarchical locality-sensitive hashing for efficient point cloud comparison.

Important Design Principle:
For every point cloud in a batch, we assume the same number of points (i.e. the num_points
dimension is always the same). Invalid points are only masked out at three specific points:
1. When counting points in buckets
2. When sampling points from buckets
3. When computing final mean distances

Overview:
### Overview
We’re working with point clouds `A` and `B` (shape `[batch_size, num_points, dims]`), using hierarchical LSH to bucket points. For each `a` in `A`, we find the lowest non-empty bucket (with `B` points), sample `e` `B` points with replacement, compute Euclidean distances, take the min distance per `a`, and mean over `A` for the loss, backpropping only through the min. All batched, no loops.

### 1. Hashing
- **What**: Hyperplane LSH (random projection) to bucket `A` and `B` across `levels` (e.g., 3) from coarse to fine.
- **How**:
  - For each level `l` (0 to `levels-1`):
    - Generate `bits_per_level * (l + 1)` random unit vectors (e.g., `bits_per_level=4`, so 4, 8, 12 hyperplanes).
    - Project points: `projs = points @ vectors.T` (`[batch_size, num_points, num_projs]`).
    - Binarize: `binary = (projs > 0).float()`.
    - Bucket IDs: Combine first `bits_per_level` bits with powers `[1, 2, 4, 8]` to get `[batch_size, num_points]` integers.
  - Precompute counts: Scatter ones to `bucket_counts` (`[batch_size, max_buckets, levels]`, where `max_buckets = 2^bits_per_level`) for `B` points per bucket.
- **Output**:
  - `hash_indices`: `[batch_size, num_points, levels]` (bucket ID per point per level).
  - `bucket_counts`: `[batch_size, max_buckets, levels]` (count of `B` points per bucket).
- **Vibe**: Fast matrix ops, scatters. GPU loves it.

### 2. Lowest Non-Empty Level
- **What**: For each `a` in `A`, find the lowest level `h` where its bucket has at least one `B` point.
- **How**:
  - Flatten `A`: `A = batch_size * num_points`, `src_hash` to `[A, levels]`.
  - Get batch indices: `batch_idx = [0, 0, ..., 1, 1, ...]` (`[A]`).
  - Check non-empty: `non_empty = bucket_counts[batch_idx, src_hash, level] > 0` (`[A, levels]`).
  - Lowest level: `finest_level = argmax(non_empty * arange(levels))` (`[A]`).
  - Bucket ID: `bucket_ids = src_hash[arange(A), finest_level]` (`[A]`).
- **Output**:
  - `finest_level`: `[A]` (lowest level with `B` points).
  - `bucket_ids`: `[A]` (bucket ID at that level).
- **Vibe**: Quick indexing, `argmax`. GPU’s happy.

### 3. Sampling
- **What**: Sample `e` target points with replacement from each source point's bucket.
- **How**:
  - Create binary mask of valid targets in same bucket: `valid_targets[b,n,m]` is 1 if target point m is in source point n's bucket in batch b
  - Convert to probability distribution by normalizing over target points
  - Use multinomial sampling to pick e target points for each source point
  - Advanced indexing to gather sampled points: `target[batch_idx, sample_idx]`
- **Output**: `sampled_points`: `[batch, num_source, num_samples, xyz]`
- **Vibe**: Clean tensor ops, no loops, proper broadcasting. GPU purrs.

### 4. Distance Calculation
- **What**: Compute min distance per `a` to its `e` sampled `B` points, mean over `A` for loss.
- **How**:
  - Tensors:
    - `A` points: Flatten `source` to `[A, dims]`.
    - `B` points: Gather `target` with `sampled_indices` to `[A, e, dims]`.
  - Distances: `dists = norm(source.unsqueeze(1) - target, dim=-1)` (`[A, e]`).
  - Min: `min_dists = dists.min(dim=-1)[0]` (`[A]`).
  - Loss: `loss = min_dists.mean()` (scalar).
- **Backprop**: Gradients flow only through min distances.
- **Vibe**: Broadcasting for distances, `min` and `mean` are fast reductions. GPU-optimized.

### Vibe Check
- **Vectorized?** Fully—matrix ops, scatters, indexing, reductions. No batch/point loops.
- **Fast?** Blazing. Hashing’s quick, sampling’s O(1), distances are batched.
- **GPU?** All CUDA-friendly, scales with `A` and `e`.
- **Bucket misses?** Random hashes per loss call smooth it out, as you said.
- **Params**: Suggest `levels=3`, `bits_per_level=4`, `e=10–50`.

This is lucid and ready to rock. You cool with this flow? Wanna pick `e` or any other tweaks?
"""

import time
import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple


def lsh_chamfer_distance(
    source: torch.Tensor,  # [batch_size, num_source_points, xyz=3]
    target: torch.Tensor,  # [batch_size, num_target_points, xyz=3]
    source_mask: torch.Tensor,  # [batch_size, num_source_points]
    target_mask: torch.Tensor,  # [batch_size, num_target_points]
    levels: int = 3,
    bits_per_level: int = 4,
    num_samples: int = 20,
    debug: bool = False,
) -> torch.Tensor:  # [batch_size] Mean distance per batch
    """LSH-based Chamfer distance between batched point clouds.
    Uses hierarchical LSH to efficiently sample points for comparison.
    """
    t_start = time.time()

    # Stage 1: Hash all points together
    t0 = time.time()
    # Concatenate source and target points
    points = torch.cat([source, target], dim=0)  # [2*B, N, xyz=3]
    points_mask = torch.cat([source_mask, target_mask], dim=0)  # [2*B, N]

    # Hash all points in the same coordinate system
    hash_indices, bucket_counts = hash_points(
        points, points_mask, levels, bits_per_level, debug
    )

    # Split back into source and target hashes
    src_hash, tgt_hash = hash_indices.chunk(2, dim=0)  # Each [B, N, L]
    bucket_counts_src, bucket_counts_tgt = bucket_counts.chunk(
        2, dim=0
    )  # Each [B, 2^b, L]
    t1 = time.time()
    if debug:
        print(f"Stage 1 (Hash points): {t1 - t0:.3f}s")

    # Stage 2: Compute distances in both directions
    t0 = time.time()
    # Input:  source [B, N, xyz=3], target [B, M, xyz=3]
    #         source_mask [B, N], target_mask [B, M]
    #         src_hash [B, N, L], tgt_hash [B, M, L]
    #         bucket_counts [B, 2^b, L]
    # Output: chamfer_dist [B] Mean distance per batch
    src_to_tgt = chamfer_one_direction(
        source,
        target,
        source_mask,
        target_mask,
        src_hash,
        tgt_hash,
        bucket_counts,
        num_samples,
        debug,
    )
    tgt_to_src = chamfer_one_direction(
        target,
        source,
        target_mask,
        source_mask,
        tgt_hash,
        src_hash,
        bucket_counts,
        num_samples,
        debug,
    )
    t1 = time.time()
    if debug:
        print(f"Stage 2 (Chamfer distances): {t1 - t0:.3f}s")

    # Stage 3: Average both directions
    t0 = time.time()
    # Input:  src_to_tgt [B], tgt_to_src [B]
    # Output: chamfer_dist [B]
    chamfer_dist = (src_to_tgt + tgt_to_src) / 2
    t1 = time.time()
    if debug:
        print(f"Stage 3 (Average): {t1 - t0:.3f}s")
        print(f"Total time: {t1 - t_start:.3f}s")

    return chamfer_dist


def hash_points(
    points: torch.Tensor,  # [batch num_points xyz=3]
    points_mask: torch.Tensor,  # [batch num_points]
    levels: int,  # scalar
    bits_per_level: int,  # scalar
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hash point clouds using hierarchical LSH.
    Uses random hyperplane projections to bucket points across multiple levels.
    Only counts valid points in buckets based on masks.

    Each level uses more hyperplanes than the previous level:
    - Level 0: bits_per_level planes (e.g. 4) -> 16 buckets (coarse)
    - Level 1: 2*bits_per_level planes (e.g. 8) -> 256 buckets (medium)
    - Level 2: 3*bits_per_level planes (e.g. 12) -> 4096 buckets (fine)

    This creates a hierarchical structure:
    - Level 0: Big buckets, many points per bucket (coarse matching)
    - Level 1: Medium buckets, fewer points per bucket (medium matching)
    - Level 2: Small buckets, very few points per bucket (fine matching)

    Points that are close together will share buckets at multiple levels,
    while distant points will only share buckets at coarse levels.

    Args:
        points: Point cloud tensor [batch num_points xyz=3]
        points_mask: Binary mask for valid points [batch num_points]
        levels: Number of LSH levels (coarse to fine)
        bits_per_level: Number of bits used per level
        debug: Whether to print timing info

    Returns:
        hash_indices: Bucket ID per point per level [batch num_points levels]
        bucket_counts: Count of points per bucket per level [batch buckets levels]
    """
    device = points.device
    batch_size, num_points, _ = points.shape
    max_buckets = 2**bits_per_level  # Number of possible buckets per level

    # Generate random unit vectors for all levels at once
    # Level 0: bits_per_level vectors
    # Level 1: 2*bits_per_level vectors
    # Level 2: 3*bits_per_level vectors
    # Total vectors = bits_per_level * (1 + 2 + 3) = bits_per_level * (levels * (levels + 1)) // 2
    total_vectors = bits_per_level * (levels * (levels + 1)) // 2  # scalar
    vectors = torch.randn(total_vectors, 3, device=device)  # [total_vectors xyz=3]
    vectors = vectors / vectors.norm(dim=-1, keepdim=True)  # [total_vectors xyz=3]

    # Project all points onto all vectors at once
    # points: [batch num_points xyz=3]
    # vectors: [total_vectors xyz=3]
    # -> projs: [batch num_points total_vectors]
    projs = torch.einsum("b n d, v d -> b n v", points, vectors)

    # Rearrange projections to group by level
    # projs: [batch num_points (levels * bits_per_level)]
    # -> level_projs: [batch num_points levels bits_per_level]
    # This splits the vectors dimension into (levels, bits_per_level)
    # So each level gets its own set of bits_per_level projections
    level_projs = rearrange(projs, "b n (l h) -> b n l h", l=levels)

    # Convert projections to binary codes
    # level_projs: [batch num_points levels bits_per_level]
    # -> binary: [batch num_points levels bits_per_level]
    # 1 if point is on positive side of hyperplane, 0 if negative
    binary = level_projs > 0

    # Convert binary codes to bucket indices
    # binary: [batch num_points levels bits_per_level]
    # bit_weights: [bits_per_level] = [1, 2, 4, 8, ...]
    # -> indices: [batch num_points levels]
    # Each index is a number from 0 to 2^bits_per_level-1
    bit_weights = 2 ** torch.arange(bits_per_level, device=device)
    indices = (binary * bit_weights).sum(dim=-1)  # Combine bits into bucket index

    # Convert indices to one-hot bucket assignments
    bucket_onehot = F.one_hot(
        indices, num_classes=max_buckets
    )  # [batch num_points levels buckets]

    # Count valid points per bucket
    # 1. Expand mask to match bucket_onehot: [batch num_points] -> [batch num_points levels 1]
    # 2. Multiply with one-hot: [batch num_points levels buckets]
    # 3. Sum over points dimension: [batch buckets levels]
    points_mask = points_mask.float()[..., None, None]  # [batch num_points 1 1]
    bucket_counts = (bucket_onehot.float() * points_mask).sum(
        dim=1
    )  # [batch buckets levels]

    return indices, bucket_counts


def chamfer_one_direction(
    source: torch.Tensor,  # [batch_size, num_points, xyz=3]
    target: torch.Tensor,  # [batch_size, num_points, xyz=3]
    source_mask: torch.Tensor,  # [batch_size, num_points]
    target_mask: torch.Tensor,  # [batch_size, num_points]
    src_hash: torch.Tensor,  # [batch_size, num_points, num_levels]
    tgt_hash: torch.Tensor,  # [batch_size, num_points, num_levels]
    bucket_counts: torch.Tensor,  # [batch_size, num_buckets=2^bits, num_levels]
    num_samples: int,  # Number of target points to sample per source point
    debug: bool = False,
) -> torch.Tensor:  # [batch_size] Mean distance per batch
    """Compute one direction of Chamfer distance (source -> target).
    For each source point, find its bucket at the finest non-empty level,
    sample target points from that bucket, and compute min distance.
    Only samples from valid target points and averages over valid source points.
    """
    t_start = time.time()
    device = source.device
    batch_size, num_points_1, _ = source.shape
    _, num_points_2, num_levels = tgt_hash.shape

    assert num_points_1 == num_points_2, (
        f"Source and target point clouds must have the same number of points, "
        f"but got {num_points_1} and {num_points_2}"
    )
    num_points = num_points_1  # Number of points in each point cloud

    # Stage 1: Find lowest non-empty level
    t0 = time.time()
    # Input:  src_hash [B, N, L], bucket_counts [B, 2^b, L]
    # Output: finest_level [B, N], bucket_ids [B, N]

    # For each source point's bucket at each level, get number of points in that bucket
    batch_idx = torch.arange(batch_size, device=device)[:, None, None]  # [B, 1, 1]
    points_in_bucket = bucket_counts[
        batch_idx,  # [B, 1, 1]
        src_hash,  # [B, N, L]
        torch.arange(num_levels, device=device),  # [L]
    ]  # [B, N, L]

    # Find finest level where bucket has points
    level_weights = torch.arange(num_levels, device=device)  # [L]
    finest_level = ((points_in_bucket > 0) * level_weights).argmax(dim=-1)  # [B, N]

    # Get bucket IDs at finest level
    bucket_ids = src_hash[
        batch_idx.squeeze(-1),  # [B, 1]
        torch.arange(num_points, device=device)[None, :],  # [1, N]
        finest_level,  # [B, N]
    ]  # [B, N]
    t1 = time.time()
    if debug:
        print(f"Stage 1 (Find level): {t1 - t0:.3f}s")

    # Stage 2: Sample valid target points
    t0 = time.time()
    # Input:  finest_level [B, N], bucket_ids [B, N]
    #         tgt_hash [B, N, L], target_mask [B, N]
    # Output: sampled_points [B, N, e, xyz=3]

    # Create mask of which points are in same bucket at finest level
    # [B, N, N] where True means target point j is in source point i's bucket
    same_bucket = tgt_hash[..., finest_level[..., None]] == bucket_ids[..., None]
    valid_targets = same_bucket & target_mask[:, None, :]  # Mask invalid points

    # Create probability distribution over valid target points
    probs = valid_targets.float()  # [B, N, N]
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)  # Normalize

    # Sample target points
    sample_idx = torch.multinomial(
        probs.reshape(-1, num_points),  # [B*N, N]
        num_samples=num_samples,
        replacement=True,
    ).reshape(batch_size, num_points, num_samples)  # [B, N, e]

    # Get sampled points
    batch_idx = torch.arange(batch_size, device=device)[:, None, None]  # [B, 1, 1]
    sampled_points = target[batch_idx, sample_idx]  # [B, N, e, 3]

    # Zero out points from empty buckets
    valid_mask = valid_targets.any(dim=-1)  # [B, N]
    sampled_points = sampled_points * valid_mask[..., None, None]  # [B, N, e, 3]
    t1 = time.time()
    if debug:
        print(f"Stage 2 (Sample points): {t1 - t0:.3f}s")

    # Stage 3: Compute masked distances
    t0 = time.time()
    # Input:  source [B, N, xyz=3], sampled_points [B, N, e, xyz=3]
    #         source_mask [B, N]
    # Output: mean_dists [B]

    # Expand source points for broadcasting with samples
    source_expanded = rearrange(source, "b n d -> b n 1 d")  # [B, N, 1, 3]

    # Compute distances between each source point and its samples
    dists = torch.norm(source_expanded - sampled_points, dim=-1)  # [B, N, e]

    # Get minimum distance to any sample for each source point
    min_dists = dists.min(dim=-1)[0]  # [B, N]

    # Mask out invalid source points and compute mean
    masked_dists = min_dists * source_mask  # [B, N]
    mean_dists = masked_dists.sum(dim=-1) / (source_mask.sum(dim=-1) + 1e-6)  # [B]
    t1 = time.time()
    if debug:
        print(f"Stage 3 (Compute distances): {t1 - t0:.3f}s")
        print(f"Total time: {t1 - t_start:.3f}s")

    return mean_dists
