"""
Tests for LSH-based Chamfer distance implementation.
"""

import torch
from src.lsh_chamfer import lsh_chamfer_distance, hash_points, chamfer_one_direction


def generate_test_clouds(
    batch_size: int = 2,
    num_source_points: int = 100,
    num_target_points: int = 150,
    device: torch.device = torch.device("cuda"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate test point clouds with known properties.

    Creates batches of point clouds with:
    - Random points in unit cube
    - Some points intentionally close to test LSH
    - Different characteristics per batch
    - Random valid/invalid points

    Args:
        batch_size: Number of point clouds in batch
        num_source_points: Number of points in source clouds
        num_target_points: Number of points in target clouds
        device: Device to place tensors on

    Returns:
        source: [batch_size, num_source_points, xyz=3] Source point clouds
        target: [batch_size, num_target_points, xyz=3] Target point clouds
        source_mask: [batch_size, num_source_points] Valid source points
        target_mask: [batch_size, num_target_points] Valid target points
    """
    # Generate random points in unit cube
    source = (
        torch.rand((batch_size, num_source_points, 3), device=device) * 2 - 1
    )  # [-1, 1]
    target = torch.zeros((batch_size, num_target_points, 3), device=device)

    # Create random masks (70-90% valid points)
    source_mask = torch.rand((batch_size, num_source_points), device=device) < 0.8
    target_mask = torch.rand((batch_size, num_target_points), device=device) < 0.8

    for b in range(batch_size):
        if b == 0:
            # Batch 0: Points close to source
            # For each source point, create 1-2 nearby target points
            for i in range(num_source_points):
                if source_mask[b, i]:  # Only create targets for valid source points
                    num_nearby = torch.randint(1, 3, (1,)).item()
                    idx_start = (i * 2) % num_target_points
                    for j in range(num_nearby):
                        idx = (idx_start + j) % num_target_points
                        target_mask[b, idx] = True
                        noise = torch.randn(3, device=device) * 0.1
                        target[b, idx] = source[b, i] + noise
        else:
            # Other batches: Points far from source
            target[b] = torch.rand_like(target[b]) * 2 + 1  # [1, 3]

    return source, target, source_mask, target_mask


def test_lsh_chamfer():
    """Test LSH-based Chamfer distance computation.

    Tests:
    1. Different sized point clouds (100 source, 150 target)
    2. Valid/invalid point masking
    3. LSH bucket sharing properties
    4. Distance preservation (close points -> same bucket)
    5. Batch independence (no cross-batch sampling)
    """
    # Stage 1: Generate test point clouds
    source, target, source_mask, target_mask = generate_test_clouds(
        batch_size=2,
        num_source_points=100,
        num_target_points=150,
    )

    # Stage 2: Test hashing
    src_hash, tgt_hash, bucket_counts = hash_points(
        source, target, source_mask, target_mask, levels=3, bits_per_level=4, debug=True
    )

    # Check shapes
    assert src_hash.shape == (2, 100, 3), "Source hash shape incorrect"
    assert tgt_hash.shape == (2, 150, 3), "Target hash shape incorrect"
    assert bucket_counts.shape == (2, 16, 3), "Bucket counts shape incorrect"

    # Check bucket sharing statistics for valid points
    for level in range(3):
        # Get hashes at this level
        src_level = src_hash[..., level]  # [2, 100]
        tgt_level = tgt_hash[..., level]  # [2, 150]

        # Compute pairwise distances and bucket sharing
        for b in range(2):
            # Only consider valid points
            valid_source = source[b][source_mask[b]]  # [V1, 3]
            valid_target = target[b][target_mask[b]]  # [V2, 3]
            valid_src_hash = src_level[b][source_mask[b]]  # [V1]
            valid_tgt_hash = tgt_level[b][target_mask[b]]  # [V2]

            # Compute distances between valid points
            dists = torch.cdist(valid_source, valid_target)  # [V1, V2]
            same_bucket = valid_src_hash[:, None] == valid_tgt_hash  # [V1, V2]

            # Points should share buckets more often when closer
            close_pairs = dists < 0.2
            far_pairs = dists > 2.0

            if close_pairs.any() and far_pairs.any():
                close_rate = same_bucket[close_pairs].float().mean()
                far_rate = same_bucket[far_pairs].float().mean()
                print(f"Batch {b}, Level {level}:")
                print(f"  Close rate: {close_rate:.3f}, Far rate: {far_rate:.3f}")
                print(
                    f"  Valid source: {valid_source.shape[0]}, Valid target: {valid_target.shape[0]}"
                )
                assert close_rate > far_rate, (
                    f"LSH not preserving distances at level {level}"
                )

    # Stage 3: Test one direction
    src_to_tgt = chamfer_one_direction(
        source,
        target,
        source_mask,
        target_mask,
        src_hash,
        tgt_hash,
        bucket_counts,
        num_samples=20,
        debug=True,
    )
    assert src_to_tgt.shape == (2,), "One-way distance shape incorrect"
    assert src_to_tgt[0] < src_to_tgt[1], (
        "Distance not reflecting point cloud similarity"
    )
    print(f"Source -> Target distances: {src_to_tgt}")

    # Stage 4: Test full distance
    dist = lsh_chamfer_distance(
        source,
        target,
        source_mask,
        target_mask,
        levels=3,
        bits_per_level=4,
        num_samples=20,
        debug=True,
    )
    assert dist.shape == (2,), "Full distance shape incorrect"
    assert dist[0] < dist[1], "Full distance not reflecting point cloud similarity"
    print(f"Final distances: {dist}")
    print("All LSH Chamfer distance tests passed!")


if __name__ == "__main__":
    test_lsh_chamfer()
