# Proto-Depth Performance Optimizations

This document details the performance optimizations implemented to accelerate the training of the Proto-Depth model.

## Summary of Optimizations

We've implemented three major categories of optimizations that should significantly reduce training time:

1. **Vectorized Mesh Transformation**
2. **Efficient Chamfer Distance Calculation**
3. **Improved Parallelization**

## 1. Vectorized Mesh Transformation

The original implementation transformed meshes using nested loops over batches, slots, and prototypes. The optimized version eliminates these loops using tensor operations:

### Key Changes:
- Replaced triple-nested loop with vectorized operations
- Batched rotation matrix computation using PyTorch3D's euler_angles_to_matrix
- Implemented efficient tensor reshaping and broadcasting for simultaneous slot processing
- Used batch matrix multiplication (torch.bmm) for applying rotations

### Expected Improvement:
- **5-10x speedup** in the mesh transformation step
- Reduction in this operation from ~5.2s to under 1s per batch
- Better GPU utilization through parallelized computation

### Implementation:
The core improvements are in `src/mesh_utils.py`, where the `transform_mesh` method has been completely rewritten to use batch operations.

## 2. Efficient Chamfer Distance Calculation

The original implementation used a computationally expensive all-pairs comparison for Chamfer distance, which scales as O(n²). We've optimized this with:

### Key Changes:
- Implemented K-nearest neighbors (KNN) approach, reducing complexity to O(n log n)
- Used PyTorch3D's optimized knn_points operation
- Vectorized slot-to-target comparisons
- Configurable k_nearest parameter (default=3) to balance accuracy vs. speed

### Expected Improvement:
- **2-5x speedup** in Chamfer loss computation
- Reduced memory usage during loss calculation
- Similar quality results with substantially less computation

### Implementation:
The improvements are in `src/mesh_utils.py` in the `compute_hybrid_chamfer_loss` method, along with supporting changes in `src/train.py` to pass the KNN parameter.

## 3. Improved Parallelization

### Key Changes:
- Increased worker threads for data loading (from 4 to 8)
- Better CPU utilization during training
- Configurable through the num_workers parameter

### Expected Improvement:
- Reduced data loading bottlenecks
- Better overall hardware utilization

## Configuration

All optimization parameters are now centralized in `train_model.py` for easy tuning:

```python
# Key optimization parameters
samples_per_slot = 200  # Points sampled per slot
k_nearest = 3          # Number of nearest neighbors for KNN
num_workers = 8        # Data loader parallelization
```

## Running the Optimized Training

To run the optimized training locally (requires PyTorch3D):
```bash
python train_model.py
```

To run on Jarvis (remote GPU machine):
```bash
./run_on_jarvis.sh
```

## Additional Notes

- These optimizations work with the original PyTorch3D version (1.13.0), no need to upgrade
- The vectorized implementation maintains the same mathematical behavior while being much faster
- Memory usage should be similar or slightly reduced compared to the original implementation
- Training time should decrease from ~15s per batch to ~3-5s per batch

## Future Optimization Ideas

1. **Mixed Precision Training**: If PyTorch3D supports a newer version of PyTorch, enable mixed precision (FP16) for further speedup
2. **Progressive Point Sampling**: Start with fewer points and increase throughout training
3. **Hierarchical Chamfer Distance**: Multi-resolution approach to further optimize point matching
4. **Reduced Mesh Resolution**: Consider using ico_sphere(level=3) instead of level=4 to reduce vertex count by 75%
