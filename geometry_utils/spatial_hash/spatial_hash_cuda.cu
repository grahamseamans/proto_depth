#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <tuple>
#include <cmath>
#include <limits>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "spatial_hash_cuda.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// Constants
constexpr int BLOCK_SIZE = 256;
constexpr float CELL_GROWTH_FACTOR = 2.0f; // Scale factor between hierarchy levels

// Define the SpatialHashLevel struct needed for backward compatibility
struct SpatialHashLevel
{
    torch::Tensor cell_counts;      // Number of triangles per cell
    torch::Tensor cell_offsets;     // Offset into triangle_indices for each cell
    torch::Tensor triangle_indices; // Indices of triangles in each cell
    int hash_table_size;            // Size of the hash table
};

// Custom atomicMin for float
__device__ float atomicMinFloat(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int;
    int expected;
    do
    {
        expected = old;
        old = atomicCAS(address_as_int, expected,
                        __float_as_int(min(val, __int_as_float(expected))));
    } while (expected != old);
    return __int_as_float(old);
}

// Custom atomicMax for float
__device__ float atomicMaxFloat(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int;
    int expected;
    do
    {
        expected = old;
        old = atomicCAS(address_as_int, expected,
                        __float_as_int(max(val, __int_as_float(expected))));
    } while (expected != old);
    return __int_as_float(old);
}

// We no longer need the large HashCell struct - replaced with a more memory-efficient approach

// Hash function for 3D cell coordinates
__device__ uint32_t hash_coords(int x, int y, int z, int hash_size)
{
    constexpr uint32_t p1 = 73856093;
    constexpr uint32_t p2 = 19349663;
    constexpr uint32_t p3 = 83492791;

    // Simple spatial hash function
    return ((uint32_t)x * p1 ^ (uint32_t)y * p2 ^ (uint32_t)z * p3) % hash_size;
}

// Convert float coordinates to cell coordinates
__device__ void get_cell_coords(const float3 &point, float cell_size, int &x, int &y, int &z)
{
    x = floor(point.x / cell_size);
    y = floor(point.y / cell_size);
    z = floor(point.z / cell_size);
}

// Extract triangle vertices from a flattened triangle array
__device__ void extract_triangle_vertices(
    const float *triangles,
    int tri_idx,
    float3 &v1,
    float3 &v2,
    float3 &v3)
{
    v1.x = triangles[(tri_idx * 3 + 0) * 3 + 0];
    v1.y = triangles[(tri_idx * 3 + 0) * 3 + 1];
    v1.z = triangles[(tri_idx * 3 + 0) * 3 + 2];

    v2.x = triangles[(tri_idx * 3 + 1) * 3 + 0];
    v2.y = triangles[(tri_idx * 3 + 1) * 3 + 1];
    v2.z = triangles[(tri_idx * 3 + 1) * 3 + 2];

    v3.x = triangles[(tri_idx * 3 + 2) * 3 + 0];
    v3.y = triangles[(tri_idx * 3 + 2) * 3 + 1];
    v3.z = triangles[(tri_idx * 3 + 2) * 3 + 2];
}

// Compute triangle centroid
__device__ float3 compute_centroid(const float3 &v1, const float3 &v2, const float3 &v3)
{
    float3 centroid;
    centroid.x = (v1.x + v2.x + v3.x) / 3.0f;
    centroid.y = (v1.y + v2.y + v3.y) / 3.0f;
    centroid.z = (v1.z + v2.z + v3.z) / 3.0f;
    return centroid;
}

// Compute squared distance between two 3D points
__device__ float squared_distance(const float3 &a, const float3 &b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

// Calculate min and max bounds for both triangles and point cloud
__global__ void calculate_bounds_kernel(
    const float *triangles, // [num_triangles, 3, 3]
    const float *points,    // [num_points, 3]
    int num_triangles,
    int num_points,
    float *min_bounds, // [3]
    float *max_bounds) // [3]
{
    extern __shared__ float shared_min_max[];
    float *shared_min = shared_min_max;
    float *shared_max = shared_min_max + 3;

    // Initialize shared memory
    if (threadIdx.x < 3)
    {
        shared_min[threadIdx.x] = FLT_MAX;
        shared_max[threadIdx.x] = -FLT_MAX;
    }
    __syncthreads();

    // Process triangles
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_triangles * 3; i += blockDim.x * gridDim.x)
    {
        int vertex_idx = i % 3;
        int tri_idx = i / 3;

        for (int dim = 0; dim < 3; dim++)
        {
            float val = triangles[(tri_idx * 3 + vertex_idx) * 3 + dim];
            atomicMinFloat(&shared_min[dim], val);
            atomicMaxFloat(&shared_max[dim], val);
        }
    }

    // Process points
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_points; i += blockDim.x * gridDim.x)
    {
        for (int dim = 0; dim < 3; dim++)
        {
            float val = points[i * 3 + dim];
            atomicMinFloat(&shared_min[dim], val);
            atomicMaxFloat(&shared_max[dim], val);
        }
    }

    __syncthreads();

    // Reduce to global memory
    if (threadIdx.x < 3)
    {
        atomicMinFloat(&min_bounds[threadIdx.x], shared_min[threadIdx.x]);
        atomicMaxFloat(&max_bounds[threadIdx.x], shared_max[threadIdx.x]);
    }
}

// First pass: Count triangles per cell
__global__ void count_triangles_kernel(
    const float *triangles, // [num_triangles, 3, 3]
    int num_triangles,
    const float *min_bounds, // [3]
    const float *max_bounds, // [3]
    int *cell_counts,        // [hash_table_size] - number of triangles per cell
    int *hash_table_sizes,   // Size of hash table for each level
    int level,               // Current level to process
    float min_cell_size)
{
    const int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tri_idx >= num_triangles)
        return;

    // Get triangle vertices
    float3 v1, v2, v3;
    v1.x = triangles[(tri_idx * 3 + 0) * 3 + 0];
    v1.y = triangles[(tri_idx * 3 + 0) * 3 + 1];
    v1.z = triangles[(tri_idx * 3 + 0) * 3 + 2];

    v2.x = triangles[(tri_idx * 3 + 1) * 3 + 0];
    v2.y = triangles[(tri_idx * 3 + 1) * 3 + 1];
    v2.z = triangles[(tri_idx * 3 + 1) * 3 + 2];

    v3.x = triangles[(tri_idx * 3 + 2) * 3 + 0];
    v3.y = triangles[(tri_idx * 3 + 2) * 3 + 1];
    v3.z = triangles[(tri_idx * 3 + 2) * 3 + 2];

    // Compute centroid
    float3 centroid = compute_centroid(v1, v2, v3);

    // Calculate cell size for current level
    float cell_size = min_cell_size * pow(CELL_GROWTH_FACTOR, level);

    // Get cell coordinates
    int cell_x, cell_y, cell_z;
    get_cell_coords(centroid, cell_size, cell_x, cell_y, cell_z);

    // Compute hash for this cell
    uint32_t hash_value = hash_coords(cell_x, cell_y, cell_z, hash_table_sizes[level]);

    // Increment the count for this cell
    atomicAdd(&cell_counts[hash_value], 1);
}

// Second pass: Fill triangle indices
__global__ void fill_triangle_indices_kernel(
    const float *triangles, // [num_triangles, 3, 3]
    int num_triangles,
    const float *min_bounds, // [3]
    const float *max_bounds, // [3]
    int *cell_counts,        // [hash_table_size] - current count per cell (used as atomic counter)
    const int *cell_offsets, // [hash_table_size] - offsets into triangle_indices
    int *triangle_indices,   // [total_triangles] - indices of triangles
    int *hash_table_sizes,   // Size of hash table for each level
    int level,               // Current level to process
    float min_cell_size)
{
    const int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tri_idx >= num_triangles)
        return;

    // Get triangle vertices
    float3 v1, v2, v3;
    v1.x = triangles[(tri_idx * 3 + 0) * 3 + 0];
    v1.y = triangles[(tri_idx * 3 + 0) * 3 + 1];
    v1.z = triangles[(tri_idx * 3 + 0) * 3 + 2];

    v2.x = triangles[(tri_idx * 3 + 1) * 3 + 0];
    v2.y = triangles[(tri_idx * 3 + 1) * 3 + 1];
    v2.z = triangles[(tri_idx * 3 + 1) * 3 + 2];

    v3.x = triangles[(tri_idx * 3 + 2) * 3 + 0];
    v3.y = triangles[(tri_idx * 3 + 2) * 3 + 1];
    v3.z = triangles[(tri_idx * 3 + 2) * 3 + 2];

    // Compute centroid
    float3 centroid = compute_centroid(v1, v2, v3);

    // Calculate cell size for current level
    float cell_size = min_cell_size * pow(CELL_GROWTH_FACTOR, level);

    // Get cell coordinates
    int cell_x, cell_y, cell_z;
    get_cell_coords(centroid, cell_size, cell_x, cell_y, cell_z);

    // Compute hash for this cell
    uint32_t hash_value = hash_coords(cell_x, cell_y, cell_z, hash_table_sizes[level]);

    // Get current index for this cell and increment atomically
    int index = atomicAdd(&cell_counts[hash_value], 1);

    // Calculate position in the global triangle indices array
    int position = cell_offsets[hash_value] + index;

    // Store the triangle index
    triangle_indices[position] = tri_idx;
}

// Find nearest triangle for each point (modified to work with new data structure)
__global__ void find_nearest_triangles_kernel(
    const float *points, // [num_points, 3]
    int num_points,
    const float *triangles, // [num_triangles, 3, 3]
    int num_triangles,
    const int *cell_counts,      // [hash_table_size] - number of triangles per cell
    const int *cell_offsets,     // [hash_table_size] - offsets into triangle_indices
    const int *triangle_indices, // [total_triangles] - indices of triangles
    const int *hash_table_sizes, // Size of hash table for each level
    int max_level,
    float min_cell_size,
    int *result_indices) // [num_points]
{
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points)
        return;

    // Get point coordinates
    float3 point;
    point.x = points[point_idx * 3 + 0];
    point.y = points[point_idx * 3 + 1];
    point.z = points[point_idx * 3 + 2];

    // First try finest level with 3x3x3 neighborhood
    float cell_size = min_cell_size;
    int cell_x, cell_y, cell_z;
    get_cell_coords(point, cell_size, cell_x, cell_y, cell_z);

    // Check 3x3x3 neighborhood
    float min_dist = FLT_MAX;
    int closest_tri = -1;
    bool found_triangle = false;

    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int z = -1; z <= 1; z++)
            {
                int neighbor_x = cell_x + x;
                int neighbor_y = cell_y + y;
                int neighbor_z = cell_z + z;

                uint32_t hash_value = hash_coords(neighbor_x, neighbor_y, neighbor_z, hash_table_sizes[0]);

                // Get information about this cell
                int count = cell_counts[hash_value];
                int offset = cell_offsets[hash_value];

                // Compute distances to triangles in this cell
                for (int i = 0; i < count; i++)
                {
                    int tri_idx = triangle_indices[offset + i];

                    // Get triangle vertices
                    float3 v1, v2, v3;
                    v1.x = triangles[(tri_idx * 3 + 0) * 3 + 0];
                    v1.y = triangles[(tri_idx * 3 + 0) * 3 + 1];
                    v1.z = triangles[(tri_idx * 3 + 0) * 3 + 2];

                    v2.x = triangles[(tri_idx * 3 + 1) * 3 + 0];
                    v2.y = triangles[(tri_idx * 3 + 1) * 3 + 1];
                    v2.z = triangles[(tri_idx * 3 + 1) * 3 + 2];

                    v3.x = triangles[(tri_idx * 3 + 2) * 3 + 0];
                    v3.y = triangles[(tri_idx * 3 + 2) * 3 + 1];
                    v3.z = triangles[(tri_idx * 3 + 2) * 3 + 2];

                    // Compute centroid and distance
                    float3 centroid = compute_centroid(v1, v2, v3);
                    float dist = squared_distance(point, centroid);

                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        closest_tri = tri_idx;
                        found_triangle = true;
                    }
                }
            }
        }
    }

    // If no triangles found at finest level, move up to coarser levels
    if (!found_triangle)
    {
        for (int level = 1; level < max_level; level++)
        {
            cell_size = min_cell_size * pow(CELL_GROWTH_FACTOR, level);
            get_cell_coords(point, cell_size, cell_x, cell_y, cell_z);

            uint32_t hash_value = hash_coords(cell_x, cell_y, cell_z, hash_table_sizes[level]);

            // Get information about this cell
            int count = cell_counts[hash_value + level * hash_table_sizes[0]]; // Offset by level

            // If we find triangles at this level, pick a random one (just use the first one for simplicity)
            if (count > 0)
            {
                int offset = cell_offsets[hash_value + level * hash_table_sizes[0]];
                closest_tri = triangle_indices[offset];
                found_triangle = true;
                break;
            }
        }
    }

    // If still no triangles found, pick a random triangle from the entire set
    if (!found_triangle && num_triangles > 0)
    {
        closest_tri = point_idx % num_triangles; // Use point_idx as a pseudo-random number
    }

    result_indices[point_idx] = closest_tri;
}

// C++ wrappers for our CUDA kernels

std::vector<SpatialHashLevel> build_spatial_hash_tables(
    torch::Tensor triangles,
    torch::Tensor point_cloud,
    int max_level,
    float min_cell_size)
{
    CHECK_INPUT(triangles);
    CHECK_INPUT(point_cloud);

    const auto num_triangles = triangles.size(0);
    const auto num_points = point_cloud.size(0);

    // Device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, triangles.device().index());

    // Calculate bounds
    auto min_bounds = torch::full({3}, std::numeric_limits<float>::max(),
                                  torch::TensorOptions().device(triangles.device()).dtype(torch::kFloat32));
    auto max_bounds = torch::full({3}, -std::numeric_limits<float>::max(),
                                  torch::TensorOptions().device(triangles.device()).dtype(torch::kFloat32));

    // Launch bounds calculation kernel
    const int num_blocks = (std::max(num_triangles * 3, num_points) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    calculate_bounds_kernel<<<num_blocks, BLOCK_SIZE, 6 * sizeof(float)>>>(
        triangles.data_ptr<float>(),
        point_cloud.data_ptr<float>(),
        num_triangles,
        num_points,
        min_bounds.data_ptr<float>(),
        max_bounds.data_ptr<float>());

    // Determine hash table sizes for each level
    std::vector<int> hash_sizes(max_level);
    for (int level = 0; level < max_level; level++)
    {
        // Each level has a different cell size, so we adjust the hash table size accordingly
        float cell_size = min_cell_size * pow(CELL_GROWTH_FACTOR, level);
        float volume = 1.0f;
        for (int dim = 0; dim < 3; dim++)
        {
            volume *= (max_bounds[dim].item<float>() - min_bounds[dim].item<float>()) / cell_size;
        }

        // Estimate number of cells and limit to a reasonable size
        int est_cells = std::max(static_cast<int>(volume), 100);
        hash_sizes[level] = std::min(est_cells * 2, 100000); // Limit to 100K entries
    }

    // Create tensor to hold hash table sizes
    auto hash_sizes_tensor = torch::from_blob(hash_sizes.data(), {max_level},
                                              torch::TensorOptions().dtype(torch::kInt32))
                                 .to(triangles.device());

    // Build the hash tables for each level separately
    std::vector<SpatialHashLevel> hash_levels;

    for (int level = 0; level < max_level; level++)
    {
        int hash_table_size = hash_sizes[level];
        SpatialHashLevel hash_level;
        hash_level.hash_table_size = hash_table_size;

        // === PHASE 1: Count triangles per cell ===

        // Allocate cell counts
        hash_level.cell_counts = torch::zeros({hash_table_size},
                                              torch::TensorOptions().device(triangles.device()).dtype(torch::kInt32));

        // Count triangles per cell
        count_triangles_kernel<<<(num_triangles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            triangles.data_ptr<float>(),
            num_triangles,
            min_bounds.data_ptr<float>(),
            max_bounds.data_ptr<float>(),
            hash_level.cell_counts.data_ptr<int>(),
            hash_sizes_tensor.data_ptr<int>(),
            level,
            min_cell_size);

        // === PHASE 2: Compute offsets and allocate space ===

        // Allocate cell offsets
        hash_level.cell_offsets = torch::zeros({hash_table_size},
                                               torch::TensorOptions().device(triangles.device()).dtype(torch::kInt32));

        // Compute exclusive sum to get offsets
        thrust::device_ptr<int> d_counts(hash_level.cell_counts.data_ptr<int>());
        thrust::device_ptr<int> d_offsets(hash_level.cell_offsets.data_ptr<int>());
        thrust::exclusive_scan(d_counts, d_counts + hash_table_size, d_offsets);

        // Get total triangle count from the last offset + count
        int last_offset, last_count;
        cudaMemcpy(&last_offset, hash_level.cell_offsets.data_ptr<int>() + hash_table_size - 1,
                   sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_count, hash_level.cell_counts.data_ptr<int>() + hash_table_size - 1,
                   sizeof(int), cudaMemcpyDeviceToHost);
        int total_triangles = last_offset + last_count;

        // Allocate triangle indices array
        hash_level.triangle_indices = torch::zeros({total_triangles},
                                                   torch::TensorOptions().device(triangles.device()).dtype(torch::kInt32));

        // Reset counts to use as counters in the next kernel
        cudaMemset(hash_level.cell_counts.data_ptr<int>(), 0, hash_table_size * sizeof(int));

        // === PHASE 3: Fill in triangle indices ===

        // Fill triangle indices
        fill_triangle_indices_kernel<<<(num_triangles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            triangles.data_ptr<float>(),
            num_triangles,
            min_bounds.data_ptr<float>(),
            max_bounds.data_ptr<float>(),
            hash_level.cell_counts.data_ptr<int>(), // Will be used as atomic counter
            hash_level.cell_offsets.data_ptr<int>(),
            hash_level.triangle_indices.data_ptr<int>(),
            hash_sizes_tensor.data_ptr<int>(),
            level,
            min_cell_size);

        // Add this level to the result
        hash_levels.push_back(hash_level);
    }

    return hash_levels;
}

torch::Tensor find_nearest_triangles(
    torch::Tensor point_cloud,
    torch::Tensor triangles,
    const std::vector<SpatialHashLevel> &hash_levels)
{
    CHECK_INPUT(point_cloud);
    CHECK_INPUT(triangles);
    for (const auto &level : hash_levels)
    {
        CHECK_INPUT(level.cell_counts);
        CHECK_INPUT(level.cell_offsets);
        CHECK_INPUT(level.triangle_indices);
    }

    const auto num_points = point_cloud.size(0);
    const auto num_triangles = triangles.size(0);
    const int max_level = hash_levels.size();

    // Determine hash table sizes
    std::vector<int> hash_sizes(max_level);
    for (int level = 0; level < max_level; level++)
    {
        hash_sizes[level] = hash_levels[level].hash_table_size;
    }

    // Create tensor to hold hash table sizes
    auto hash_sizes_tensor = torch::from_blob(hash_sizes.data(), {max_level},
                                              torch::TensorOptions().dtype(torch::kInt32))
                                 .to(point_cloud.device());

    // Allocate output tensor
    auto result = torch::empty({num_points},
                               torch::TensorOptions().device(point_cloud.device()).dtype(torch::kInt32));

    // Create combined arrays for all levels
    std::vector<torch::Tensor> all_counts, all_offsets, all_indices;
    for (int level = 0; level < max_level; level++)
    {
        all_counts.push_back(hash_levels[level].cell_counts);
        all_offsets.push_back(hash_levels[level].cell_offsets);
        all_indices.push_back(hash_levels[level].triangle_indices);
    }

    // Concatenate arrays
    auto cell_counts = torch::cat(all_counts);
    auto cell_offsets = torch::cat(all_offsets);
    auto triangle_indices = torch::cat(all_indices);

    // Launch kernel
    const int num_blocks = (num_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
    find_nearest_triangles_kernel<<<num_blocks, BLOCK_SIZE>>>(
        point_cloud.data_ptr<float>(),
        num_points,
        triangles.data_ptr<float>(),
        num_triangles,
        cell_counts.data_ptr<int>(),
        cell_offsets.data_ptr<int>(),
        triangle_indices.data_ptr<int>(),
        hash_sizes_tensor.data_ptr<int>(),
        max_level,
        1.0f, // min_cell_size - this should be a parameter or inferred from hash table
        result.data_ptr<int>());

    return result;
}

//==============================================================================
// New Hierarchical Grid Implementation
//==============================================================================

// Compute bin index for each triangle (finest level only)
__global__ void compute_triangle_bins_kernel(
    const float *triangles, // [num_triangles, 3, 3]
    int num_triangles,
    int *bin_indices, // Output bin indices [num_triangles]
    float cell_size,
    int3 grid_dims) // Number of bins in each dimension
{
    const int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tri_idx >= num_triangles)
        return;

    // Get triangle vertices
    float3 v1, v2, v3;
    v1.x = triangles[(tri_idx * 3 + 0) * 3 + 0];
    v1.y = triangles[(tri_idx * 3 + 0) * 3 + 1];
    v1.z = triangles[(tri_idx * 3 + 0) * 3 + 2];

    v2.x = triangles[(tri_idx * 3 + 1) * 3 + 0];
    v2.y = triangles[(tri_idx * 3 + 1) * 3 + 1];
    v2.z = triangles[(tri_idx * 3 + 1) * 3 + 2];

    v3.x = triangles[(tri_idx * 3 + 2) * 3 + 0];
    v3.y = triangles[(tri_idx * 3 + 2) * 3 + 1];
    v3.z = triangles[(tri_idx * 3 + 2) * 3 + 2];

    // Compute centroid
    float3 centroid = compute_centroid(v1, v2, v3);

    // Get cell coordinates
    int cell_x, cell_y, cell_z;
    get_cell_coords(centroid, cell_size, cell_x, cell_y, cell_z);

    // Clamp to grid bounds
    cell_x = max(0, min(cell_x, grid_dims.x - 1));
    cell_y = max(0, min(cell_y, grid_dims.y - 1));
    cell_z = max(0, min(cell_z, grid_dims.z - 1));

    // Compute flat bin index (row-major order)
    int bin_idx = cell_x + cell_y * grid_dims.x + cell_z * grid_dims.x * grid_dims.y;

    // Store result
    bin_indices[tri_idx] = bin_idx;
}

// Compute bin boundaries from sorted bin indices
__global__ void compute_bin_boundaries_kernel(
    const int *bin_indices, // [num_triangles] - sorted bin indices
    int num_triangles,
    int *bin_boundaries, // [max_bins + 1] - output bin boundaries
    int max_bins)
{
    // Initialize all boundaries to num_triangles (empty bin sentinel)
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx <= max_bins;
         idx += blockDim.x * gridDim.x)
    {
        if (idx <= max_bins)
        {
            bin_boundaries[idx] = num_triangles;
        }
    }
    __syncthreads();

    // First triangle always starts a bin
    if (blockIdx.x == 0 && threadIdx.x == 0 && num_triangles > 0)
    {
        bin_boundaries[bin_indices[0]] = 0;
    }

    // For each pair of consecutive triangles
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
         idx < num_triangles;
         idx += blockDim.x * gridDim.x)
    {
        // If bin changes, mark the start of the new bin
        if (bin_indices[idx] != bin_indices[idx - 1])
        {
            bin_boundaries[bin_indices[idx]] = idx;
        }
    }

    // Use a single thread to convert to bin boundaries format
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // Scan to fill in empty bins and add final sentinel
        int last_valid = num_triangles;
        for (int i = max_bins; i >= 0; i--)
        {
            if (bin_boundaries[i] == num_triangles)
            {
                bin_boundaries[i] = last_valid;
            }
            else
            {
                last_valid = bin_boundaries[i];
            }
        }
    }
}

// Compute counts for higher level bins
__global__ void compute_level_counts_kernel(
    const int *fine_bin_counts, // [fine_grid_size^3] - triangle counts for finest level
    int fine_grid_size,         // Size of finest grid (per dimension)
    int *coarse_bin_counts,     // [coarse_grid_size^3] - output counts for coarser level
    int coarse_grid_size)       // Size of coarser grid (per dimension)
{
    // Get coarse bin coordinates
    int coarse_x = blockIdx.x * blockDim.x + threadIdx.x;
    int coarse_y = blockIdx.y;
    int coarse_z = blockIdx.z;

    if (coarse_x >= coarse_grid_size || coarse_y >= coarse_grid_size || coarse_z >= coarse_grid_size)
        return;

    // Compute flat bin index for coarse bin
    int coarse_idx = coarse_x + coarse_y * coarse_grid_size + coarse_z * coarse_grid_size * coarse_grid_size;

    // Scaling factor between grids
    int scale = fine_grid_size / coarse_grid_size;

    // Initialize count
    int count = 0;

    // Iterate through all fine cells that map to this coarse cell
    for (int fx = coarse_x * scale; fx < (coarse_x + 1) * scale; fx++)
    {
        for (int fy = coarse_y * scale; fy < (coarse_y + 1) * scale; fy++)
        {
            for (int fz = coarse_z * scale; fz < (coarse_z + 1) * scale; fz++)
            {
                if (fx < fine_grid_size && fy < fine_grid_size && fz < fine_grid_size)
                {
                    int fine_idx = fx + fy * fine_grid_size + fz * fine_grid_size * fine_grid_size;
                    count += fine_bin_counts[fine_idx];
                }
            }
        }
    }

    // Store the total count
    coarse_bin_counts[coarse_idx] = count;
}

// Sample triangles for non-empty bins
__global__ void sample_triangles_kernel(
    const int *bin_boundaries,   // [max_bins + 1] - bin boundaries
    const int *triangle_indices, // [num_triangles] - sorted triangle indices
    int *sample_triangles,       // [max_bins] - output sample triangle per bin
    int max_bins)
{
    // Process one bin per thread
    int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_idx >= max_bins)
        return;

    // Get bin range
    int start = bin_boundaries[bin_idx];
    int end = bin_boundaries[bin_idx + 1];

    // If bin is not empty, sample the first triangle
    if (start < end)
    {
        sample_triangles[bin_idx] = triangle_indices[start];
    }
    else
    {
        sample_triangles[bin_idx] = -1; // Mark empty bin
    }
}

// Find nearest triangle using hierarchical grid
__global__ void find_nearest_triangles_hierarchical_kernel(
    const float *points, // [num_points, 3]
    int num_points,
    const float *triangles, // [num_triangles, 3, 3]
    int num_triangles,
    const int *triangle_indices,       // [num_triangles] - sorted triangle indices
    const int *bin_boundaries,         // [max_bins + 1] - bin boundaries
    const int *level_bin_counts,       // [num_levels][max_bins] - bin counts for each level
    const int *level_sample_triangles, // [num_levels][max_bins] - sample triangles
    int *result_indices,               // [num_points] - output nearest triangle indices
    int num_levels,                    // Number of hierarchy levels
    int *level_grid_sizes,             // [num_levels] - grid size for each level
    float base_cell_size)              // Cell size for finest level
{
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points)
        return;

    // Get point coordinates
    float3 point;
    point.x = points[point_idx * 3 + 0];
    point.y = points[point_idx * 3 + 1];
    point.z = points[point_idx * 3 + 2];

    // Start with finest level (level 0)
    bool found_triangle = false;
    int closest_tri = -1;
    float min_dist = FLT_MAX;

    // Get cell coordinates for finest level
    int cell_x, cell_y, cell_z;
    get_cell_coords(point, base_cell_size, cell_x, cell_y, cell_z);
    int finest_grid_size = level_grid_sizes[0];

    // Clamp to grid bounds
    cell_x = max(0, min(cell_x, finest_grid_size - 1));
    cell_y = max(0, min(cell_y, finest_grid_size - 1));
    cell_z = max(0, min(cell_z, finest_grid_size - 1));

    // Check 3x3x3 neighborhood at finest level
    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int z = -1; z <= 1; z++)
            {
                int nx = cell_x + x;
                int ny = cell_y + y;
                int nz = cell_z + z;

                // Skip out-of-bounds cells
                if (nx < 0 || nx >= finest_grid_size ||
                    ny < 0 || ny >= finest_grid_size ||
                    nz < 0 || nz >= finest_grid_size)
                    continue;

                // Get bin index
                int bin_idx = nx + ny * finest_grid_size + nz * finest_grid_size * finest_grid_size;

                // Get triangle range for this bin
                int start = bin_boundaries[bin_idx];
                int end = bin_boundaries[bin_idx + 1];

                // Process triangles in this bin
                for (int i = start; i < end; i++)
                {
                    int tri_idx = triangle_indices[i];

                    // Get triangle vertices
                    float3 v1, v2, v3;
                    v1.x = triangles[(tri_idx * 3 + 0) * 3 + 0];
                    v1.y = triangles[(tri_idx * 3 + 0) * 3 + 1];
                    v1.z = triangles[(tri_idx * 3 + 0) * 3 + 2];

                    v2.x = triangles[(tri_idx * 3 + 1) * 3 + 0];
                    v2.y = triangles[(tri_idx * 3 + 1) * 3 + 1];
                    v2.z = triangles[(tri_idx * 3 + 1) * 3 + 2];

                    v3.x = triangles[(tri_idx * 3 + 2) * 3 + 0];
                    v3.y = triangles[(tri_idx * 3 + 2) * 3 + 1];
                    v3.z = triangles[(tri_idx * 3 + 2) * 3 + 2];

                    // Compute centroid and distance
                    float3 centroid = compute_centroid(v1, v2, v3);
                    float dist = squared_distance(point, centroid);

                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        closest_tri = tri_idx;
                        found_triangle = true;
                    }
                }
            }
        }
    }

    // If no triangles found at finest level, check coarser levels
    if (!found_triangle)
    {
        for (int level = 1; level < num_levels; level++)
        {
            // Get grid size for this level
            int grid_size = level_grid_sizes[level];
            float cell_size = base_cell_size * (1 << level); // Each level doubles cell size

            // Get cell coordinates for this level
            get_cell_coords(point, cell_size, cell_x, cell_y, cell_z);

            // Clamp to grid bounds
            cell_x = max(0, min(cell_x, grid_size - 1));
            cell_y = max(0, min(cell_y, grid_size - 1));
            cell_z = max(0, min(cell_z, grid_size - 1));

            // Get bin index
            int bin_idx = cell_x + cell_y * grid_size + cell_z * grid_size * grid_size;

            // Compute offset for this level in the flattened arrays
            int level_offset = 0;
            for (int l = 0; l < level; l++)
            {
                int size = level_grid_sizes[l];
                level_offset += size * size * size;
            }

            // Check if bin has triangles
            if (level_bin_counts[level_offset + bin_idx] > 0)
            {
                // Just use a sample triangle from this bin
                closest_tri = level_sample_triangles[level_offset + bin_idx];
                found_triangle = true;
                break;
            }
        }
    }

    // If still no triangle found, use a random one
    if (!found_triangle && num_triangles > 0)
    {
        closest_tri = point_idx % num_triangles;
    }

    // Set result
    result_indices[point_idx] = closest_tri;
}

// Main hierarchical grid implementation

HierarchicalGrid *create_or_update_hierarchical_grid(
    torch::Tensor triangles,
    torch::Tensor point_cloud,
    HierarchicalGrid *prev_grid,
    int num_levels,
    float base_cell_size,
    float growth_factor)
{
    CHECK_INPUT(triangles);
    CHECK_INPUT(point_cloud);

    const auto device = triangles.device();
    const auto options = torch::TensorOptions().device(device).dtype(torch::kInt32);
    const auto float_options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
    const auto num_triangles = triangles.size(0);

    // Initialize or reuse grid structure
    HierarchicalGrid *grid = prev_grid;
    if (!grid)
    {
        grid = new HierarchicalGrid();
        grid->initialized = false;
        grid->num_triangles = num_triangles;
    }

    // If triangles count changed, reinitialize
    if (grid->initialized && grid->num_triangles != num_triangles)
    {
        // Free existing memory
        grid->triangle_indices = torch::Tensor();
        grid->bin_indices = torch::Tensor();
        grid->bin_boundaries = torch::Tensor();
        grid->levels.clear();
        grid->initialized = false;
    }

    // Configure grid parameters
    grid->num_levels = num_levels;
    grid->base_cell_size = base_cell_size;
    grid->growth_factor = growth_factor;

    // Calculate bounds for scene
    auto min_bounds = torch::full({3}, std::numeric_limits<float>::max(), float_options);
    auto max_bounds = torch::full({3}, -std::numeric_limits<float>::max(), float_options);

    const int num_blocks = (std::max(num_triangles * 3, (int64_t)point_cloud.size(0)) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    calculate_bounds_kernel<<<num_blocks, BLOCK_SIZE, 6 * sizeof(float)>>>(
        triangles.data_ptr<float>(),
        point_cloud.data_ptr<float>(),
        num_triangles,
        point_cloud.size(0),
        min_bounds.data_ptr<float>(),
        max_bounds.data_ptr<float>());

    // Compute grid dimensions for each level
    std::vector<int> grid_sizes(num_levels);
    int max_bins = 0;

    for (int level = 0; level < num_levels; level++)
    {
        float cell_size = base_cell_size * std::pow(growth_factor, level);
        float max_extent = 0;

        // Find the largest dimension
        for (int dim = 0; dim < 3; dim++)
        {
            float extent = max_bounds[dim].item<float>() - min_bounds[dim].item<float>();
            max_extent = std::max(max_extent, extent);
        }

        // Calculate grid size (uniform across dimensions for simplicity)
        int grid_size = std::max(1, static_cast<int>(std::ceil(max_extent / cell_size)));
        grid_sizes[level] = grid_size;

        // Track maximum number of bins
        max_bins = std::max(max_bins, grid_size * grid_size * grid_size);
    }

    // Convert grid sizes to tensor
    auto grid_sizes_tensor = torch::from_blob(grid_sizes.data(), {num_levels},
                                              torch::TensorOptions().dtype(torch::kInt32))
                                 .to(device);

    // Initialize arrays if first time
    if (!grid->initialized)
    {
        // Allocate arrays for finest level
        grid->triangle_indices = torch::arange(num_triangles, options);
        grid->bin_indices = torch::zeros({num_triangles}, options);
        grid->bin_boundaries = torch::zeros({max_bins + 1}, options);

        // Initialize levels
        grid->levels.resize(num_levels);
        for (int level = 0; level < num_levels; level++)
        {
            int level_grid_size = grid_sizes[level];
            int level_bins = level_grid_size * level_grid_size * level_grid_size;

            grid->levels[level].grid_size = level_grid_size;
            grid->levels[level].cell_size = base_cell_size * std::pow(growth_factor, level);
            grid->levels[level].bin_counts = torch::zeros({level_bins}, options);
            grid->levels[level].sample_triangles = torch::zeros({level_bins}, options);
        }

        grid->max_bins = max_bins;
        grid->initialized = true;
    }

    // Compute bin indices for triangles (finest level only)
    int3 grid_dims;
    grid_dims.x = grid_dims.y = grid_dims.z = grid_sizes[0];

    compute_triangle_bins_kernel<<<(num_triangles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        triangles.data_ptr<float>(),
        num_triangles,
        grid->bin_indices.data_ptr<int>(),
        base_cell_size,
        grid_dims);

    // Sort triangles by bin directly on GPU
    if (num_triangles > 0)
    {
        thrust::sort_by_key(
            thrust::device,
            thrust::device_pointer_cast(grid->bin_indices.data_ptr<int>()),
            thrust::device_pointer_cast(grid->bin_indices.data_ptr<int>() + num_triangles),
            thrust::device_pointer_cast(grid->triangle_indices.data_ptr<int>()));
    }

    // Compute bin boundaries for fast triangle access
    compute_bin_boundaries_kernel<<<(max_bins / BLOCK_SIZE) + 1, BLOCK_SIZE>>>(
        grid->bin_indices.data_ptr<int>(),
        num_triangles,
        grid->bin_boundaries.data_ptr<int>(),
        max_bins);

    // Compute bin counts for each level
    for (int level = 0; level < num_levels; level++)
    {
        // For level 0, compute directly from bin boundaries
        if (level == 0)
        {
            int finest_bins = grid_sizes[0] * grid_sizes[0] * grid_sizes[0];

            // Launch a kernel to count triangles per bin at finest level
            sample_triangles_kernel<<<(finest_bins + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                grid->bin_boundaries.data_ptr<int>(),
                grid->triangle_indices.data_ptr<int>(),
                grid->levels[0].sample_triangles.data_ptr<int>(),
                finest_bins);

            // Compute bin counts for level 0
            for (int bin = 0; bin < finest_bins; bin++)
            {
                int start = bin < finest_bins ? grid->bin_boundaries.index({bin}).item<int>() : num_triangles;
                int end = bin + 1 < finest_bins ? grid->bin_boundaries.index({bin + 1}).item<int>() : num_triangles;
                grid->levels[0].bin_counts.index_put_({bin}, end - start);
            }
        }
        else
        {
            // For higher levels, sum counts from finer level
            int fine_grid_size = grid_sizes[level - 1];
            int coarse_grid_size = grid_sizes[level];

            // Use a 3D grid for the kernel
            dim3 block_size(32, 1, 1);
            dim3 grid_size(
                (coarse_grid_size + block_size.x - 1) / block_size.x,
                coarse_grid_size,
                coarse_grid_size);

            compute_level_counts_kernel<<<grid_size, block_size>>>(
                grid->levels[level - 1].bin_counts.data_ptr<int>(),
                fine_grid_size,
                grid->levels[level].bin_counts.data_ptr<int>(),
                coarse_grid_size);

            // Sample triangles for non-empty bins at this level
            int level_bins = coarse_grid_size * coarse_grid_size * coarse_grid_size;

            // For each non-empty bin at this level, find a child bin at the finest level
            // that has triangles, and sample one of those triangles
            for (int bin = 0; bin < level_bins; bin++)
            {
                // If bin has triangles, propagate a sample from finest level
                if (grid->levels[level].bin_counts.index({bin}).item<int>() > 0)
                {
                    // Calculate bin coordinates
                    int bz = bin / (coarse_grid_size * coarse_grid_size);
                    int by = (bin / coarse_grid_size) % coarse_grid_size;
                    int bx = bin % coarse_grid_size;

                    // Scale factor between this level and finest
                    int scale = grid_sizes[0] / coarse_grid_size;

                    // Map to a fine bin (just take the first child)
                    int fine_x = bx * scale;
                    int fine_y = by * scale;
                    int fine_z = bz * scale;

                    // Get fine bin index
                    int fine_bin = fine_x + fine_y * grid_sizes[0] + fine_z * grid_sizes[0] * grid_sizes[0];

                    // Get a sample triangle from the fine bin
                    grid->levels[level].sample_triangles.index_put_({bin},
                                                                    grid->levels[0].sample_triangles.index({fine_bin}));
                }
            }
        }
    }

    return grid;
}

torch::Tensor find_nearest_triangles_hierarchical(
    torch::Tensor point_cloud,
    torch::Tensor triangles,
    const HierarchicalGrid *grid)
{
    CHECK_INPUT(point_cloud);
    CHECK_INPUT(triangles);
    TORCH_CHECK(grid != nullptr, "Hierarchical grid is null");
    TORCH_CHECK(grid->initialized, "Hierarchical grid is not initialized");

    const auto num_points = point_cloud.size(0);
    const auto device = point_cloud.device();

    // Prepare level grid sizes and tensor
    std::vector<int> level_grid_sizes(grid->num_levels);
    for (int level = 0; level < grid->num_levels; level++)
    {
        level_grid_sizes[level] = grid->levels[level].grid_size;
    }

    auto level_grid_sizes_tensor = torch::from_blob(
                                       level_grid_sizes.data(),
                                       {grid->num_levels},
                                       torch::TensorOptions().dtype(torch::kInt32))
                                       .to(device);

    // Prepare flattened level counts and sample triangles tensors
    std::vector<torch::Tensor> level_counts_list;
    std::vector<torch::Tensor> level_samples_list;

    for (int level = 0; level < grid->num_levels; level++)
    {
        level_counts_list.push_back(grid->levels[level].bin_counts);
        level_samples_list.push_back(grid->levels[level].sample_triangles);
    }

    auto level_bin_counts = torch::cat(level_counts_list);
    auto level_sample_triangles = torch::cat(level_samples_list);

    // Allocate output tensor
    auto result = torch::empty({num_points},
                               torch::TensorOptions().device(device).dtype(torch::kInt32));

    // Launch kernel
    const int num_blocks = (num_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
    find_nearest_triangles_hierarchical_kernel<<<num_blocks, BLOCK_SIZE>>>(
        point_cloud.data_ptr<float>(),
        num_points,
        triangles.data_ptr<float>(),
        triangles.size(0),
        grid->triangle_indices.data_ptr<int>(),
        grid->bin_boundaries.data_ptr<int>(),
        level_bin_counts.data_ptr<int>(),
        level_sample_triangles.data_ptr<int>(),
        result.data_ptr<int>(),
        grid->num_levels,
        level_grid_sizes_tensor.data_ptr<int>(),
        grid->base_cell_size);

    return result;
}

void free_hierarchical_grid(HierarchicalGrid *grid)
{
    if (grid)
    {
        // Clear tensors
        grid->triangle_indices = torch::Tensor();
        grid->bin_indices = torch::Tensor();
        grid->bin_boundaries = torch::Tensor();

        // Clear levels
        for (auto &level : grid->levels)
        {
            level.bin_counts = torch::Tensor();
            level.sample_triangles = torch::Tensor();
        }
        grid->levels.clear();

        // Delete grid
        delete grid;
    }
}
