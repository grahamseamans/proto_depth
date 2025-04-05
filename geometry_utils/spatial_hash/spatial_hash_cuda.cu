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
#include "spatial_hash_cuda.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// Constants
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_TRIANGLES_PER_CELL = 512; // Maximum number of triangles per cell
constexpr float CELL_GROWTH_FACTOR = 2.0f;  // Each level up multiplies cell size by this factor

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
