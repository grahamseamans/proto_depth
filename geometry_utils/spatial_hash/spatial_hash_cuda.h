#pragma once

#include <torch/extension.h>

/**
 * Struct to hold all data for one level of the spatial hash
 */
struct SpatialHashLevel
{
    torch::Tensor cell_counts;      // Number of triangles per cell
    torch::Tensor cell_offsets;     // Offset into triangle_indices for each cell
    torch::Tensor triangle_indices; // Indices of triangles in each cell
    int hash_table_size;            // Size of the hash table
};

/**
 * Builds spatial hash tables for multiple levels using memory-efficient approach.
 *
 * @param triangles Tensor of shape [num_triangles, 3, 3] containing triangle vertices
 * @param point_cloud Tensor of shape [num_points, 3] containing the query points
 * @param max_level Number of levels in the hierarchy (default: 3)
 * @param min_cell_size Base cell size for the finest level (default: 1.0)
 * @return Vector of SpatialHashLevel objects, one for each level
 */
std::vector<SpatialHashLevel> build_spatial_hash_tables(
    torch::Tensor triangles,
    torch::Tensor point_cloud,
    int max_level = 3,
    float min_cell_size = 1.0);

/**
 * Finds the nearest triangle for each point in the point cloud using spatial hash tables.
 *
 * @param point_cloud Tensor of shape [num_points, 3] containing query points
 * @param triangles Tensor of shape [num_triangles, 3, 3] containing triangle vertices
 * @param hash_levels Vector of SpatialHashLevel objects, one for each level
 * @return Tensor of shape [num_points] containing indices of the nearest triangles
 */
torch::Tensor find_nearest_triangles(
    torch::Tensor point_cloud,
    torch::Tensor triangles,
    const std::vector<SpatialHashLevel> &hash_levels);
