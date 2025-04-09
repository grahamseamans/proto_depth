#pragma once

#include <torch/extension.h>

/**
 * Struct representing a level in the hierarchical grid
 */
struct GridLevel
{
    torch::Tensor bin_counts;       // Number of triangles in each bin
    torch::Tensor sample_triangles; // Representative triangle for non-empty bins
    int grid_size;                  // Number of bins per dimension
    float cell_size;                // Size of each cell at this level
};

/**
 * Struct to hold persistent hierarchical spatial hash data
 */
struct HierarchicalGrid
{
    // Finest level data (with sorted triangles)
    torch::Tensor triangle_indices; // Triangle IDs sorted by bin [num_triangles]
    torch::Tensor bin_indices;      // Bin ID for each triangle [num_triangles]
    torch::Tensor bin_boundaries;   // Start indices for bins [num_bins + 1]

    // Grid levels (from finest to coarsest)
    std::vector<GridLevel> levels;

    // Configuration
    int num_levels;       // Number of hierarchical levels
    float base_cell_size; // Size of finest cell
    float growth_factor;  // Scale factor between levels

    // Memory tracking to avoid reallocations
    bool initialized;  // Whether structure is initialized
    int num_triangles; // Number of triangles (for buffer sizing)
    int max_bins;      // Maximum number of bins across levels
};

/**
 * Creates a new hierarchical grid or updates an existing one.
 * Pre-allocates memory on first call and reuses it in subsequent calls.
 *
 * @param triangles Tensor of shape [num_triangles, 3, 3] containing triangle vertices
 * @param point_cloud Tensor of shape [num_points, 3] containing the query points
 * @param prev_grid Optional previously created grid to update (if nullptr, creates new)
 * @param num_levels Number of levels in the hierarchy (default: 3)
 * @param base_cell_size Base cell size for the finest level (default: 1.0)
 * @param growth_factor Scale factor between levels (default: 2.0)
 * @return Updated or newly created HierarchicalGrid
 */
HierarchicalGrid *create_or_update_hierarchical_grid(
    torch::Tensor triangles,
    torch::Tensor point_cloud,
    HierarchicalGrid *prev_grid = nullptr,
    int num_levels = 3,
    float base_cell_size = 1.0,
    float growth_factor = 2.0);

/**
 * Finds the nearest triangle for each point in the point cloud using the hierarchical grid.
 *
 * @param point_cloud Tensor of shape [num_points, 3] containing query points
 * @param triangles Tensor of shape [num_triangles, 3, 3] containing triangle vertices
 * @param grid The hierarchical grid structure
 * @return Tensor of shape [num_points] containing indices of the nearest triangles
 */
torch::Tensor find_nearest_triangles_hierarchical(
    torch::Tensor point_cloud,
    torch::Tensor triangles,
    const HierarchicalGrid *grid);

/**
 * Free memory associated with a hierarchical grid.
 *
 * @param grid Pointer to the hierarchical grid to free
 */
void free_hierarchical_grid(HierarchicalGrid *grid);
