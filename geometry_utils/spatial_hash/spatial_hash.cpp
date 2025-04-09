#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "spatial_hash_cuda.h"

namespace py = pybind11;

// Wrapper for HierarchicalGrid object management
class PyHierarchicalGrid
{
private:
      HierarchicalGrid *grid;

public:
      PyHierarchicalGrid() : grid(nullptr) {}

      ~PyHierarchicalGrid()
      {
            if (grid)
            {
                  free_hierarchical_grid(grid);
                  grid = nullptr;
            }
      }

      void create_or_update(
          torch::Tensor triangles,
          torch::Tensor point_cloud,
          int num_levels = 3,
          float base_cell_size = 1.0,
          float growth_factor = 2.0)
      {
            grid = create_or_update_hierarchical_grid(
                triangles,
                point_cloud,
                grid,
                num_levels,
                base_cell_size,
                growth_factor);
      }

      torch::Tensor find_nearest_triangles(
          torch::Tensor point_cloud,
          torch::Tensor triangles)
      {
            if (!grid)
            {
                  throw std::runtime_error("Hierarchical grid not initialized. Call create_or_update first.");
            }

            return find_nearest_triangles_hierarchical(
                point_cloud,
                triangles,
                grid);
      }

      py::dict get_stats()
      {
            if (!grid)
            {
                  throw std::runtime_error("Hierarchical grid not initialized.");
            }

            py::dict stats;
            stats["num_levels"] = grid->num_levels;
            stats["base_cell_size"] = grid->base_cell_size;
            stats["growth_factor"] = grid->growth_factor;
            stats["num_triangles"] = grid->num_triangles;

            py::list level_stats;
            for (int i = 0; i < grid->num_levels; i++)
            {
                  py::dict level_dict;
                  level_dict["grid_size"] = grid->levels[i].grid_size;
                  level_dict["cell_size"] = grid->levels[i].cell_size;
                  level_dict["num_non_empty_bins"] =
                      torch::nonzero(grid->levels[i].bin_counts).size(0);
                  level_stats.append(level_dict);
            }
            stats["levels"] = level_stats;

            return stats;
      }
};

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      py::class_<PyHierarchicalGrid>(m, "HierarchicalGrid")
          .def(py::init<>())
          .def("create_or_update", &PyHierarchicalGrid::create_or_update,
               "Create or update the hierarchical grid",
               py::arg("triangles"),
               py::arg("point_cloud"),
               py::arg("num_levels") = 3,
               py::arg("base_cell_size") = 1.0,
               py::arg("growth_factor") = 2.0)
          .def("find_nearest_triangles", &PyHierarchicalGrid::find_nearest_triangles,
               "Find nearest triangle for each point in the point cloud",
               py::arg("point_cloud"),
               py::arg("triangles"))
          .def("get_stats", &PyHierarchicalGrid::get_stats,
               "Get statistics about the hierarchical grid");
}
