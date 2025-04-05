#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "spatial_hash_cuda.h"

namespace py = pybind11;

// Helper function to convert SpatialHashLevel to Python dict
py::dict convert_hash_level_to_dict(const SpatialHashLevel &level)
{
      py::dict result;
      result["cell_counts"] = level.cell_counts;
      result["cell_offsets"] = level.cell_offsets;
      result["triangle_indices"] = level.triangle_indices;
      result["hash_table_size"] = level.hash_table_size;
      return result;
}

// Helper function to convert Python dict to SpatialHashLevel
SpatialHashLevel convert_dict_to_hash_level(const py::dict &dict)
{
      SpatialHashLevel level;
      level.cell_counts = dict["cell_counts"].cast<torch::Tensor>();
      level.cell_offsets = dict["cell_offsets"].cast<torch::Tensor>();
      level.triangle_indices = dict["triangle_indices"].cast<torch::Tensor>();
      level.hash_table_size = dict["hash_table_size"].cast<int>();
      return level;
}

// Build hash tables and convert to Python list of dicts
py::list build_spatial_hash(
    torch::Tensor triangles,
    torch::Tensor point_cloud,
    int max_level,
    float min_cell_size)
{
      auto hash_levels = build_spatial_hash_tables(triangles, point_cloud, max_level, min_cell_size);

      py::list result;
      for (const auto &level : hash_levels)
      {
            result.append(convert_hash_level_to_dict(level));
      }

      return result;
}

// Convert Python list of dicts to SpatialHashLevel and find nearest triangles
torch::Tensor find_nearest_triangles_from_hash(
    torch::Tensor point_cloud,
    torch::Tensor triangles,
    const py::list &hash_levels_list)
{
      std::vector<SpatialHashLevel> hash_levels;

      for (const auto &item : hash_levels_list)
      {
            auto dict = item.cast<py::dict>();
            hash_levels.push_back(convert_dict_to_hash_level(dict));
      }

      return find_nearest_triangles(point_cloud, triangles, hash_levels);
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("build_spatial_hash_tables", &build_spatial_hash,
            "Build spatial hash tables for multiple levels",
            py::arg("triangles"),
            py::arg("point_cloud"),
            py::arg("max_level") = 3,
            py::arg("min_cell_size") = 1.0);

      m.def("find_nearest_triangles", &find_nearest_triangles_from_hash,
            "Find nearest triangle for each point in the point cloud",
            py::arg("point_cloud"),
            py::arg("triangles"),
            py::arg("hash_levels"));
}
