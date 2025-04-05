import os
import sys
import subprocess
import torch

try:
    from .spatial_hash import build_spatial_hash_tables, find_nearest_triangles
except ImportError:
    print("Spatial hash extension not found, attempting to build it...")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        build_ext_cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
        subprocess.check_call(build_ext_cmd, cwd=current_dir)
        from .spatial_hash import build_spatial_hash_tables, find_nearest_triangles
    except Exception as e:
        print(f"Failed to build spatial hash extension: {e}")
        raise


def create_spatial_hash(triangles, point_cloud, max_level=3, min_cell_size=1.0):
    """
    Creates spatial hash tables for nearest triangle lookup using memory-efficient approach.

    Args:
        triangles (torch.Tensor): Triangles tensor of shape [num_triangles, 3, 3]
        point_cloud (torch.Tensor): Points tensor of shape [num_points, 3]
        max_level (int): Maximum hierarchy level (default: 3)
        min_cell_size (float): Cell size for finest level (default: 1.0)

    Returns:
        list: List of dictionaries representing spatial hash levels, with keys:
            - 'cell_counts': Tensor counting triangles per cell
            - 'cell_offsets': Tensor of offsets into triangle_indices
            - 'triangle_indices': Tensor of triangle indices
            - 'hash_table_size': Size of the hash table
    """
    return build_spatial_hash_tables(triangles, point_cloud, max_level, min_cell_size)


def find_nearest_triangle_indices(point_cloud, triangles, hash_levels):
    """
    Finds the nearest triangle for each point using spatial hash tables.

    Args:
        point_cloud (torch.Tensor): Points tensor of shape [num_points, 3]
        triangles (torch.Tensor): Triangles tensor of shape [num_triangles, 3, 3]
        hash_levels (list): List of hash level dictionaries from create_spatial_hash

    Returns:
        torch.Tensor: Tensor of indices to the nearest triangles, shape [num_points]
    """
    return find_nearest_triangles(point_cloud, triangles, hash_levels)
