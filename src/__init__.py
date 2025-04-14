"""
Proto-Depth: A system for learning 3D scene representations from depth maps
"""

from .scene import Scene
from .point_cloud import render_depth_and_pointcloud
from .utils import transform_vertices

__all__ = [
    "Scene",
    "render_depth_and_pointcloud",
    "transform_vertices",
]
