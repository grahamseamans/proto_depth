"""
Proto-Depth: A system for learning 3D scene representations from depth maps
"""

from .scene import Scene
from .point_cloud import depth_to_pointcloud

__all__ = ["Scene", "depth_to_pointcloud"]
