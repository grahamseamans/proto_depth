"""
Geometry Utilities without PyTorch3D dependency

This package provides essential 3D geometry utilities implemented
in pure PyTorch, with no dependencies on PyTorch3D.
"""

from .icosphere import generate_icosphere
from .mesh import SimpleMesh, sample_points_from_mesh
from .transforms import euler_angles_to_matrix

__all__ = [
    "generate_icosphere",
    "SimpleMesh",
    "sample_points_from_mesh",
    "euler_angles_to_matrix",
]
