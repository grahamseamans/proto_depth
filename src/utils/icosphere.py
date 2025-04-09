"""
Icosphere generation utilities implemented in pure PyTorch

This module provides functions to generate icospheres (subdivided
icosahedrons) without any dependency on PyTorch3D.
"""

import torch
import math


class IcoSphere:
    """
    A class for generating icosphere meshes with variable levels of detail.

    This class encapsulates the state and behavior for creating and refining
    icosphere meshes, providing a more robust approach than pure functions.
    """

    def __init__(self, level=4, device=None):
        """
        Initialize the icosphere.

        Args:
            level: Subdivision level (0 = icosahedron, each level subdivides faces)
            device: PyTorch device
        """
        self.device = device

        # Create base icosahedron
        self.vertices, self.faces = self._create_base_icosahedron()

        # Move to specified device
        if device is not None:
            self.vertices = self.vertices.to(device)
            self.faces = self.faces.to(device)

        # Subdivide mesh based on level
        for _ in range(level):
            self._subdivide_mesh()

    def _create_base_icosahedron(self):
        """
        Create the base icosahedron (level 0).

        Returns:
            vertices: Tensor of shape [12, 3]
            faces: Tensor of shape [20, 3]
        """
        # Golden ratio for icosahedron construction
        phi = (1 + math.sqrt(5)) / 2

        # Icosahedron vertices
        vertices = torch.tensor(
            [
                [-1, phi, 0],
                [1, phi, 0],
                [-1, -phi, 0],
                [1, -phi, 0],
                [0, -1, phi],
                [0, 1, phi],
                [0, -1, -phi],
                [0, 1, -phi],
                [phi, 0, -1],
                [phi, 0, 1],
                [-phi, 0, -1],
                [-phi, 0, 1],
            ],
            dtype=torch.float32,
        )

        # Normalize to unit sphere
        vertices = vertices / torch.norm(vertices, dim=1, keepdim=True)

        # Icosahedron faces
        faces = torch.tensor(
            [
                [0, 11, 5],
                [0, 5, 1],
                [0, 1, 7],
                [0, 7, 10],
                [0, 10, 11],
                [1, 5, 9],
                [5, 11, 4],
                [11, 10, 2],
                [10, 7, 6],
                [7, 1, 8],
                [3, 9, 4],
                [3, 4, 2],
                [3, 2, 6],
                [3, 6, 8],
                [3, 8, 9],
                [4, 9, 5],
                [2, 4, 11],
                [6, 2, 10],
                [8, 6, 7],
                [9, 8, 1],
            ],
            dtype=torch.long,
        )

        return vertices, faces

    def _subdivide_mesh(self):
        """
        Subdivide each triangle in the current mesh into 4 triangles.
        Updates the internal vertices and faces.
        """
        # Create dict to track midpoints - key: (v1_idx, v2_idx), value: new_vertex_idx
        edge_midpoints = {}

        # New faces will replace existing ones (4 for each original face)
        new_faces = []

        # For each face, create 4 new faces by adding vertices at the midpoints
        for face in self.faces:
            # Get indices of the three vertices
            v1, v2, v3 = face.tolist()

            # Get midpoint indices (creating new vertices as needed)
            m1 = self._get_midpoint_index(edge_midpoints, v1, v2)
            m2 = self._get_midpoint_index(edge_midpoints, v2, v3)
            m3 = self._get_midpoint_index(edge_midpoints, v3, v1)

            # Create 4 new faces
            new_faces.append(torch.tensor([v1, m1, m3], device=self.device))
            new_faces.append(torch.tensor([v2, m2, m1], device=self.device))
            new_faces.append(torch.tensor([v3, m3, m2], device=self.device))
            new_faces.append(torch.tensor([m1, m2, m3], device=self.device))

        # Update faces to the new subdivided faces
        self.faces = torch.stack(new_faces)

    def _get_midpoint_index(self, edge_midpoints, idx1, idx2):
        """
        Get the index of the midpoint between two vertices.
        If the midpoint doesn't exist, create it and add to the vertices.

        Args:
            edge_midpoints: Dict mapping edge keys to midpoint indices
            idx1, idx2: Indices of the two vertices

        Returns:
            Index of the midpoint vertex
        """
        # Ensure idx1 < idx2 for consistent edge keys
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        # Check if this edge's midpoint already exists
        edge_key = (idx1, idx2)
        if edge_key in edge_midpoints:
            return edge_midpoints[edge_key]

        # Calculate midpoint (normalized to lie on unit sphere)
        v1 = self.vertices[idx1]
        v2 = self.vertices[idx2]
        midpoint = (v1 + v2) / 2.0
        midpoint = midpoint / torch.norm(midpoint)

        # Add midpoint to vertices
        midpoint_idx = len(self.vertices)
        self.vertices = torch.cat([self.vertices, midpoint.unsqueeze(0)], dim=0)

        # Store in dictionary
        edge_midpoints[edge_key] = midpoint_idx

        return midpoint_idx


def generate_icosphere(level=4, device=None):
    """
    Generate an icosphere mesh without PyTorch3D dependency.

    This function provides backward compatibility with the original API,
    but uses the more robust IcoSphere class internally.

    Args:
        level: Subdivision level (0 = icosahedron, each level subdivides faces)
        device: PyTorch device

    Returns:
        vertices: Tensor of shape [V, 3]
        faces: Tensor of shape [F, 3]
    """
    icosphere = IcoSphere(level=level, device=device)
    return icosphere.vertices, icosphere.faces
