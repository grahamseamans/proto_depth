"""
Mesh utilities implemented in pure PyTorch

This module provides a simple mesh container and utilities for working with 3D meshes
without any dependency on PyTorch3D.
"""

import torch
import numpy as np


class SimpleMesh:
    """
    A lightweight mesh container that provides essential functionality
    similar to PyTorch3D's Meshes class but without the dependency.
    """

    def __init__(self, verts, faces):
        """
        Initialize a mesh from vertices and faces.

        Args:
            verts: List of tensors of shape [V, 3] or a single tensor of shape [V, 3]
            faces: List of tensors of shape [F, 3] or a single tensor of shape [F, 3]
        """
        # Handle both batched and non-batched inputs
        if isinstance(verts, list):
            self._verts_list = verts
            self._faces_list = faces
            self.is_batched = True
        else:
            self._verts_list = [verts]
            self._faces_list = [faces]
            self.is_batched = False

        # Check that all tensors have the right shape
        for v, f in zip(self._verts_list, self._faces_list):
            assert v.ndim == 2 and v.shape[1] == 3, (
                f"Vertices must have shape [V, 3], got {v.shape}"
            )
            assert f.ndim == 2 and f.shape[1] == 3, (
                f"Faces must have shape [F, 3], got {f.shape}"
            )

        # Cache device and dtype
        self.device = self._verts_list[0].device
        self.dtype = self._verts_list[0].dtype

    def __getitem__(self, idx):
        """
        Make the mesh subscriptable, returning a new SimpleMesh with single mesh.

        Args:
            idx: Integer index or slice

        Returns:
            SimpleMesh: A new SimpleMesh with the selected mesh(es)
        """
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self._verts_list):
                raise IndexError(
                    f"Index {idx} out of range for SimpleMesh with {len(self._verts_list)} meshes"
                )
            return SimpleMesh(verts=self._verts_list[idx], faces=self._faces_list[idx])
        elif isinstance(idx, slice):
            return SimpleMesh(verts=self._verts_list[idx], faces=self._faces_list[idx])
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def verts_list(self):
        """
        Return the list of vertex tensors.

        This method is provided for compatibility with PyTorch3D's Meshes API.

        Returns:
            List[torch.Tensor]: List of vertex tensors
        """
        return self._verts_list

    def faces_list(self):
        """
        Return the list of face tensors.

        This method is provided for compatibility with PyTorch3D's Meshes API.

        Returns:
            List[torch.Tensor]: List of face tensors
        """
        return self._faces_list

    def verts_padded(self):
        """
        Return vertices as a padded tensor of shape [B, V, 3].

        Returns:
            torch.Tensor: Padded vertices tensor
        """
        if not self.is_batched:
            return self._verts_list[0].unsqueeze(0)

        # Find maximum number of vertices
        max_verts = max(v.shape[0] for v in self._verts_list)

        # Pad each tensor
        padded_verts = []
        for verts in self._verts_list:
            if verts.shape[0] < max_verts:
                padding = torch.zeros(
                    (max_verts - verts.shape[0], 3),
                    dtype=self.dtype,
                    device=self.device,
                )
                padded = torch.cat([verts, padding], dim=0)
            else:
                padded = verts
            padded_verts.append(padded)

        return torch.stack(padded_verts)

    def faces_padded(self):
        """
        Return faces as a padded tensor of shape [B, F, 3].

        Returns:
            torch.Tensor: Padded faces tensor
        """
        if not self.is_batched:
            return self._faces_list[0].unsqueeze(0)

        # Find maximum number of faces
        max_faces = max(f.shape[0] for f in self._faces_list)

        # Pad each tensor
        padded_faces = []
        for i, faces in enumerate(self._faces_list):
            if faces.shape[0] < max_faces:
                padding = torch.zeros(
                    (max_faces - faces.shape[0], 3),
                    dtype=faces.dtype,
                    device=faces.device,
                )
                padded = torch.cat([faces, padding], dim=0)
            else:
                padded = faces
            padded_faces.append(padded)

        return torch.stack(padded_faces)

    def num_verts_per_mesh(self):
        """
        Return the number of vertices in each mesh.

        Returns:
            torch.Tensor: Tensor of shape [B] with number of vertices
        """
        return torch.tensor([v.shape[0] for v in self._verts_list], device=self.device)

    def num_faces_per_mesh(self):
        """
        Return the number of faces in each mesh.

        Returns:
            torch.Tensor: Tensor of shape [B] with number of faces
        """
        return torch.tensor([f.shape[0] for f in self._faces_list], device=self.device)

    def face_normals(self):
        """
        Compute face normals for each mesh.

        Returns:
            List of torch.Tensor: Face normals for each mesh
        """
        normals_list = []

        for verts, faces in zip(self._verts_list, self._faces_list):
            # Get vertices of each face
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]

            # Compute face normals using cross product
            e1 = v1 - v0
            e2 = v2 - v0
            face_normals = torch.cross(e1, e2, dim=1)

            # Normalize
            face_normals = face_normals / (
                torch.norm(face_normals, dim=1, keepdim=True) + 1e-10
            )
            normals_list.append(face_normals)

        return normals_list

    def vertex_normals(self):
        """
        Compute vertex normals for each mesh by averaging face normals.

        Returns:
            List of torch.Tensor: Vertex normals for each mesh
        """
        vertex_normals_list = []

        for verts, faces, face_normals in zip(
            self._verts_list, self._faces_list, self.face_normals()
        ):
            # Initialize vertex normals
            vertex_normals = torch.zeros_like(verts)

            # For each face, add its normal to all vertices
            for i in range(3):
                vertex_indices = faces[:, i]
                vertex_normals.index_add_(0, vertex_indices, face_normals)

            # Normalize
            vertex_normals = vertex_normals / (
                torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-10
            )
            vertex_normals_list.append(vertex_normals)

        return vertex_normals_list


def _compute_face_areas(verts, faces):
    """
    Compute areas of faces in the mesh.

    Args:
        verts: Tensor of shape [V, 3] with vertex positions
        faces: Tensor of shape [F, 3] with face indices

    Returns:
        torch.Tensor: Tensor of shape [F] with face areas
    """
    # Get vertices of each face
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Compute face normals using cross product
    e1 = v1 - v0
    e2 = v2 - v0
    face_normals = torch.cross(e1, e2, dim=1)

    # Area is half the norm of the cross product
    return torch.norm(face_normals, dim=1) * 0.5


def sample_points_from_mesh(mesh, num_samples, return_normals=False):
    """
    Sample points from mesh surfaces.

    Args:
        mesh: SimpleMesh object
        num_samples: Number of points to sample per mesh
        return_normals: Whether to return normals with the points

    Returns:
        torch.Tensor or tuple: Sampled points tensor [B, S, 3] or tuple with points and normals
    """
    device = mesh.device

    sampled_points_list = []
    sampled_normals_list = []

    # Compute face normals if needed
    face_normals_list = (
        mesh.face_normals() if return_normals else [None] * len(mesh.verts_list())
    )

    for verts, faces, face_normals in zip(
        mesh.verts_list(), mesh.faces_list(), face_normals_list
    ):
        # Compute face areas for weighted sampling
        face_areas = _compute_face_areas(verts, faces)

        # Convert to probabilities
        face_probs = face_areas / face_areas.sum()

        # Sample faces based on area
        face_indices = torch.multinomial(face_probs, num_samples, replacement=True)

        # Get selected faces
        selected_faces = faces[face_indices]

        # Sample random barycentric coordinates
        u = torch.sqrt(torch.rand(num_samples, device=device))
        v = torch.rand(num_samples, device=device)
        barycentric = torch.stack([1 - u, u * (1 - v), u * v], dim=1)

        # Get vertices of selected faces
        face_verts = verts[selected_faces]

        # Compute points from barycentric coordinates
        points = (barycentric.unsqueeze(1) * face_verts).sum(dim=1)
        sampled_points_list.append(points)

        # Compute normals if needed
        if return_normals:
            sampled_normals = face_normals[face_indices]
            sampled_normals_list.append(sampled_normals)

    # Stack results
    points = torch.stack(sampled_points_list)

    if return_normals:
        normals = torch.stack(sampled_normals_list)
        return points, normals

    return points
