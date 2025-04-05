"""
Energy-Based Scene Optimization using Kaolin's GPU-accelerated nearest neighbor search

This module implements the energy-based scene optimization using Kaolin for
GPU-accelerated nearest neighbor search, providing significant performance improvement
over the original CPU-based KD-tree implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm
import kaolin.metrics.pointcloud as kaolin_metrics

# Import our custom geometry utilities instead of PyTorch3D
from geometry_utils import generate_icosphere, SimpleMesh, euler_angles_to_matrix
from geometry_utils.spatial_hash import (
    create_spatial_hash,
    find_nearest_triangle_indices,
)


class EnergyBasedSceneOptimizer:
    def __init__(
        self,
        num_slots=64,
        num_prototypes=10,
        device=None,
        slot_lr=0.01,
        prototype_lr=0.001,
        noise_std=0.0,  # Standard deviation for optional noise in EBM
        ico_level=4,  # Detail level for icosphere
    ):
        """
        Initialize the energy-based scene optimizer.

        Args:
            num_slots: Number of slots (objects) in the scene
            num_prototypes: Number of prototype archetypes to use
            device: Device to use for computation (CPU/GPU)
            slot_lr: Learning rate for slot parameters
            prototype_lr: Learning rate for prototype vertex offsets
            noise_std: Standard deviation for noise to add during optimization (0 = deterministic)
            ico_level: Level of detail for icosphere (higher = more vertices)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_slots = num_slots
        self.num_prototypes = num_prototypes
        self.noise_std = noise_std

        # Create base mesh (icosphere) using our generator instead of PyTorch3D
        base_verts, base_faces = generate_icosphere(level=ico_level, device=device)
        self.faces = base_faces  # [F, 3]
        self.num_verts = base_verts.shape[0]

        # Initialize learnable parameters

        # 1. Slot parameters - for each slot:
        # - Position (x, y, z): 3 parameters
        # - Orientation (yaw, pitch, roll): 3 parameters
        # - Scale (uniform): 1 parameter
        # Total: num_slots × 7 parameters
        self.slot_params = nn.Parameter(torch.zeros(num_slots, 7, device=device))

        # Initialize with reasonable values
        # Positions: random in [-10, 10] meters
        self.slot_params.data[:, :3] = torch.rand(num_slots, 3, device=device) * 20 - 10
        # Orientations: random in [-π/4, π/4] radians
        self.slot_params.data[:, 3:6] = (
            torch.rand(num_slots, 3, device=device) - 0.5
        ) * (torch.pi / 2)
        # Scales: random in [0.5, 2.0]
        self.slot_params.data[:, 6:] = (
            torch.rand(num_slots, 1, device=device) * 1.5 + 0.5
        )

        # 2. Prototype vertex offsets - for each prototype:
        # - Vertex offsets: num_verts × 3 parameters
        # Total: num_prototypes × num_verts × 3 parameters
        self.prototype_offsets = nn.Parameter(
            torch.zeros(num_prototypes, self.num_verts, 3, device=device)
        )

        # 3. Prototype weights - for each slot, weights for all prototypes:
        # - Weight for each prototype: num_prototypes parameters per slot
        # Total: num_slots × num_prototypes parameters
        # Initialize with random values, then apply softmax to ensure proper weighting
        self.prototype_weights_logits = nn.Parameter(
            torch.randn(num_slots, num_prototypes, device=device) * 0.1
        )

        # Store base vertices (not optimized)
        self.base_verts = base_verts

        # Setup optimizer - use Adam for both parameters, with different learning rates
        self.optimizer = optim.Adam(
            [
                {"params": self.slot_params, "lr": slot_lr},
                {"params": self.prototype_offsets, "lr": prototype_lr},
                {
                    "params": self.prototype_weights_logits,
                    "lr": slot_lr,
                },  # Use same lr as slots
            ]
        )

        # For tracking progress
        self.iteration = 0
        self.loss_history = []

    def get_slots(self):
        """
        Generate slot meshes by blending and transforming prototypes.

        Returns:
            slot_meshes: SimpleMesh object containing all slots
            all_triangles: Tensor of all triangle vertices [num_triangles, 3, 3]
            triangle_indices: Tensor mapping triangles to slots [num_triangles]
        """
        # Extract parameters
        positions = self.slot_params[:, :3]  # [num_slots, 3]
        orientations = self.slot_params[:, 3:6]  # [num_slots, 3]
        scales = self.slot_params[:, 6:]  # [num_slots, 1]

        # Get prototype weights using softmax (ensuring they sum to 1 for each slot)
        prototype_weights = torch.softmax(
            self.prototype_weights_logits, dim=1
        )  # [num_slots, num_prototypes]

        # Create a list to store transformed meshes and triangles
        verts_list = []
        faces_list = []
        all_triangles = []
        triangle_indices = []

        # Process each slot
        for s in range(self.num_slots):
            # Compute the weighted sum of prototype vertices
            # Initialize with base vertices
            weighted_verts = self.base_verts.clone()

            # Add weighted offsets from each prototype
            for p in range(self.num_prototypes):
                weight = prototype_weights[s, p]
                # Skip if weight is very small (optimization)
                if weight > 1e-5:
                    weighted_verts = weighted_verts + weight * self.prototype_offsets[p]

            # Apply transformations
            # 1. Scale
            scaled_verts = weighted_verts * scales[s]

            # 2. Rotate - using our custom euler_angles_to_matrix function
            rot_matrix = euler_angles_to_matrix(orientations[s], "XYZ")  # [3, 3]
            rotated_verts = torch.matmul(scaled_verts, rot_matrix.T)  # [V, 3]

            # 3. Translate
            transformed_verts = rotated_verts + positions[s]  # [V, 3]

            # Add to lists
            verts_list.append(transformed_verts)
            faces_list.append(self.faces)

            # Extract triangles for this mesh
            # For each face, get the 3 vertex coordinates
            face_verts = transformed_verts[self.faces]  # [F, 3, 3]
            all_triangles.append(face_verts)

            # Track which triangles belong to which slot
            num_triangles = self.faces.shape[0]
            triangle_indices.append(torch.full((num_triangles,), s, device=self.device))

        # Create mesh objects using our SimpleMesh class
        mesh_list = SimpleMesh(verts=verts_list, faces=faces_list)

        # Combine all triangles and indices
        all_triangles = torch.cat(all_triangles, dim=0)  # [total_triangles, 3, 3]
        triangle_indices = torch.cat(triangle_indices, dim=0)  # [total_triangles]

        return mesh_list, all_triangles, triangle_indices

    def compute_loss(self, point_cloud):
        """
        Compute the L2 squared distance loss using GPU-accelerated spatial hash table.

        Args:
            point_cloud: Tensor of shape [N, 3] - the target point cloud

        Returns:
            loss: Scalar tensor - the L2 squared distance loss
        """
        # Get slot meshes
        _, all_triangles, _ = self.get_slots()

        # Build or update spatial hash tables occasionally
        rebuild_interval = 10  # Rebuild the spatial hash every 10 iterations
        if (
            not hasattr(self, "spatial_hash_tables")
            or self.iteration % rebuild_interval == 0
        ):
            # Create spatial hash with 3 levels (fine, medium, coarse)
            self.spatial_hash_tables = create_spatial_hash(
                all_triangles, point_cloud, max_level=3, min_cell_size=1.0
            )

        # Find nearest triangles using spatial hash
        triangle_indices = find_nearest_triangle_indices(
            point_cloud, all_triangles, self.spatial_hash_tables
        )

        # Get corresponding triangles
        closest_triangles = all_triangles[triangle_indices]  # [N, 3, 3]

        # Compute centroids of closest triangles
        closest_centroids = closest_triangles.mean(dim=1)  # [N, 3]

        # Compute L2 squared distances
        squared_dists = torch.sum((point_cloud - closest_centroids) ** 2, dim=1)  # [N]

        # Compute mean loss
        loss = squared_dists.mean()

        return loss

    def optimize_step(self, point_cloud):
        """
        Perform one optimization step.

        Args:
            point_cloud: Tensor of shape [N, 3] - the target point cloud

        Returns:
            loss: The loss for this step
        """
        # Reset gradients
        self.optimizer.zero_grad()

        # Compute loss
        loss = self.compute_loss(point_cloud)

        # Backpropagate
        loss.backward()

        # Update parameters
        self.optimizer.step()

        # Add noise if using EBM approach
        if self.noise_std > 0:
            with torch.no_grad():
                self.slot_params.add_(
                    torch.randn_like(self.slot_params) * self.noise_std
                )
                self.prototype_offsets.add_(
                    torch.randn_like(self.prototype_offsets) * self.noise_std * 0.1
                )

        # Track progress
        self.iteration += 1
        self.loss_history.append(loss.item())

        return loss.item()

    def convert_transforms_for_viz(self):
        """
        Convert internal slot parameters to a format compatible with the visualization exporter.

        Returns:
            transforms: Tensor of shape [1, num_slots, num_prototypes, 6] for visualization
        """
        # Create a batch dimension for slot parameters
        slot_params = self.slot_params.unsqueeze(0)  # [1, num_slots, 7]
        device = slot_params.device

        # Create transforms tensor with the right shape for all prototypes
        # The format expects [B, num_slots, num_prototypes, 6]
        # where 6 = (x, y, z, yaw, pitch, roll)
        transforms = torch.zeros(
            1, self.num_slots, self.num_prototypes, 6, device=device
        )

        # Fill in the values - for now, we replicate the same transform for all prototypes
        # but weight them differently via prototype_weights
        for p in range(self.num_prototypes):
            # Positions (x, y, z)
            transforms[:, :, p, :3] = slot_params[:, :, :3]
            # Rotations (yaw, pitch, roll)
            transforms[:, :, p, 3:] = slot_params[:, :, 3:6]

        return transforms

    def get_prototype_weights(self):
        """
        Get the softmax-normalized prototype weights.

        Returns:
            prototype_weights: Tensor of shape [1, num_slots, num_prototypes]
        """
        # Apply softmax to get normalized weights and add batch dimension
        weights = torch.softmax(self.prototype_weights_logits, dim=1).unsqueeze(0)
        return weights

    def optimize(self, point_cloud, num_iterations=1000, callback=None):
        """
        Run the optimization process for a given number of iterations.

        Args:
            point_cloud: Tensor of shape [N, 3] - the target point cloud
            num_iterations: Number of iterations to run
            callback: Optional callback function for visualization

        Returns:
            loss_history: List of loss values during optimization
        """
        # Ensure point cloud is on the correct device
        point_cloud = point_cloud.to(self.device)

        # Reset iteration counter
        start_iteration = self.iteration

        # Create progress bar
        progress_bar = tqdm(range(num_iterations), desc="Energy-Based Optimization")

        for i in progress_bar:
            # Perform optimization step
            loss = self.optimize_step(point_cloud)

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss:.6f}"})

            # Call callback if provided (e.g., for visualization)
            if callback is not None:
                callback(self, point_cloud, i + start_iteration)

        return self.loss_history[-num_iterations:]


# End of module
