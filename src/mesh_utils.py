import torch
import torch.nn as nn
import pytorch3d.transforms as t3d
import pytorch3d.loss
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils import ico_sphere
import pytorch3d.ops


class MeshTransformer(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Create unit sphere prototype (doesn't get backpropped)
        sphere = ico_sphere(level=4, device=device)
        verts = sphere.verts_padded().detach()  # [1, V, 3]
        faces = sphere.faces_padded().detach()  # [1, F, 3]
        self.sphere_mesh = Meshes(verts=verts, faces=faces)

        # Store the number of vertices for later use
        self.num_verts = verts.shape[1]

    def transform_mesh(self, scales, transforms, prototype_weights, prototype_offsets):
        """
        Args:
            scales: Tensor of shape [B, num_slots, 1]
            transforms: Tensor of shape [B, num_slots, num_prototypes, 6] (x,y,z, yaw,pitch,roll)
            prototype_weights: Tensor of shape [B, num_slots, num_prototypes]
            prototype_offsets: Tensor of shape [num_prototypes, num_verts, 3]
        Returns:
            transformed_meshes: List of B Meshes objects, each containing num_slots meshes
        """
        B, num_slots, num_prototypes = prototype_weights.shape
        num_verts = self.num_verts

        # Get base vertices and faces (no gradients)
        with torch.no_grad():
            verts = self.sphere_mesh.verts_padded()  # [1, V, 3]
            faces = self.sphere_mesh.faces_padded()  # [1, F, 3]

        # Reshape inputs for vectorized operations
        # [B*num_slots, num_prototypes, 1]
        weights = prototype_weights.reshape(B * num_slots, num_prototypes, 1)

        # Create a batch dimension for prototype_offsets: [1, num_prototypes, num_verts, 3]
        # Then expand to [B*num_slots, num_prototypes, num_verts, 3]
        offsets = prototype_offsets.unsqueeze(0).expand(B * num_slots, -1, -1, -1)

        # Reshape transforms to [B*num_slots, num_prototypes, 6]
        transforms_flat = transforms.reshape(B * num_slots, num_prototypes, 6)

        # Reshape scales to [B*num_slots, 1, 1]
        scales_flat = scales.reshape(B * num_slots, 1, 1)

        # Get the base vertices for each slot
        # Expand to [B*num_slots, num_prototypes, V, 3]
        base_verts = (
            verts.expand(B * num_slots, -1, -1)
            .unsqueeze(1)
            .expand(-1, num_prototypes, -1, -1)
        )

        # Apply offsets to create deformed shapes
        # [B*num_slots, num_prototypes, V, 3]
        deformed_verts = base_verts + offsets

        # Extract translation and rotation from transforms
        translations = transforms_flat[:, :, :3].unsqueeze(
            2
        )  # [B*num_slots, num_prototypes, 1, 3]
        rotations = transforms_flat[:, :, 3:]  # [B*num_slots, num_prototypes, 3]

        # Vectorized rotation matrices computation
        # First reshape rotations to [B*num_slots*num_prototypes, 3]
        rotations_flat = rotations.reshape(-1, 3)

        # Compute rotation matrices [B*num_slots*num_prototypes, 3, 3]
        rot_matrices = t3d.euler_angles_to_matrix(rotations_flat, "XYZ")

        # Reshape back to [B*num_slots, num_prototypes, 3, 3]
        rot_matrices = rot_matrices.reshape(B * num_slots, num_prototypes, 3, 3)

        # Apply rotations using batch matrix multiplication
        # [B*num_slots, num_prototypes, V, 3] × [B*num_slots, num_prototypes, 3, 3]
        # First reshape deformed_verts to [B*num_slots*num_prototypes, V, 3]
        deformed_flat = deformed_verts.reshape(-1, num_verts, 3)
        rot_matrices_flat = rot_matrices.reshape(-1, 3, 3)

        # Batch matrix multiply
        rotated_flat = torch.bmm(deformed_flat, rot_matrices_flat.transpose(1, 2))

        # Reshape back to [B*num_slots, num_prototypes, V, 3]
        rotated = rotated_flat.reshape(B * num_slots, num_prototypes, num_verts, 3)

        # Apply scaling (broadcasting)
        scaled = rotated * scales_flat.unsqueeze(-1)

        # Apply translation (broadcasting)
        translated = scaled + translations

        # Weight and sum across prototype dimension
        # [B*num_slots, num_prototypes, V, 3] × [B*num_slots, num_prototypes, 1, 1]
        # Result: [B*num_slots, V, 3]
        weighted_sum = (translated * weights.unsqueeze(-1)).sum(dim=1)

        # Reshape to [B, num_slots, V, 3]
        final_verts = weighted_sum.reshape(B, num_slots, num_verts, 3)

        # Create mesh objects
        meshes = []
        for b in range(B):
            # Extract faces for this batch (we still need to create a list per slot)
            b_faces = faces.expand(num_slots, -1, -1)

            # Create mesh for this batch
            mesh = Meshes(verts=list(final_verts[b]), faces=list(b_faces))
            meshes.append(mesh)

        return meshes

    def compute_chamfer_loss(self, meshes, target_pcls):
        """
        Args:
            meshes: List of B Meshes objects, each containing num_slots meshes
            target_pcls: List of B Tensor objects, each of shape [N_i, 3] - target point clouds
        Returns:
            loss: Scalar tensor
        """
        # Use our new hybrid loss with default weights
        total_loss, global_loss, per_slot_loss = self.compute_hybrid_chamfer_loss(
            meshes, target_pcls, global_weight=0.7, slot_weight=0.3
        )
        return total_loss

    def compute_hybrid_chamfer_loss(
        self,
        meshes,
        target_pcls,
        global_weight=0.7,
        slot_weight=0.3,
        repulsion_weight=0.2,
        samples_per_slot=500,
        min_distance=0.5,
        k_nearest=3,  # New parameter for KNN-based approach
    ):
        """
        Optimized implementation of hybrid loss with three components:
        1. Global chamfer: Target points → All predicted mesh points
        2. Per-slot proximity: Each slot's points → Target points (vectorized)
        3. Centroid repulsion: Preventing slots from overlapping

        Args:
            meshes: List of B Meshes objects, each containing num_slots meshes
            target_pcls: List of B Tensor objects, each of shape [N_i, 3] - target point clouds
            global_weight: Weight for the global component
            slot_weight: Weight for the per-slot component
            repulsion_weight: Weight for the centroid repulsion component
            samples_per_slot: Number of points to sample per slot (much lower than before)
            min_distance: Minimum desired distance between centroids (in meters)
            k_nearest: Number of nearest neighbors to consider for KNN-based approach
        Returns:
            total_loss: Combined loss scalar tensor
            global_loss: Global component scalar tensor
            per_slot_loss: Per-slot component scalar tensor
        """
        B = len(meshes)
        global_loss = 0
        per_slot_loss = 0
        repulsion_loss = 0

        # Process each batch
        for b in range(B):
            # Sample fewer points from all meshes in this batch
            pred_pcl_per_slot = pytorch3d.ops.sample_points_from_meshes(
                meshes[b], num_samples=samples_per_slot
            )  # [num_slots, samples_per_slot, 3]
            num_slots = pred_pcl_per_slot.shape[0]
            tgt_pcl = target_pcls[b]  # [N, 3]

            # 1. Global loss: Target points → All predicted points
            # ---------------------------------------------------
            # Flatten all slot points
            pred_pcl_flat = pred_pcl_per_slot.reshape(
                -1, 3
            )  # [num_slots*samples_per_slot, 3]

            # Use more efficient KNN-based approach for Chamfer distance
            # Target → Prediction: For each target point, find k nearest predicted points
            target_to_pred_dists, _ = pytorch3d.ops.knn_points(
                tgt_pcl.unsqueeze(0),  # [1, N, 3]
                pred_pcl_flat.unsqueeze(0),  # [1, num_slots*samples_per_slot, 3]
                K=k_nearest,
            )
            # Mean distance to k nearest neighbors for each point, then mean across all points
            global_dist = target_to_pred_dists.mean()
            global_loss += global_dist

            # 2. Per-slot proximity: ALL slot points → Target points (vectorized)
            # -------------------------------------------------------------------
            # Stack all slots to a single batch dimension
            # [num_slots, samples_per_slot, 3] → [num_slots, 1, samples_per_slot, 3]
            slots_batch = pred_pcl_per_slot.unsqueeze(1)
            # Prepare target for batch comparison
            # [N, 3] → [1, N, 3] → [num_slots, 1, N, 3]
            tgt_batch = tgt_pcl.unsqueeze(0).unsqueeze(0).expand(num_slots, -1, -1, -1)

            # Compute slot → target distances for all slots at once
            # For each slot's points, find nearest target points (vectorized)
            slot_to_target_dists, _ = pytorch3d.ops.knn_points(
                slots_batch.reshape(
                    num_slots, samples_per_slot, 3
                ),  # [num_slots, samples_per_slot, 3]
                tgt_batch.reshape(num_slots, tgt_pcl.shape[0], 3),  # [num_slots, N, 3]
                K=1,
            )

            # Mean over points dimension, then mean over slots
            slot_losses = slot_to_target_dists.squeeze(-1).mean(dim=1)  # [num_slots]
            batch_slot_loss = slot_losses.mean()
            per_slot_loss += batch_slot_loss

            # 3. Centroid repulsion: Prevent slots from overlapping (already vectorized)
            # ------------------------------------------------------------------------
            # Get vertices for each slot in this batch
            verts_list = meshes[
                b
            ].verts_list()  # List of num_slots tensors of shape [V, 3]

            # Compute centroid for each slot
            centroids = torch.stack(
                [verts.mean(dim=0) for verts in verts_list]
            )  # [num_slots, 3]

            # Compute pairwise distances between centroids
            distances = torch.cdist(centroids, centroids, p=2)  # [num_slots, num_slots]

            # Create mask to ignore self-distances
            mask = 1.0 - torch.eye(num_slots, device=distances.device)

            # Apply smooth thresholding function that creates repulsion when closer than min_distance
            distance_diff = min_distance - distances
            falloff_rate = 5.0  # Controls how quickly repulsion falls off
            repulsion = torch.exp(falloff_rate * torch.clamp(distance_diff, min=0.0))
            # Zero out self-repulsion
            repulsion = repulsion * mask

            # Sum all pairwise repulsions and normalize
            if num_slots > 1:  # Avoid division by zero if only one slot
                batch_repulsion_loss = repulsion.sum() / (num_slots * (num_slots - 1))
                repulsion_loss += batch_repulsion_loss

        # Average across batch
        global_loss = global_loss / B
        per_slot_loss = per_slot_loss / B
        repulsion_loss = repulsion_loss / B

        # Combine all components with weights
        total_loss = (
            global_weight * global_loss
            + slot_weight * per_slot_loss
            + repulsion_weight * repulsion_loss
        )

        return total_loss, global_loss, per_slot_loss

    def compute_prototype_regularization(self, prototype_offsets):
        """
        Compute regularization losses for the prototype meshes before any transformation.
        This ensures the base shapes remain smooth and well-formed.

        Args:
            prototype_offsets: Tensor of shape [num_prototypes, num_verts, 3]
        Returns:
            edge_loss: Edge length regularization loss
            normal_loss: Normal consistency loss
            laplacian_loss: Laplacian smoothing loss
        """
        num_prototypes = prototype_offsets.shape[0]

        # Get the base sphere mesh (no gradients)
        with torch.no_grad():
            base_verts = self.sphere_mesh.verts_padded()  # [1, V, 3]
            base_faces = self.sphere_mesh.faces_padded()  # [1, F, 3]

        # Initialize losses
        proto_edge_loss = 0
        proto_normal_loss = 0
        proto_laplacian_loss = 0

        # For each prototype, apply offsets and compute regularization
        for p in range(num_prototypes):
            # Apply offsets to create the prototype mesh
            proto_verts = base_verts + prototype_offsets[p].unsqueeze(0)
            proto_mesh = Meshes(verts=proto_verts, faces=base_faces)

            # Compute regularization losses for this prototype
            edge_loss = pytorch3d.loss.mesh_edge_loss(proto_mesh)
            normal_loss = pytorch3d.loss.mesh_normal_consistency(proto_mesh)
            laplacian_loss = pytorch3d.loss.mesh_laplacian_smoothing(
                proto_mesh, method="uniform"
            )

            # Add to total losses
            proto_edge_loss += edge_loss
            proto_normal_loss += normal_loss
            proto_laplacian_loss += laplacian_loss

        # Average over all prototypes
        return (
            proto_edge_loss / num_prototypes,
            proto_normal_loss / num_prototypes,
            proto_laplacian_loss / num_prototypes,
        )

    # Keeping this for backward compatibility, but returning zeros
    def compute_regularization_losses(self, meshes):
        """
        This function is maintained for backward compatibility but now returns zeros
        as we're moving regularization to the prototype level instead.

        Args:
            meshes: List of B Meshes objects, each containing num_slots meshes
        Returns:
            edge_loss: Edge length regularization loss (zero)
            normal_loss: Normal consistency loss (zero)
            laplacian_loss: Laplacian smoothing loss (zero)
        """
        return (
            torch.tensor(0.0, device=self.device),
            torch.tensor(0.0, device=self.device),
            torch.tensor(0.0, device=self.device),
        )
