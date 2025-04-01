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

        # Get base vertices and faces (no gradients)
        with torch.no_grad():
            verts = self.sphere_mesh.verts_padded()  # [1, V, 3]
            faces = self.sphere_mesh.faces_padded()  # [1, F, 3]
            # Expand vertices for batch and slots
            verts = verts.expand(B * num_slots, -1, -1)  # [B*num_slots, V, 3]
            faces = faces.expand(B * num_slots, -1, -1)  # [B*num_slots, F, 3]

        # For each slot, blend the transforms using prototype weights
        final_verts = []
        for b in range(B):
            for s in range(num_slots):
                slot_verts = verts[b * num_slots + s]  # [V, 3]
                slot_weights = prototype_weights[b, s]  # [num_prototypes]

                # Initialize transformed vertices
                transformed = torch.zeros_like(slot_verts)

                # Apply each prototype transform weighted by its weight
                for p in range(num_prototypes):
                    weight = slot_weights[p]
                    transform = transforms[b, s, p]  # [6]
                    scale = scales[b, s]  # [1]

                    # Get the vertex offsets for this prototype
                    offsets = prototype_offsets[p]  # [V, 3]

                    # Apply vertex offsets to create the deformed shape
                    deformed_verts = slot_verts + offsets

                    # Split transform into components
                    translation = transform[:3].view(1, 3)  # [1, 3]
                    rotation = transform[3:]  # [3]

                    # Apply transformation
                    rot_matrix = t3d.euler_angles_to_matrix(
                        rotation.unsqueeze(0), "XYZ"
                    )  # [1, 3, 3]
                    rotated = torch.matmul(deformed_verts, rot_matrix[0].t())  # [V, 3]
                    scaled = rotated * scale
                    translated = scaled + translation

                    # Add to weighted sum
                    transformed += weight * translated

                final_verts.append(transformed)

        # Stack all vertices
        final_verts = torch.stack(final_verts)  # [B*num_slots, V, 3]

        # Create mesh objects
        meshes = []
        for b in range(B):
            # Get vertices and faces for this batch
            b_verts = final_verts[
                b * num_slots : (b + 1) * num_slots
            ]  # [num_slots, V, 3]
            b_faces = faces[b * num_slots : (b + 1) * num_slots]  # [num_slots, F, 3]

            # Create mesh for this batch
            mesh = Meshes(verts=list(b_verts), faces=list(b_faces))
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
    ):
        """
        Memory-efficient implementation of hybrid loss with three components:
        1. Global chamfer: Target points → All predicted mesh points
        2. Per-slot proximity: Each slot's points → Target points
        3. Centroid repulsion: Preventing slots from overlapping

        Args:
            meshes: List of B Meshes objects, each containing num_slots meshes
            target_pcls: List of B Tensor objects, each of shape [N_i, 3] - target point clouds
            global_weight: Weight for the global component
            slot_weight: Weight for the per-slot component
            repulsion_weight: Weight for the centroid repulsion component
            samples_per_slot: Number of points to sample per slot (much lower than before)
            min_distance: Minimum desired distance between centroids (in meters)
        Returns:
            total_loss: Combined loss scalar tensor
            global_loss: Global component scalar tensor
            per_slot_loss: Per-slot component scalar tensor
        """
        B = len(meshes)
        global_loss = 0
        per_slot_loss = 0
        repulsion_loss = 0

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

            # Use PyTorch3D's optimized implementation (one-directional)
            global_dist, _ = pytorch3d.loss.chamfer_distance(
                tgt_pcl.unsqueeze(0),  # [1, N, 3]
                pred_pcl_flat.unsqueeze(0),  # [1, num_slots*samples_per_slot, 3]
                point_reduction="mean",
                single_directional=True,  # Target → Prediction only
            )
            global_loss += global_dist

            # 2. Per-slot proximity: Slot points → Target points
            # -------------------------------------------------
            slot_losses = []
            for s in range(num_slots):
                slot_pcl = pred_pcl_per_slot[s]  # [samples_per_slot, 3]

                # One-directional: Slot → Target
                slot_dist, _ = pytorch3d.loss.chamfer_distance(
                    slot_pcl.unsqueeze(0),  # [1, samples_per_slot, 3]
                    tgt_pcl.unsqueeze(0),  # [1, N, 3]
                    point_reduction="mean",
                    single_directional=True,  # Slot → Target
                )
                slot_losses.append(slot_dist)

            # Average the per-slot losses
            batch_slot_loss = torch.mean(torch.stack(slot_losses))
            per_slot_loss += batch_slot_loss

            # 3. Centroid repulsion: Prevent slots from overlapping
            # ---------------------------------------------------
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
