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
        B = len(meshes)
        total_loss = 0

        for b in range(B):
            # Sample points from all meshes in this batch
            pred_pcl = pytorch3d.ops.sample_points_from_meshes(
                meshes[b], num_samples=5000
            )  # [num_slots, 5000, 3]
            pred_pcl = pred_pcl.reshape(-1, 3)  # [num_slots*5000, 3]

            # Get target points
            tgt_pcl = target_pcls[b]  # [N, 3]

            # For each target point, find distance to closest predicted point
            # [N, num_slots*5000]
            distances = torch.cdist(tgt_pcl, pred_pcl, p=2)

            # For each target point, get distance to closest predicted point
            min_distances, _ = torch.min(distances, dim=1)  # [N]

            # Average over all target points
            loss = torch.mean(min_distances)
            total_loss += loss

        return total_loss / B

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
