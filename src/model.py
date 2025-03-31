import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch3d.utils import ico_sphere
import math


class DepthEncoder(nn.Module):
    def __init__(self, num_prototypes=10, num_slots=5, device=None):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_slots = num_slots

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # ResNet backbone (remove final classification layer)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Project to slots
        backbone_dim = 512  # ResNet18's final feature dimension
        # For each slot we output:
        # - scale (1)
        # - transforms for each prototype (num_prototypes * 6)
        # - prototype weights (num_prototypes)
        slot_dim = 1 + (num_prototypes * 6) + num_prototypes
        self.slot_projector = nn.Linear(backbone_dim, num_slots * slot_dim)

        # Create learnable vertex offsets for each prototype
        # Get the number of vertices in the sphere mesh
        sphere = ico_sphere(level=4, device=device)
        num_verts = sphere.verts_packed().shape[0]

        # Initialize learnable vertex offsets for each prototype
        # Shape: [num_prototypes, num_verts, 3]
        self.prototype_offsets = nn.Parameter(
            torch.zeros(num_prototypes, num_verts, 3, device=device)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, 3, H, W] - the depth map
               (converted to 3 channels in dataloader)
        Returns:
            scales: Tensor of shape [B, num_slots, 1]
            scene_scale_transforms: Tensor of shape [B, num_slots, num_prototypes, 6]
                                   (x,y,z in meters with scaling, rotations in radians bounded to [-π, π])
            prototype_weights: Tensor of shape [B, num_slots, num_prototypes]
            prototype_offsets: Tensor of shape [num_prototypes, num_verts, 3]
        """
        B = x.shape[0]

        # Extract features
        features = self.backbone(x)  # [B, 512, 1, 1]
        features = features.view(B, -1)  # [B, 512]

        # Project to slots
        slots = self.slot_projector(features)  # [B, num_slots * slot_dim]
        slots = slots.view(B, self.num_slots, -1)  # [B, num_slots, slot_dim]

        # Split into components
        scales = F.softplus(
            slots[..., :1]
        )  # [B, num_slots, 1] - ensure positive values

        # Get transforms for each prototype
        transform_dim = self.num_prototypes * 6
        raw_transforms = slots[
            ..., 1 : 1 + transform_dim
        ]  # [B, num_slots, num_prototypes*6]
        raw_transforms = raw_transforms.view(B, self.num_slots, self.num_prototypes, 6)

        # Apply scaling to create scene-scale transforms
        scene_scale_transforms = torch.zeros_like(raw_transforms)

        # MODIFIED: Positions need large range (meters) - use softplus with increased scaling
        # Old value was 50.0, increasing to 100.0 to make shapes more visible
        position_scale = 100.0
        # Also add a minimum offset to ensure positions are not too close to zero
        min_position_offset = 10.0  # Minimum 10 meters

        # Apply scaling and offset to positions
        scene_scale_transforms[..., :3] = (
            F.softplus(raw_transforms[..., :3]) * position_scale + min_position_offset
        )

        # Rotations bounded to [-π, π] - use tanh with pi scaling
        scene_scale_transforms[..., 3:] = torch.tanh(raw_transforms[..., 3:]) * math.pi

        # Get prototype weights
        logits = slots[..., 1 + transform_dim :]  # [B, num_slots, num_prototypes]
        prototype_weights = torch.softmax(logits, dim=-1)

        # MODIFIED: Scale up the overall model scales to make shapes more visible
        # Old scales were just softplus, multiply by a factor for more visibility
        scale_multiplier = 5.0
        scaled_scales = scales * scale_multiplier

        # Print debug info about scales and positions for the first batch
        if B > 0:
            print(f"\nDEBUG MODEL OUTPUT:")
            print(
                f"  Raw scales (before multiplier): Min={scales[0].min().item():.4f}, Max={scales[0].max().item():.4f}"
            )
            print(
                f"  Final scales (after multiplier): Min={scaled_scales[0].min().item():.4f}, Max={scaled_scales[0].max().item():.4f}"
            )

            # Check position values
            position_min = scene_scale_transforms[0, :, :, :3].min().item()
            position_max = scene_scale_transforms[0, :, :, :3].max().item()
            position_mean = scene_scale_transforms[0, :, :, :3].mean().item()
            print(
                f"  Position values: Min={position_min:.4f}, Max={position_max:.4f}, Mean={position_mean:.4f}"
            )

            # Check prototype weights
            weight_max = prototype_weights[0].max().item()
            weight_min = prototype_weights[0].min().item()
            print(f"  Prototype weights: Min={weight_min:.4f}, Max={weight_max:.4f}")
            print(
                f"  Number of weights > 0.1: {(prototype_weights[0] > 0.1).sum().item()}"
            )
            print("==========================================\n")

        return (
            scaled_scales,
            scene_scale_transforms,
            prototype_weights,
            self.prototype_offsets,
        )
