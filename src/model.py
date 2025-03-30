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
        # Positions need large range (meters) - use softplus with scaling
        scene_scale_transforms[..., :3] = F.softplus(raw_transforms[..., :3]) * 50.0
        # Rotations bounded to [-π, π] - use tanh with pi scaling
        scene_scale_transforms[..., 3:] = torch.tanh(raw_transforms[..., 3:]) * math.pi

        # Get prototype weights
        logits = slots[..., 1 + transform_dim :]  # [B, num_slots, num_prototypes]
        prototype_weights = torch.softmax(logits, dim=-1)

        return scales, scene_scale_transforms, prototype_weights, self.prototype_offsets
