import torch
import torch.nn as nn
import torchvision.models as models


class DepthEncoder(nn.Module):
    def __init__(self, num_prototypes=10, num_slots=5):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_slots = num_slots

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

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, 3, H, W] - the depth map
               (converted to 3 channels in dataloader)
        Returns:
            scales: Tensor of shape [B, num_slots, 1]
            transforms: Tensor of shape [B, num_slots, num_prototypes, 6] (x,y,z, yaw,pitch,roll)
            prototype_weights: Tensor of shape [B, num_slots, num_prototypes]
        """
        B = x.shape[0]

        # Extract features
        features = self.backbone(x)  # [B, 512, 1, 1]
        features = features.view(B, -1)  # [B, 512]

        # Project to slots
        slots = self.slot_projector(features)  # [B, num_slots * slot_dim]
        slots = slots.view(B, self.num_slots, -1)  # [B, num_slots, slot_dim]

        # Split into components
        scales = slots[..., :1]  # [B, num_slots, 1]

        # Get transforms for each prototype
        transform_dim = self.num_prototypes * 6
        transforms = slots[
            ..., 1 : 1 + transform_dim
        ]  # [B, num_slots, num_prototypes*6]
        transforms = transforms.view(B, self.num_slots, self.num_prototypes, 6)

        # Get prototype weights
        logits = slots[..., 1 + transform_dim :]  # [B, num_slots, num_prototypes]
        prototype_weights = torch.softmax(logits, dim=-1)

        return scales, transforms, prototype_weights
