"""
Scene state representation for 4D reality learning system.
Handles object parameters, camera parameters, and type distributions.
"""

import torch
import torch.nn as nn


class SceneState:
    """Manages the optimizable state of the scene"""

    def __init__(self, num_objects=2, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_objects = num_objects

        # Initialize object parameters
        self.objects = [ObjectState(device=device) for _ in range(num_objects)]

        # Initialize camera parameters
        self.camera = CameraState(device=device)

    def parameters(self):
        """Get all optimizable parameters"""
        params = []
        for obj in self.objects:
            params.extend(obj.parameters())
        params.extend(self.camera.parameters())
        return params

    def get_object_positions(self):
        """Get positions of all objects"""
        return torch.stack([obj.position for obj in self.objects])

    def get_object_rotations(self):
        """Get rotations of all objects"""
        return torch.stack([obj.rotation for obj in self.objects])

    def get_object_scales(self):
        """Get scales of all objects"""
        return torch.stack([obj.scale for obj in self.objects])

    def get_type_distributions(self):
        """Get type distributions for all objects"""
        return torch.stack([obj.get_type_distribution() for obj in self.objects])


class ObjectState:
    """Individual object state parameters"""

    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Position (x, y, z)
        self.position = nn.Parameter(torch.zeros(3, device=device))

        # Rotation (yaw, pitch, roll)
        self.rotation = nn.Parameter(torch.zeros(3, device=device))

        # Scale (uniform)
        self.scale = nn.Parameter(torch.ones(1, device=device))

        # Type parameters (categorical distribution logits)
        # Start with 2 types (dragon, bunny)
        self.type_logits = nn.Parameter(torch.zeros(2, device=device))

        # Initialize with reasonable values
        # Position: random in [-1, 1] meters
        self.position.data = torch.rand(3, device=device) * 2 - 1
        # Rotation: random in [-π/4, π/4] radians
        self.rotation.data = (torch.rand(3, device=device) - 0.5) * (torch.pi / 2)
        # Scale: random in [1.0, 3.0] (bigger to see better)
        self.scale.data = torch.rand(1, device=device) * 2.0 + 1.0

    def parameters(self):
        """Get all parameters for this object"""
        return [self.position, self.rotation, self.scale, self.type_logits]

    def get_type_distribution(self):
        """Get softmax distribution over types"""
        return torch.softmax(self.type_logits, dim=0)


class CameraState:
    """Camera parameters"""

    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Position (x, y, z)
        self.position = nn.Parameter(torch.zeros(3, device=device))

        # Rotation (yaw, pitch, roll)
        self.rotation = nn.Parameter(torch.zeros(3, device=device))

    def parameters(self):
        """Get all camera parameters"""
        return [self.position, self.rotation]
