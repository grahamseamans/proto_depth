import torch
from kaolin.math.quat import quat_unit, rot33_from_quat


def transform_vertices(
    vertices: torch.Tensor,
    position: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Transform mesh vertices based on position, rotation (quaternion), and scale.
    Uses Kaolin's quaternion utilities for robust, differentiable transforms.
    """
    # Ensure rotation is a 1D tensor of length 4
    if rotation.shape[-1] != 4:
        raise ValueError("Rotation must be a quaternion of shape (4,)")

    # Normalize quaternion to ensure valid rotation
    q = quat_unit(rotation)
    # Convert quaternion to rotation matrix
    R = rot33_from_quat(q.unsqueeze(0)).squeeze(0)  # [3,3]

    # Scale the rotation matrix
    R = R * scale

    # Apply rotation and translation
    transformed = vertices @ R.T + position.unsqueeze(0)
    return transformed
