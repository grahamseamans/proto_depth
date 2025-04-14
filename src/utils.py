import torch


def transform_vertices(
    vertices: torch.Tensor,
    position: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Transform mesh vertices based on position, rotation, and scale"""
    # Make transform matrix from position, rotation (euler angles), and scale
    cos_r = torch.cos(rotation)
    sin_r = torch.sin(rotation)

    # Rotation matrices
    R_x = torch.tensor(
        [[1, 0, 0], [0, cos_r[0], -sin_r[0]], [0, sin_r[0], cos_r[0]]],
        device=device,
    )
    R_y = torch.tensor(
        [[cos_r[1], 0, sin_r[1]], [0, 1, 0], [-sin_r[1], 0, cos_r[1]]],
        device=device,
    )
    R_z = torch.tensor(
        [[cos_r[2], -sin_r[2], 0], [sin_r[2], cos_r[2], 0], [0, 0, 1]],
        device=device,
    )

    # Combine into single transform
    R = torch.matmul(torch.matmul(R_z, R_y), R_x)
    R = R * scale  # Scale the rotation matrix

    # Apply rotation and translation
    return vertices @ R.T + position.unsqueeze(0)
