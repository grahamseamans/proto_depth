import torch


from scipy.spatial.transform import Rotation as R


def transform_vertices(
    vertices: torch.Tensor,
    position: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Transform mesh vertices based on position, rotation (quaternion), and scale"""
    # Convert quaternion (x, y, z, w) to rotation matrix
    # Ensure rotation is a 1D tensor of length 4
    if rotation.shape[-1] != 4:
        raise ValueError("Rotation must be a quaternion of shape (4,)")

    # Convert to numpy for scipy
    quat_np = rotation.detach().cpu().numpy()
    rot_matrix = R.from_quat(quat_np).as_matrix()  # (3, 3)
    R_torch = torch.tensor(rot_matrix, dtype=vertices.dtype, device=device)

    # Scale the rotation matrix
    R_torch = R_torch * scale

    # Apply rotation and translation
    return vertices @ R_torch.T + position.unsqueeze(0)
