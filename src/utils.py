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


def compute_camera_extrinsics(
    eye,
    center,
    up,
    device=None,
    dtype=None,
):
    """
    Compute world-to-cam and cam-to-world extrinsic matrices from eye, center, up.
    Matches Kaolin's CameraExtrinsics.from_lookat convention (OpenGL/glm style).

    Args:
        eye: (3,) list, np.ndarray, or torch.Tensor
        center: (3,) list, np.ndarray, or torch.Tensor
        up: (3,) list, np.ndarray, or torch.Tensor
        device: torch.device or None
        dtype: torch.dtype or None

    Returns:
        world_to_cam: (4, 4) torch.Tensor
        cam_to_world: (4, 4) torch.Tensor
    """
    import torch

    # Convert to torch tensors
    eye = torch.as_tensor(eye, dtype=dtype) if dtype else torch.as_tensor(eye)
    center = torch.as_tensor(center, dtype=dtype) if dtype else torch.as_tensor(center)
    up = torch.as_tensor(up, dtype=dtype) if dtype else torch.as_tensor(up)
    if device:
        eye = eye.to(device)
        center = center.to(device)
        up = up.to(device)

    # Ensure shape (3,)
    eye = eye.flatten()
    center = center.flatten()
    up = up.flatten()

    # Compute axes
    backward = center - eye
    backward = backward / torch.norm(backward)
    right = torch.cross(backward, up)
    right = right / torch.norm(right)
    up_vec = torch.cross(right, backward)

    # Rotation matrix (Kaolin expects axes as columns: right, up, -backward)
    R = torch.zeros((3, 3), dtype=eye.dtype, device=eye.device)
    R[:, 0] = right
    R[:, 1] = up_vec
    R[:, 2] = -backward

    # Kaolin's view matrix is column-major, so assign R.T to [:3, :3]
    # Translation is -R.T @ eye
    Rt = R.t()
    t = -Rt @ eye.unsqueeze(1)  # (3, 1)

    # Assemble world-to-cam (view) matrix
    world_to_cam = torch.eye(4, dtype=Rt.dtype, device=Rt.device)
    world_to_cam[:3, :3] = Rt
    world_to_cam[:3, 3] = t.squeeze()

    # Cam-to-world is the inverse
    R_inv = Rt.t()
    t_inv = -R_inv @ t
    cam_to_world = torch.eye(4, dtype=Rt.dtype, device=Rt.device)
    cam_to_world[:3, :3] = R_inv
    cam_to_world[:3, 3] = t_inv.squeeze()

    return world_to_cam, cam_to_world


def test_camera_extrinsics():
    """
    Test that compute_camera_extrinsics matches Kaolin's CameraExtrinsics.from_lookat.
    """
    import torch
    from kaolin.render.camera.extrinsics import CameraExtrinsics

    # Example camera parameters (matching Scene)
    eye = torch.tensor([1.0, 0.3, 0.0])
    center = torch.tensor([0.0, 0.0, 0.0])
    up = torch.tensor([0.0, 1.0, 0.0])
    device = eye.device
    dtype = eye.dtype

    # Kaolin extrinsics
    kaolin_extr = CameraExtrinsics.from_lookat(
        eye=eye, at=center, up=up, device=device, dtype=dtype
    )
    kaolin_w2c = kaolin_extr.view_matrix().squeeze(0)
    kaolin_c2w = kaolin_extr.inv_view_matrix().squeeze(0)

    # Our function
    w2c, c2w = compute_camera_extrinsics(eye, center, up, device=device, dtype=dtype)

    print("Kaolin world-to-cam:\n", kaolin_w2c)
    print("Ours world-to-cam:\n", w2c)
    print("Kaolin cam-to-world:\n", kaolin_c2w)
    print("Ours cam-to-world:\n", c2w)

    assert torch.allclose(w2c, kaolin_w2c, atol=1e-6), (
        "World-to-cam matrices do not match!"
    )
    assert torch.allclose(c2w, kaolin_c2w, atol=1e-6), (
        "Cam-to-world matrices do not match!"
    )
    print(
        "Test passed: compute_camera_extrinsics matches Kaolin's CameraExtrinsics.from_lookat."
    )
