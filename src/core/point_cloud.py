"""
Point cloud operations for 4D reality learning system.
"""

import torch
from kaolin.render.camera import Camera


def depth_to_pointcloud(depth_map: torch.Tensor, camera: Camera) -> torch.Tensor:
    """Convert a depth map to a point cloud using camera rays.

    Args:
        depth_map: [H, W] tensor of depth values
        camera: Kaolin Camera object used to generate the depth map

    Returns:
        points: [N, 3] tensor of points in world space, where N is the number
               of valid depth values (not at far plane)
    """
    H, W = depth_map.shape[:2]  # Handle [H, W] or [H, W, 1]
    depth_map = depth_map.squeeze()  # Remove any extra dimensions

    # Generate pixel coordinates in camera space
    i, j = torch.meshgrid(
        torch.arange(H, device=depth_map.device),
        torch.arange(W, device=depth_map.device),
    )

    # Convert to NDC space [-1, 1]
    x = (2.0 * j.float() / W) - 1.0
    y = (2.0 * i.float() / H) - 1.0

    # Get valid depths (ignore far plane and background)
    mask = (depth_map != camera.far) & (depth_map > 0)  # [H, W]
    valid_depths = depth_map[mask]  # [N]

    # Handle case with no valid points
    if not valid_depths.numel():
        return torch.zeros((0, 3), device=depth_map.device)

    # Get valid pixel coordinates
    valid_x = x[mask]  # [N]
    valid_y = y[mask]  # [N]

    # Convert to 3D points in camera space
    points = torch.stack(
        [
            valid_x * valid_depths,  # X = x * depth
            -valid_y * valid_depths,  # Y = -y * depth (flip Y to match OpenGL)
            -valid_depths,  # Z = -depth (forward is -Z in camera space)
        ],
        dim=-1,
    )  # [N, 3]

    # Transform points from camera to world space
    points = camera.extrinsics.transform(points)  # [N, 3]

    return points
