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

    # Remove any batch dimensions from depth map first
    depth_map = depth_map.squeeze()  # [H, W]

    # Generate pixel coordinates in camera space
    y_coords = (
        torch.arange(H, device=depth_map.device).view(-1, 1).expand(-1, W)
    )  # [H, W]
    x_coords = (
        torch.arange(W, device=depth_map.device).view(1, -1).expand(H, -1)
    )  # [H, W]

    # Convert to NDC space [-1, 1]
    x = (2.0 * x_coords.float() / (W - 1)) - 1.0  # [H, W]
    y = (2.0 * y_coords.float() / (H - 1)) - 1.0  # [H, W]

    # Get valid depths (ignore far plane and background)
    mask = (depth_map < -0.1) & (depth_map > -10.0)  # Only reasonable depths
    valid_depths = depth_map[mask]  # [N]

    # Debug info
    # print(f"depth range: {depth_map.min():.2f} to {depth_map.max():.2f}")
    # print(f"num valid points: {valid_depths.shape[0]}")

    # Handle case with no valid points
    if not valid_depths.numel():
        return torch.zeros((0, 3), device=depth_map.device)

    # Get valid pixel coordinates
    valid_x = x[mask]  # [N]
    valid_y = y[mask]  # [N]

    # Convert to 3D points in camera space
    # Note: Camera looks down -Z, so depths are already negative
    points = torch.stack(
        [
            valid_x * valid_depths,  # X = x * depth
            -valid_y * valid_depths,  # Y = -y * depth (flip Y to match OpenGL)
            valid_depths,  # Z = depth (already negative from renderer)
        ],
        dim=-1,
    )  # [N, 3]

    # Keep points in camera space
    return points.unsqueeze(0)  # Add batch dimension [1, N, 3]
