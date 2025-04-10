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
    H, W = depth_map.shape

    # Get rays for all pixels
    ray_origins, ray_dirs = camera.generate_rays()  # [H*W, 3]

    # Reshape rays to match depth map
    ray_dirs = ray_dirs.reshape(H, W, 3)  # [H, W, 3]
    ray_origins = ray_origins.reshape(H, W, 3)  # [H, W, 3]

    # Get valid depths (ignore far plane and background)
    mask = (depth_map != camera.far) & (depth_map < 0)  # [H, W]
    valid_depths = depth_map[mask]  # [N]

    # Get valid rays
    valid_rays = ray_dirs[mask]  # [N, 3]
    valid_origins = ray_origins[mask]  # [N, 3]

    # Scale rays by depth to get points in camera space
    points = valid_origins + valid_rays * valid_depths.unsqueeze(-1)  # [N, 3]

    return points
