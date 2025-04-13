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
        points: [N, 3] tensor of points in camera space, where N is the number
               of valid depth values (not at far plane)
    """
    depth_map = depth_map.squeeze()  # Remove any extra dimensions
    depth_map *= -1  # Convert to positive magnitudes

    # Get camera rays
    ray_orig, ray_dir = camera.generate_rays()  # [H*W, 3] each

    # Normalize ray directions before scaling by depth
    ray_dir = ray_dir / torch.linalg.norm(ray_dir, dim=-1, keepdim=True)

    # Scale normalized rays by depth values
    points = ray_orig + ray_dir * depth_map.reshape(-1, 1)  # [H*W, 3]

    world2cam = camera.extrinsics.view_matrix().squeeze(0)  # [4, 4]
    points = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)  # [H*W, 4]
    points = points @ world2cam.T  # [H*W, 4]
    points = points[:, :3]  # [H*W, 3]

    return points  # [N, 3] points in camera space
    # depth_map = depth_map.squeeze()  # Remove any extra dimensions
    # depth_map *= -1

    # # Get camera rays
    # ray_orig, ray_dir = camera.generate_rays()  # [H*W, 3] each

    # # Scale rays by depth values
    # points = ray_orig + ray_dir * depth_map.reshape(-1, 1)  # [H*W, 3]

    # world2cam = camera.extrinsics.view_matrix().squeeze(0)  # [4, 4]

    # points = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)  # [H*W, 4]

    # points = points @ world2cam.T  # [H*W, 4]

    # points = points[:, :3]  # [H*W, 3]

    # return points  # [N, 3] points in camera space
