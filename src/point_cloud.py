"""
Point cloud operations for 4D reality learning system.
"""

import torch
from .utils import transform_vertices
from kaolin.render.camera import Camera
import nvdiffrast.torch as nvdiff


# def depth_to_pointcloud(depth_map: torch.Tensor, camera: Camera) -> torch.Tensor:
#     """Convert a depth map to a point cloud using camera rays.

#     Args:
#         depth_map: [H, W] tensor of depth values
#         camera: Kaolin Camera object used to generate the depth map

#     Returns:
#         points: [N, 3] tensor of points in camera space, where N is the number
#                of valid depth values (not at far plane)
#     """
#     depth_map = depth_map.squeeze()  # Remove any extra dimensions
#     depth_map *= -1

#     # Get camera rays
#     ray_orig, ray_dir = camera.generate_rays()  # [H*W, 3] each

#     # Scale rays by depth values
#     points = ray_orig + ray_dir * depth_map.reshape(-1, 1)  # [H*W, 3]

#     world2cam = camera.extrinsics.view_matrix().squeeze(0)  # [4, 4]

#     points = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)  # [H*W, 4]

#     points = points @ world2cam.T  # [H*W, 4]

#     points = points[:, :3]  # [H*W, 3]

#     return points  # [N, 3] points in camera space


def render_depth_and_pointcloud(
    camera: Camera,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get point cloud directly through rasterization.

    Args:
        camera: Camera to render from
        vertices: Base mesh vertices
        faces: Base mesh faces
        positions: Object positions
        rotations: Object rotations
        scales: Object scales

    Returns:
        depth_map: [H, W] depth values (for compatibility)
        points: [N, 3] point cloud in camera space
    """
    # Create CUDA context
    glctx = nvdiff.RasterizeCudaContext()

    # # Transform vertices for each object and combine into one big mesh
    # verts_list = []
    # faces_list = []
    # faces_offset = 0
    # for i in range(self.num_objects):
    #     # Transform vertices to world space
    #     verts = transform_vertices(
    #         vertices,
    #         positions[i],
    #         rotations[i],
    #         scales[i],
    #     )
    #     verts_list.append(verts)
    #     # Update face indices
    #     faces_list.append(faces + faces_offset)
    #     faces_offset += len(vertices)

    # # Combine into one big mesh
    # verts_all = torch.cat(verts_list, dim=0)  # [total_verts, 3]
    # faces_all = torch.cat(faces_list, dim=0)  # [total_faces, 3]

    # just use one object for testing
    verts_all = transform_vertices(
        vertices,
        positions[0],
        rotations[0],
        scales[0],
        device,
    )
    faces_all = faces  # [total_faces, 3]

    # Add homogeneous coordinate
    verts_h = torch.cat(
        [verts_all, torch.ones_like(verts_all[:, :1])], dim=1
    ).contiguous()  # [total_verts, 4]

    # Transform to camera space and scale down # [4, 4], has batch w/o sqeeze
    world2cam = camera.extrinsics.view_matrix().squeeze(0)

    # Transform and scale
    verts_cam = verts_h @ world2cam.T  # [total_verts, 4]

    # Scale down and ensure w=1
    # verts_clip = verts_cam * 0.1  # Scale everything down
    # verts_clip[..., 3] = 1.0  # Reset w to 1 after transform
    # verts_clip = verts_clip.contiguous()

    # Make faces contiguous and convert to int32
    faces_all = faces_all.contiguous().to(dtype=torch.int32)

    # Rasterize
    print("Rasterizing...")
    rast, _ = nvdiff.rasterize(
        glctx,
        verts_cam.unsqueeze(0),  # Add batch dim [1, total_verts, 4]
        faces_all,  # [total_faces, 3]
        (256, 256),  # resolution
    )

    # Get valid hits
    # valid_hits = rast[0, ..., 3] >= 0  # [H, W]

    # Initialize depth map
    depth_map = torch.full((256, 256), camera.far, device=device)
    # depth_map[valid_hits] = -rast[0, ..., 2][valid_hits]  # Negate z for our convention
    depth_map = rast[0, ..., 2]  # Negate z for our convention
    print(f"Depth map shape: {depth_map.shape}")
    # print min max mean ... of depth map
    print(f"Depth map min: {depth_map.min()}")
    print(f"Depth map max: {depth_map.max()}")
    print(f"Depth map mean: {depth_map.mean()}")
    print(f"Depth map std: {depth_map.std()}")

    # if valid_hits.sum() > 0:
    # Get barycentric coords
    # bary_u = rast[0, ..., 0][valid_hits]  # [K]
    bary_u = rast[0, ..., 0]
    # bary_v = rast[0, ..., 1][valid_hits]  # [K]
    bary_v = rast[0, ..., 1]
    bary_w = 1 - bary_u - bary_v  # [K]

    # Get triangle indices
    # tri_idx = rast[0, ..., 3][valid_hits].long()  # [K]
    tri_idx = rast[0, ..., 3].long()  # [K]

    # Get triangle vertices from world space mesh
    v0 = verts_all[faces_all[tri_idx, 0]]  # [K, 3]
    v1 = verts_all[faces_all[tri_idx, 1]]  # [K, 3]
    v2 = verts_all[faces_all[tri_idx, 2]]  # [K, 3]

    # Interpolate points in world space
    points = (
        bary_u.unsqueeze(-1) * v0
        + bary_v.unsqueeze(-1) * v1
        + bary_w.unsqueeze(-1) * v2
    )  # [K, 3]
    print(f"Points shape: {points.shape}")
    points = points.reshape(-1, 3)  # Reshape to [N, 3]
    print(f"Points reshaped: {points.shape}")
    # assert 2 == 3
    # else:
    #     points = torch.zeros((0, 3), device=device)

    return depth_map, points
