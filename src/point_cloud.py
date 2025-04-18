"""
Point cloud operations for 4D reality learning system.
"""

import torch
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from .utils import transform_vertices
from kaolin.render.camera import Camera
import nvdiffrast.torch as nvdiff


@dataclass
class O3DCamera:
    eye: list  # [x, y, z]
    center: list  # [x, y, z]
    up: list  # [x, y, z]
    fov: float  # degrees
    width: int
    height: int


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
) -> torch.Tensor:
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

    # point clouds are in camera space by design
    # Get triangle vertices from world space mesh
    verts_cam = verts_cam[..., :3]  # Drop homogeneous coordinate
    v0 = verts_cam[faces_all[tri_idx, 0]]  # [K, 3]
    v1 = verts_cam[faces_all[tri_idx, 1]]  # [K, 3]
    v2 = verts_cam[faces_all[tri_idx, 2]]  # [K, 3]

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

    return points


def render_pointcloud_o3d(
    o3d_camera: O3DCamera,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Render a point cloud from a mesh using Open3D RaycastingScene and explicit camera parameters."""

    print("open3d cuda:", o3d.core.cuda.is_available())  # Should return True

    # Transform vertices to world space (N, 3)
    verts_all = (
        transform_vertices(
            vertices,
            positions[0],
            rotations[0],
            scales[0],
            device,
        )
        .cpu()
        .numpy()
    )
    faces_np = faces.cpu().numpy().astype(np.int32)

    # Print mesh centroid and bounding box
    mesh_centroid = verts_all.mean(axis=0)
    mesh_min = verts_all.min(axis=0)
    mesh_max = verts_all.max(axis=0)
    print("Mesh centroid:", mesh_centroid)
    print("Mesh bounding box min:", mesh_min, "max:", mesh_max)

    # Create Open3D t.geometry TriangleMesh
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex["positions"] = o3d.core.Tensor(verts_all, dtype=o3d.core.Dtype.Float32)
    mesh.triangle["indices"] = o3d.core.Tensor(faces_np, dtype=o3d.core.Dtype.UInt32)

    # Create RaycastingScene and add mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)

    # Create rays for the camera using explicit parameters
    rays = scene.create_rays_pinhole(
        fov_deg=o3d_camera.fov,
        center=o3d_camera.center,
        eye=o3d_camera.eye,
        up=o3d_camera.up,
        width_px=o3d_camera.width,
        height_px=o3d_camera.height,
    )

    # Print first few ray origins and directions
    rays_np = rays.numpy()  # (H, W, 6)
    print("First 5 ray origins:\n", rays_np.reshape(-1, 6)[:5, :3])
    print("First 5 ray directions:\n", rays_np.reshape(-1, 6)[:5, 3:])

    # Cast rays and get intersection results
    ray_cast_results = scene.cast_rays(rays)
    t_hit = ray_cast_results["t_hit"].numpy()  # (H, W)

    # Print number of rays that hit the mesh
    num_rays = t_hit.size
    num_hits = np.isfinite(t_hit).sum()
    print(f"Number of rays: {num_rays}, Number of hits: {num_hits}")

    origins = rays_np[..., :3]
    directions = rays_np[..., 3:]

    # Compute intersection points: origin + direction * t_hit
    points = origins + directions * t_hit[..., None]

    # Filter out rays that missed (t_hit == inf)
    mask = np.isfinite(t_hit)
    points = points[mask]

    # Convert to torch tensor on the correct device
    points = torch.from_numpy(points).to(device).reshape(-1, 3)

    # Transform points from world space to camera space using new world2cam matrix
    from .utils import compute_camera_extrinsics

    world2cam, _ = compute_camera_extrinsics(
        eye=o3d_camera.eye,
        center=o3d_camera.center,
        up=o3d_camera.up,
        device=points.device,
        dtype=points.dtype,
    )
    # Convert to homogeneous coordinates
    points_h = torch.cat(
        [
            points,
            torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype),
        ],
        dim=1,
    )
    points_cam = (world2cam @ points_h.T).T[:, :3]

    return points_cam
