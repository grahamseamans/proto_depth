"""
Point cloud operations for 4D reality learning system.
"""

import torch
import numpy as np

# import open3d as o3d
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


# def render_depth_and_pointcloud(
#     camera: Camera,
#     vertices: torch.Tensor,
#     faces: torch.Tensor,
#     positions: torch.Tensor,
#     rotations: torch.Tensor,
#     scales: torch.Tensor,
#     device: torch.device,
# ) -> torch.Tensor:
#     """Get point cloud directly through rasterization.

#     Args:
#         camera: Camera to render from
#         vertices: Base mesh vertices
#         faces: Base mesh faces
#         positions: Object positions
#         rotations: Object rotations
#         scales: Object scales

#     Returns:
#         depth_map: [H, W] depth values (for compatibility)
#         points: [N, 3] point cloud in camera space
#     """
#     # Create CUDA context
#     glctx = nvdiff.RasterizeCudaContext()

#     # # Transform vertices for each object and combine into one big mesh
#     # verts_list = []
#     # faces_list = []
#     # faces_offset = 0
#     # for i in range(self.num_objects):
#     #     # Transform vertices to world space
#     #     verts = transform_vertices(
#     #         vertices,
#     #         positions[i],
#     #         rotations[i],
#     #         scales[i],
#     #     )
#     #     verts_list.append(verts)
#     #     # Update face indices
#     #     faces_list.append(faces + faces_offset)
#     #     faces_offset += len(vertices)

#     # # Combine into one big mesh
#     # verts_all = torch.cat(verts_list, dim=0)  # [total_verts, 3]
#     # faces_all = torch.cat(faces_list, dim=0)  # [total_faces, 3]

#     # just use one object for testing
#     verts_all = transform_vertices(
#         vertices,
#         positions[0],
#         rotations[0],
#         scales[0],
#         device,
#     )
#     faces_all = faces  # [total_faces, 3]

#     # Add homogeneous coordinate
#     verts_h = torch.cat(
#         [verts_all, torch.ones_like(verts_all[:, :1])], dim=1
#     ).contiguous()  # [total_verts, 4]

#     # Transform to camera space and scale down # [4, 4], has batch w/o sqeeze
#     world2cam = camera.extrinsics.view_matrix().squeeze(0)

#     # Transform to camera space
#     verts_cam = verts_h @ world2cam.T  # [total_verts, 4]

#     # Apply projection to get clip and NDC coordinates
#     proj = camera.projection_matrix().squeeze(0)
#     verts_clip = verts_cam @ proj.T  # [total_verts, 4]
#     verts_ndc = verts_clip[:, :3] / verts_clip[:, 3:4]  # [total_verts, 3]
#     verts_ndc_h = torch.cat(
#         [verts_ndc, torch.ones_like(verts_ndc[:, :1])], dim=1
#     )  # [total_verts, 4]

#     # Gather per-face vertices
#     face_vs = verts_ndc_h[faces_all]  # [total_faces, 3, 4]
#     face_vertices_image = face_vs[..., :2]  # [total_faces, 3, 2]
#     face_vertices_z = face_vs[..., 2]  # [total_faces, 3]

#     # Convert to OpenGL conventions (flip Y, preserve Z, add homogeneous)
#     verts_opengl, faces_opengl = to_opengl_no_z_normalize(
#         face_vertices_image, face_vertices_z
#     )

#     # Make faces contiguous and convert to int32
#     faces_opengl = faces_opengl.contiguous().to(dtype=torch.int32)

#     # Rasterize
#     print("Rasterizing...")
#     rast, _ = nvdiff.rasterize(
#         glctx,
#         verts_opengl.unsqueeze(0),  # Add batch dim [1, total_verts, 4]
#         faces_opengl,  # [total_faces, 3]
#         (256, 256),  # resolution
#     )

#     # Get valid hits
#     # valid_hits = rast[0, ..., 3] >= 0  # [H, W]

#     # if valid_hits.sum() > 0:
#     # Get barycentric coords
#     # bary_u = rast[0, ..., 0][valid_hits]  # [K]
#     bary_u = rast[0, ..., 0]
#     # bary_v = rast[0, ..., 1][valid_hits]  # [K]
#     bary_v = rast[0, ..., 1]
#     bary_w = 1 - bary_u - bary_v  # [K]

#     # Get triangle indices
#     # tri_idx = rast[0, ..., 3][valid_hits].long()  # [K]
#     tri_idx = rast[0, ..., 3].long()  # [K]

#     # point clouds are in camera space by design
#     # Get triangle vertices from world space mesh
#     verts_cam = verts_cam[..., :3]  # Drop homogeneous coordinate
#     v0 = verts_cam[faces_all[tri_idx, 0]]  # [K, 3]
#     v1 = verts_cam[faces_all[tri_idx, 1]]  # [K, 3]
#     v2 = verts_cam[faces_all[tri_idx, 2]]  # [K, 3]

#     # Interpolate points in world space
#     points = (
#         bary_u.unsqueeze(-1) * v0
#         + bary_v.unsqueeze(-1) * v1
#         + bary_w.unsqueeze(-1) * v2
#     )  # [K, 3]
#     print(f"Points shape: {points.shape}")
#     points = points.reshape(-1, 3)  # Reshape to [N, 3]
#     print(f"Points reshaped: {points.shape}")
#     # assert 2 == 3
#     # else:
#     #     points = torch.zeros((0, 3), device=device)

#     return points


def render_pointcloud(
    camera: Camera,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Render a point cloud by barycentric interpolation of camera‐space xyz.

    Returns:
        points: [N,3] tensor of 3D points in camera space.
    """
    # Build mesh
    verts_world = transform_vertices(
        vertices, positions[0], rotations[0], scales[0], device
    )
    faces_all = faces.contiguous().to(torch.int64)

    # World -> Camera space (homogeneous)
    verts_h = torch.cat(
        [verts_world, torch.ones_like(verts_world[:, :1])], dim=1
    )  # [V,4]
    view = camera.extrinsics.view_matrix().squeeze(0)  # [4,4]
    verts_cam = verts_h @ view.T  # [V,4]

    # Camera -> Clip -> NDC
    proj = camera.intrinsics.projection_matrix().squeeze(0)  # [4,4]
    verts_clip = verts_cam @ proj.T  # [V,4]
    ndc = verts_clip[:, :3] / verts_clip[:, 3:4]  # [V,3]
    verts_ndc_h = torch.cat([ndc, torch.ones_like(ndc[:, :1])], dim=1)  # [V,4]

    # Per-face vertices
    face_vs = verts_ndc_h[faces_all]  # [F,3,4]
    face_xy = face_vs[..., :2]  # [F,3,2]
    face_z = face_vs[..., 2]  # [F,3]

    # Features: camera-space xyz
    face_xyz = verts_cam[faces_all, :3]  # [F,3,3]

    # # Backface culling in camera space
    # v0 = face_xyz[:, 0, :]
    # v1 = face_xyz[:, 1, :]
    # v2 = face_xyz[:, 2, :]
    # normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    # normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
    # centroids = (v0 + v1 + v2) / 3
    # view_dirs = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-8)
    # facing = (normals * view_dirs).sum(dim=1) > 0  # [F]
    # valid_faces = facing.unsqueeze(0)  # [1, F]

    # Rasterize via Kaolin CUDA backend
    from kaolin.render.mesh import rasterize

    image_xyz, face_idx = rasterize(
        camera.height,
        camera.width,
        face_z.unsqueeze(0),  # [1,F,3]
        face_xy.unsqueeze(0),  # [1,F,3,2]
        face_xyz.unsqueeze(0),  # [1,F,3,3]
        # valid_faces=valid_faces,
        backend="cuda",
    )
    # image_xyz[0]: [H,W,3]

    # Mask and collect valid points
    mask = face_idx[0] >= 0  # [H,W]
    points = image_xyz[0][mask]  # [N,3]
    return points


# def to_opengl_no_z_normalize(face_vertices_image, face_vertices_z):
#     """
#     Convert face vertex image coordinates and z to OpenGL conventions,
#     but DO NOT normalize z (preserve absolute depth).

#     Args:
#         face_vertices_image: (B, F, 3, 2) or (F, 3, 2)
#         face_vertices_z:     (B, F, 3) or (F, 3)

#     Returns:
#         pos: (B, F*3, 4) or (F*3, 4)
#         tri: (F, 3)
#     """
#     # Reshape image coordinates to (..., N*3, 2)
#     _face_vertices_image = face_vertices_image.reshape(
#         *face_vertices_image.shape[:-3], -1, 2
#     )
#     # Flip Y axis for OpenGL
#     pos = torch.stack(
#         [
#             _face_vertices_image[..., 0],  # X stays the same
#             -_face_vertices_image[..., 1],  # Y is flipped
#             face_vertices_z.reshape(*face_vertices_z.shape[:-2], -1),  # Z is preserved
#         ],
#         dim=-1,
#     )
#     # Add homogeneous coordinate (w=1)
#     pos = torch.nn.functional.pad(pos, (0, 1), value=1.0)
#     # Triangle indices: [0, 1, 2], [3, 4, 5], ...
#     tri = torch.arange(pos.shape[-2], device=pos.device, dtype=torch.int).reshape(-1, 3)
#     return pos, tri


# def from_opengl_no_z_normalize(pos):
#     """
#     Convert OpenGL-style positions back to original conventions (undo Y flip).

#     Args:
#         pos: (B, N, 4) or (N, 4)  # Homogeneous coordinates

#     Returns:
#         face_vertices_image: (B, N, 2) or (N, 2)
#         face_vertices_z:     (B, N) or (N,)
#     """
#     # Undo Y axis flip
#     x = pos[..., 0]
#     y = -pos[..., 1]
#     z = pos[..., 2]
#     face_vertices_image = torch.stack([x, y], dim=-1)
#     face_vertices_z = z
#     return face_vertices_image, face_vertices_z
