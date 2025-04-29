"""
Point cloud operations for 4D reality learning system.
"""

import torch
from .utils import transform_vertices
from kaolin.render.camera import Camera
from kaolin.render.mesh import rasterize


def render_pointcloud(
    camera: Camera,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    # Build a combined mesh of all objects
    verts_list = []
    faces_list = []
    offset = 0
    num_objs = positions.shape[0]

    for i in range(num_objs):
        # Transform object i to world space
        v = transform_vertices(vertices, positions[i], rotations[i], scales[i], device)
        verts_list.append(v)  # [V,3]

        # Offset faces and collect
        f = faces.contiguous().to(torch.int64) + offset  # [F,3]
        faces_list.append(f)

        offset += v.shape[0]

    # Concatenate into mega‐mesh
    verts_all = torch.cat(verts_list, dim=0)  # [num_objs*V,3]
    faces_all = torch.cat(faces_list, dim=0)  # [num_objs*F,3]

    faces_all = faces_all.contiguous().to(torch.int64)
    faces_all = faces_all[:, [0, 2, 1]]  # swap v1 and v2 for every triangle

    # World -> Camera space (homogeneous)
    verts_h = torch.cat([verts_all, torch.ones_like(verts_all[:, :1])], dim=1)  # [V,4]
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
    # Features: camera-space xyz and depth
    face_xyz = verts_cam[faces_all, :3]  # [F,3,3]
    face_z = face_xyz[..., 2]  # [F,3]

    # Rasterize via Kaolin CUDA backend

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
    assert len(points.shape) == 2 and points.shape[1] == 3 and points.shape[0] >= 0, (
        f"Expected shape [n, 3] where n is a natural number, got {points.shape}"
    )
    return points
