"""
Simple point cloud generation using tinygrad for ray-mesh intersection.
"""

from tinygrad import Tensor, Device, dtypes
import trimesh
import numpy as np
import open3d as o3d


def load_bunny():
    """Load bunny mesh using trimesh and convert to tinygrad tensors."""
    mesh = trimesh.load_mesh("3d_models/bunny.obj")
    vertices = Tensor(np.array(mesh.vertices, dtype=np.float32))
    faces = Tensor(np.array(mesh.faces, dtype=np.int32))
    return vertices, faces, mesh  # Return trimesh mesh for visualization


def create_camera(fov=60, width=256, height=256):
    """Create a simple pinhole camera and generate rays toward the origin."""
    # Camera position (move closer to mesh center)
    position = np.array([0.0, 0.1, 0.3], dtype=np.float32)

    # Generate pixel grid in image plane at z=0
    aspect = width / height
    tan_half_fov = np.tan(np.radians(fov / 2))
    x = np.linspace(-aspect * tan_half_fov, aspect * tan_half_fov, width)
    y = np.linspace(-tan_half_fov, tan_half_fov, height)
    xx, yy = np.meshgrid(x, y)

    # Image plane is at z=0, so world points are (xx, yy, 0)
    pixel_points = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3)
    # Rays: from camera position to pixel points
    directions = pixel_points - position[None, :]
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    rays_d = Tensor(directions.astype(np.float32))
    position = Tensor(position)

    return position, rays_d


def cross(a, b):
    # a, b: (..., 3)
    c0 = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    c1 = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    c2 = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return c0.stack(c1, c2, dim=-1)


def get_ray_intersections(rays_o, rays_d, vertices, faces):
    """Compute ray-triangle intersections using tinygrad."""
    # Get triangle vertices
    v0 = vertices[faces[:, 0]]  # [N, 3]
    v1 = vertices[faces[:, 1]]  # [N, 3]
    v2 = vertices[faces[:, 2]]  # [N, 3]

    # Compute triangle edges
    edge1 = v1 - v0  # [N, 3]
    edge2 = v2 - v0  # [N, 3]

    # rays_o: [3] or [num_rays, 3]
    # We'll use rays_o[i:i+batch_size].unsqueeze(1) for correct broadcasting

    # Initialize output arrays
    closest_t = Tensor.full((len(rays_d),), float("inf")).contiguous()
    intersection_points = Tensor.zeros((len(rays_d), 3)).contiguous()

    # Process rays in batches to avoid OOM
    batch_size = 1024
    num_rays = len(rays_d)
    num_faces = v0.shape[0]
    if num_faces == 0:
        return np.zeros((0, 3), dtype=np.float32)

    for i in range(0, num_rays, batch_size):
        batch_rays = rays_d[i : i + batch_size]  # [B, 3]
        if num_faces == 0:
            continue

        # Möller–Trumbore algorithm
        h = cross(batch_rays.unsqueeze(1), edge2.unsqueeze(0))  # [B, N, 3]
        a = edge1.unsqueeze(0).mul(h).sum(axis=-1)  # [B, N]

        # Skip if ray is parallel to triangle
        valid = a.abs() > 1e-8

        f = 1.0 / (a + (~valid) * 1e9)  # Avoid division by zero for invalid
        # camera_pos: [3], batch_rays: [B, 3], v0: [N, 3]
        s = rays_o.expand(batch_rays.shape[0], 1, 3) - v0.unsqueeze(0)  # [B, N, 3]
        u = f * s.mul(h).sum(axis=-1)

        valid = valid & (u >= 0) & (u <= 1)

        q = cross(s, edge1.unsqueeze(0))  # [B, N, 3]
        v = f * batch_rays.unsqueeze(1).mul(q).sum(axis=-1)

        valid = valid & (v >= 0) & (u + v <= 1)

        t = f * edge2.unsqueeze(0).mul(q).sum(axis=-1)
        valid = valid & (t > 0)

        # Update closest intersections
        t = t.where(valid, float("inf"))
        min_t = t.min(axis=1)
        min_idx = t.argmin(axis=1)
        update = min_t < closest_t[i : i + batch_size]

        points = rays_o + batch_rays * min_t.unsqueeze(-1)
        intersection_points[i : i + batch_size] = points.where(
            update.unsqueeze(-1),
            intersection_points[i : i + batch_size],
        )
        closest_t[i : i + batch_size] = min_t.where(
            update, closest_t[i : i + batch_size]
        )

    # Filter out misses
    valid_hits = closest_t < float("inf")
    # Use numpy for boolean indexing
    return intersection_points.numpy()[valid_hits.numpy()]


def visualize(trimesh_mesh, camera_pos, intersection_points):
    """Visualize the mesh, camera, and intersection points using Open3D."""
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Tinygrad Point Cloud", width=1600, height=900)

    # Convert bunny mesh to Open3D format
    vertices = np.array(trimesh_mesh.vertices)
    triangles = np.array(trimesh_mesh.faces, dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
    vis.add_geometry(mesh)

    # Add intersection points
    if len(intersection_points) > 0:
        points = intersection_points.numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([1, 0, 0])  # Red points
        vis.add_geometry(pcd)

    # Add camera
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    camera_frame.translate(camera_pos.numpy())
    vis.add_geometry(camera_frame)

    # Add coordinate frame at origin
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(frame)

    # Set view
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])  # Look along -Z axis
    view_control.set_lookat([0, 0, 0])  # Look at origin
    view_control.set_up([0, 1, 0])  # Y is up
    view_control.set_zoom(0.7)

    # Run visualizer
    vis.run()
    vis.destroy_window()


def main():
    """Main function to test ray-mesh intersection with tinygrad."""
    print("Loading bunny mesh...")
    vertices, faces, trimesh_mesh = load_bunny()

    # Print mesh bounds for debugging
    mesh_np = np.array(trimesh_mesh.vertices)
    print(f"Mesh bounds: min={mesh_np.min(axis=0)}, max={mesh_np.max(axis=0)}")

    print("Creating camera...")
    camera_pos, rays_d = create_camera()
    print(f"Camera position: {camera_pos.numpy()}")
    print(f"Sample ray directions: {rays_d.numpy()[:5]}")

    # --- Face culling based on normals ---
    v_np = np.array(trimesh_mesh.vertices)
    f_np = np.array(trimesh_mesh.faces, dtype=np.int32)
    # Compute face centers and normals
    face_verts = v_np[f_np]  # [F, 3, 3]
    face_centers = face_verts.mean(axis=1)  # [F, 3]
    face_normals = np.cross(
        face_verts[:, 1] - face_verts[:, 0], face_verts[:, 2] - face_verts[:, 0]
    )
    face_normals = face_normals / (
        np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
    )
    # Vector from face center to camera
    cam_pos_np = camera_pos.numpy()
    to_camera = cam_pos_np - face_centers  # [F, 3]
    to_camera = to_camera / (np.linalg.norm(to_camera, axis=1, keepdims=True) + 1e-8)
    # Cull faces whose normals point away from camera
    dot = (face_normals * to_camera).sum(axis=1)
    keep = dot > 0  # Only keep faces facing the camera
    print(f"Culled {np.sum(~keep)} faces out of {len(f_np)}")
    f_culled = f_np[keep]
    # Convert to tinygrad tensors
    faces = Tensor(f_culled)
    # (vertices remain the same)

    print("Computing ray intersections...")
    intersection_points = get_ray_intersections(camera_pos, rays_d, vertices, faces)

    print(f"Found {len(intersection_points)} intersection points")
    print("Visualizing results...")
    visualize(trimesh_mesh, camera_pos, intersection_points)


if __name__ == "__main__":
    main()
