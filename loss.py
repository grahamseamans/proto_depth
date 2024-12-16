import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
import trimesh
import matplotlib.pyplot as plt
import imageio

# ========== Utility functions ==========


def load_ground_truth_pointcloud(dolphin_path):
    # Load a known mesh (dolphin or any shape)
    gt_mesh = trimesh.load(dolphin_path)
    # Sample points from GT mesh
    gt_points = gt_mesh.sample(2000)  # e.g. 2000 points
    return gt_points


def create_sphere_mesh(subdiv=2):
    # Create an icosphere with trimesh
    sphere = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)

    # sphere.vertices is a TrackedArray, convert to np array first
    verts_np = np.array(
        sphere.vertices, dtype=np.float32
    )  # Now it's a proper NumPy array
    faces_np = np.array(sphere.faces, dtype=np.int32)

    # Create Tinygrad Tensor from verts_np
    mesh_verts = Tensor(verts_np, requires_grad=True)
    # mesh_faces = Tensor(faces_np)  # faces can stay as NumPy since they are just indices
    mesh_faces = faces_np

    return mesh_verts, mesh_faces


def point_to_triangle_distance(p, tri):
    # p: (3,) Tensor
    # tri: (3,3) Tensor of vertex coords
    # Compute closest point on triangle to point p
    # Steps:
    v0, v1, v2 = tri[0], tri[1], tri[2]
    # Edges
    e0 = v1 - v0
    e1 = v2 - v0
    v = p - v0

    # Compute dot products
    dot00 = (e0 * e0).sum()
    dot01 = (e0 * e1).sum()
    dot02 = (e0 * v).sum()
    dot11 = (e1 * e1).sum()
    dot12 = (e1 * v).sum()

    # Compute barycentric coords
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-9)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    # Check where closest point lies
    # If inside the triangle
    inside_mask = (u >= 0) * (v >= 0) * (u + v <= 1)
    closest = inside_mask * (v0 + e0 * u + e1 * v) + (1 - inside_mask) * 0.0
    # If not inside, need to handle edges/vertices
    # For simplicity: handle inside only. In practice you must handle edges/vertices.
    # For a full implementation, you'd do separate checks for edges/vertices.
    # Let's assume small u,v means inside or you handle that logic similarly.
    # NOTE: This is a simplification for demonstration.

    dist = ((p - closest) ** 2).sum()
    return dist


def vector_cross(u, v):
    # u, v: Tensor of shape (3,)
    ux, uy, uz = u[0], u[1], u[2]
    vx, vy, vz = v[0], v[1], v[2]
    return Tensor.stack(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)


def triangle_area(tri):
    # tri: (3,3) Tensor for vertices: [v0, v1, v2]
    v0, v1, v2 = tri[0], tri[1], tri[2]
    e0 = v1 - v0
    e1 = v2 - v0
    cross = vector_cross(e0, e1)
    area = 0.5 * (cross * cross).sum().sqrt()
    return area


def plot(gt_points, mesh_verts, mesh_faces):

    # Convert mesh_verts to numpy for plotting
    sphere_verts_np = mesh_verts.detach().numpy()  # (V,3)

    # Set up matplotlib figure
    fig = plt.figure(figsize=(12, 6))

    # Left subplot: ground-truth point cloud
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(
        gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], s=1, c="blue", alpha=0.5
    )
    ax1.set_title("Ground-Truth Point Cloud")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(elev=20, azim=30)

    # Right subplot: predicted sphere mesh
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    # plot_trisurf: expects arrays for x, y, z and a 'triangles' parameter
    ax2.plot_trisurf(
        sphere_verts_np[:, 0],
        sphere_verts_np[:, 1],
        sphere_verts_np[:, 2],
        triangles=mesh_faces,
        color="grey",
        alpha=0.7,
        edgecolor="none",
    )
    ax2.set_title("Initial Predicted Mesh (Sphere)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.show()


def triangle_area_batch(tri):
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    e0 = v1 - v0
    e1 = v2 - v0
    cross = vector_cross_batch(e0, e1)  # implement vector_cross_batch similarly
    area = 0.5 * (cross * cross).sum(axis=1).sqrt()
    return area


def point_to_triangle_distance_batch(p, tri):
    # p: (N,3)
    # tri: (N,3,3)
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]

    e0 = v1 - v0  # (N,3)
    e1 = v2 - v0  # (N,3)
    v = p - v0  # (N,3)

    dot00 = (e0 * e0).sum(axis=1)  # (N,)
    dot01 = (e0 * e1).sum(axis=1)
    dot11 = (e1 * e1).sum(axis=1)
    dot02 = (e0 * v).sum(axis=1)
    dot12 = (e1 * v).sum(axis=1)

    invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-9)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    w = (dot00 * dot12 - dot01 * dot02) * invDenom

    # If inside the triangle (u>=0, w>=0, u+w<=1), closest point inside
    # Otherwise handle edges/vertices similarly (you must implement that logic)
    # For brevity, assume inside only as before
    inside_mask = (u >= 0) * (w >= 0) * ((u + w) <= 1)
    # closest point:
    closest_pt = v0 + e0 * u.reshape(-1, 1) + e1 * w.reshape(-1, 1)
    dist = ((p - closest_pt) ** 2).sum(axis=1)
    # NOTE: This ignores edge/vertex cases. In a real implementation:
    # handle edges by checking conditions and projecting onto edges,
    # handle vertices by taking min(distance to each vertex).
    return dist


def vector_cross_batch(u, v):
    # u, v: (N,3)
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    return Tensor.stack(cx, cy, cz, dim=1)  # (N,3)


if __name__ == "__main__":
    obj_path = "data/obj/stanford-bunny.obj"
    gt_points = load_ground_truth_pointcloud(obj_path)  # (N,3) np
    mesh_verts, mesh_faces = create_sphere_mesh()  # initial sphere

    Tensor.training = True
    optimizer = Adam([mesh_verts], lr=1e-2)

    # We'll store snapshots in a list of filenames
    snapshot_files = []

    before = mesh_verts.detach().numpy()

    for epoch in range(100):
        # Convert to numpy for face assignment
        current_verts = mesh_verts.detach().numpy()
        pred_mesh = trimesh.Trimesh(
            vertices=current_verts, faces=mesh_faces, process=False
        )
        closest_points, distances, face_ids = pred_mesh.nearest.on_surface(gt_points)

        # back in Tinygrad
        gt_points_t = Tensor(np.array(gt_points, np.float32))

        f_vi = mesh_faces[face_ids]  # (N,3)
        f_vi_t = Tensor(f_vi.astype(np.int32))
        tri_for_points = mesh_verts[f_vi_t]  # (N,3,3)

        dist_array = point_to_triangle_distance_batch(
            gt_points_t, tri_for_points
        )  # (N,)

        # aggregate
        F = mesh_faces.shape[0]
        dist_np = dist_array.detach().numpy()
        counts = np.bincount(face_ids, minlength=F)
        sum_distances = np.bincount(face_ids, weights=dist_np, minlength=F)
        E_f = sum_distances / np.maximum(counts, 1)
        E_f_t = Tensor(E_f.astype(np.float32))

        mesh_faces_t = Tensor(mesh_faces.astype(np.int32))
        tri_all = mesh_verts[mesh_faces_t]  # (F,3,3)
        A_f_t = triangle_area_batch(tri_all)

        total_loss = (E_f_t * A_f_t).sum()
        # total_loss = (E_f_t).sum()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss={total_loss.numpy().item()}")

        # Every 10 epochs, save a snapshot
        if epoch % 10 == 0:
            # save a plot
            filename = f"snapshot_{epoch}.png"
            fig = plt.figure(figsize=(12, 6))

            # plot gt_points
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax1.scatter(
                gt_points[:, 0],
                gt_points[:, 1],
                gt_points[:, 2],
                s=1,
                c="blue",
                alpha=0.5,
            )
            ax1.set_title("Ground Truth")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")

            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            sphere_verts_np = mesh_verts.realize().detach().numpy()
            ax2.plot_trisurf(
                sphere_verts_np[:, 0],
                sphere_verts_np[:, 1],
                sphere_verts_np[:, 2],
                triangles=mesh_faces,
                color="grey",
                alpha=0.7,
                edgecolor="none",
            )
            ax2.set_title(f"Predicted (Epoch {epoch})")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("Z")

            plt.tight_layout()
            plt.savefig(filename)
            plt.close(fig)
            snapshot_files.append(filename)

    after = mesh_verts.detach().numpy()

    for x1, x2 in zip(before, after):
        for y1, y2 in zip(x1, x2):
            if y1 != y2:
                print(y1, y2)

    # After training, combine snapshots into a GIF
    images = []
    for f in snapshot_files:
        images.append(imageio.imread(f))
    imageio.mimsave("training_animation.gif", images, fps=2)
