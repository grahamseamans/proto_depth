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


def point_to_triangle_distance_batch(p, tri):
    """
    Compute an approximate minimum distance from each point to the given triangle using a soft-min approach.
    p: (N,3) Tensor for points
    tri: (N,3,3) Tensor for triangles
    beta: float, controls the softness. Larger beta => closer to true min but less smooth.
    """
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

    # Closest point if inside
    u_exp = u.reshape(-1, 1)
    w_exp = w.reshape(-1, 1)
    closest_pt_inside = v0 + e0 * u_exp + e1 * w_exp
    dist_inside = ((p - closest_pt_inside) ** 2).sum(axis=1)  # (N,)

    # Distances to vertices
    dist_v0 = ((p - v0) ** 2).sum(axis=1)
    dist_v1 = ((p - v1) ** 2).sum(axis=1)
    dist_v2 = ((p - v2) ** 2).sum(axis=1)

    # Edge distance helper
    def edge_distance(P, A, D):
        DD = (D * D).sum(axis=1) + 1e-9
        PA = P - A
        t = ((PA * D).sum(axis=1) / DD).clip(0.0, 1.0)
        t_exp = t.reshape(-1, 1)
        edge_closest = A + D * t_exp
        return ((P - edge_closest) ** 2).sum(axis=1)

    # Edges
    dist_e0 = edge_distance(p, v0, e0)  # (v0->v1)
    dist_e1 = edge_distance(p, v0, e1)  # (v0->v2)
    e2 = v2 - v1
    dist_e2 = edge_distance(p, v1, e2)  # (v1->v2)

    # Stack all distances: inside, vertices and edges
    # shape: (N,7)
    all_dists = Tensor.stack(
        dist_inside, dist_v0, dist_v1, dist_v2, dist_e0, dist_e1, dist_e2, dim=1
    )
    min_dist = all_dists.min(axis=1)  # (N,)

    # return softmin_dist
    return min_dist


def triangle_area_batch(tri):
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    e0 = v1 - v0
    e1 = v2 - v0
    cross = vector_cross_batch(e0, e1)  # implement vector_cross_batch similarly
    area = 0.5 * (cross * cross).sum(axis=1).sqrt()
    return area


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

    print(f"GT points: {gt_points.shape}")
    print(f"Initial mesh: {mesh_verts.shape}, {mesh_faces.shape}")

    Tensor.training = True
    optimizer = Adam([mesh_verts], lr=1e-1)

    # We'll store snapshots in a list of filenames
    snapshot_files = []

    for epoch in range(200):
        # Convert to numpy for face assignment
        current_verts = mesh_verts.detach().numpy()
        pred_mesh = trimesh.Trimesh(
            vertices=current_verts, faces=mesh_faces, process=False
        )
        closest_points, distances, face_ids = pred_mesh.nearest.on_surface(gt_points)

        f_vi = mesh_faces[face_ids]  # f_vi: (N,3) array of vertex indices
        f_vi_t = Tensor(np.array(f_vi, np.int32))

        tri_for_points = mesh_verts[f_vi_t]  # (N,3,3)
        # tri_for_points now has the triangle for each point
        # This indexing is differentiable because f_vi_t is constant and mesh_verts is a parameter.

        gt_points_t = Tensor(np.array(gt_points, np.float32))  # (N,3)
        dist_array = point_to_triangle_distance_batch(
            gt_points_t, tri_for_points
        )  # (N,)

        # After you have face_ids and dist_array:
        F = mesh_faces.shape[0]
        face_ids_t = Tensor(face_ids.astype(np.int32))  # Label tensor, no grads needed

        # Compute per-face area inside the graph
        mesh_faces_t = Tensor(mesh_faces.astype(np.int32))
        tri_all = mesh_verts[mesh_faces_t]  # (F,3,3)
        A_f_t = triangle_area_batch(tri_all)  # (F,) area of each face

        # Get area per point by indexing A_f_t with face_ids_t
        A_pt = A_f_t[face_ids_t]
        # (N,) gives area of the face for each point's assigned face

        # Weight each point's error by the area of the face it belongs to
        weighted_dist = dist_array * (A_pt + 1)  # (N,)

        # Final loss:
        total_loss = weighted_dist.mean()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # total_loss = dist_array.mean()  # e.g. average distance

        # optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step()

        # # back in Tinygrad
        # gt_points_t = Tensor(np.array(gt_points, np.float32))

        # f_vi = mesh_faces[face_ids]  # (N,3)
        # f_vi_t = Tensor(f_vi.astype(np.int32))
        # tri_for_points = mesh_verts[f_vi_t]  # (N,3,3)

        # dist_array = point_to_triangle_distance_batch(
        #     gt_points_t, tri_for_points
        # )  # (N,)

        # # aggregate
        # F = mesh_faces.shape[0]
        # dist_np = dist_array.detach().numpy()
        # counts = np.bincount(face_ids, minlength=F)
        # sum_distances = np.bincount(face_ids, weights=dist_np, minlength=F)
        # E_f = sum_distances / np.maximum(counts, 1)
        # E_f_t = Tensor(E_f.astype(np.float32))

        # mesh_faces_t = Tensor(mesh_faces.astype(np.int32))
        # tri_all = mesh_verts[mesh_faces_t]  # (F,3,3)
        # A_f_t = triangle_area_batch(tri_all)

        # # total_loss = (E_f_t * A_f_t).sum()
        # total_loss = (E_f_t).sum()

        # optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step()

        print(f"Epoch {epoch}: Loss={total_loss.numpy().item()}")

        # Every 10 epochs, save a snapshot
        if epoch % 5 == 0:
            # save a plot
            filename = f"anim/snapshot_{epoch}.png"
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

    plot(gt_points, mesh_verts, mesh_faces)

    # After training, combine snapshots into a GIF
    images = []
    for f in snapshot_files:
        images.append(imageio.imread(f))
    imageio.mimsave("training_animation.gif", images, fps=2)
