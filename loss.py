import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
import trimesh
import matplotlib.pyplot as plt
import imageio


def load_ground_truth_pointcloud(obj_path, normalize=False):
    gt_mesh = trimesh.load(obj_path)
    if normalize:
        gt_mesh.vertices -= gt_mesh.vertices.mean(axis=0)
        gt_mesh.vertices /= np.linalg.norm(gt_mesh.vertices, axis=1).max()

    sampled = gt_mesh.sample(2000)
    return sampled


def create_sphere_mesh(subdiv=2):
    sphere = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
    verts_np = np.array(sphere.vertices, dtype=np.float32)
    faces_np = np.array(sphere.faces, dtype=np.int32)
    mesh_verts = Tensor(verts_np, requires_grad=True)
    mesh_faces = faces_np
    return mesh_verts, mesh_faces


def vector_cross_batch(u, v):
    ux, uy, uz = u[:, :, 0], u[:, :, 1], u[:, :, 2]
    vx, vy, vz = v[:, :, 0], v[:, :, 1], v[:, :, 2]
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    return Tensor.stack(cx, cy, cz, dim=2)  # (N,F,3)


def brute_force_point_to_face_distance(points, tri_all):
    N = points.shape[0]
    F = tri_all.shape[0]

    p = points.reshape(N, 1, 3).expand(N, F, 3)
    v0 = tri_all[:, 0, :].reshape(1, F, 3).expand(N, F, 3)
    v1 = tri_all[:, 1, :].reshape(1, F, 3).expand(N, F, 3)
    v2 = tri_all[:, 2, :].reshape(1, F, 3).expand(N, F, 3)

    e0 = v1 - v0
    e1 = v2 - v0
    v = p - v0

    dot00 = (e0 * e0).sum(axis=2)
    dot01 = (e0 * e1).sum(axis=2)
    dot11 = (e1 * e1).sum(axis=2)
    dot02 = (e0 * v).sum(axis=2)
    dot12 = (e1 * v).sum(axis=2)

    invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-9)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    w = (dot00 * dot12 - dot01 * dot02) * invDenom

    u_exp = u.reshape(N, F, 1)
    w_exp = w.reshape(N, F, 1)
    closest_pt_inside = v0 + e0 * u_exp + e1 * w_exp
    dist_inside = ((p - closest_pt_inside) ** 2).sum(axis=2)

    dist_v0 = ((p - v0) ** 2).sum(axis=2)
    dist_v1 = ((p - v1) ** 2).sum(axis=2)
    dist_v2 = ((p - v2) ** 2).sum(axis=2)

    def edge_distance(P, A, D):
        DD = (D * D).sum(axis=2) + 1e-9
        PA = P - A
        t = ((PA * D).sum(axis=2) / DD).clip(0, 1)
        t_exp = t.reshape(N, F, 1)
        edge_closest = A + D * t_exp
        return ((P - edge_closest) ** 2).sum(axis=2)

    dist_e0 = edge_distance(p, v0, e0)
    dist_e1 = edge_distance(p, v0, e1)
    e2 = v2 - v1
    dist_e2 = edge_distance(p, v1, e2)

    all_dists = Tensor.stack(
        dist_inside, dist_v0, dist_v1, dist_v2, dist_e0, dist_e1, dist_e2, dim=2
    )
    face_min_dist = all_dists.min(axis=2)

    min_dist = face_min_dist.min(axis=1)
    chosen_face = face_min_dist.argmin(axis=1)

    return min_dist, chosen_face, face_min_dist


def triangle_area_batch(tri_all):
    # tri_all: (F,3,3)
    v0 = tri_all[:, 0, :]
    v1 = tri_all[:, 1, :]
    v2 = tri_all[:, 2, :]
    # Expand dims to handle vector ops easily
    # For a single operation, just do the cross product:
    e0 = v1 - v0
    e1 = v2 - v0
    # We'll assume a small batch dimension since we can vectorize easily
    ux, uy, uz = e0[:, 0], e0[:, 1], e0[:, 2]
    vx, vy, vz = e1[:, 0], e1[:, 1], e1[:, 2]
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    cross_norm = (cx * cx + cy * cy + cz * cz) ** 0.5
    area = 0.5 * cross_norm
    return area


if __name__ == "__main__":
    obj_path = "data/obj/xyzrgb_dragon.obj"
    obj_path = "data/obj/stanford-bunny.obj"
    gt_points = load_ground_truth_pointcloud(obj_path, normalize=True)

    mesh_verts, mesh_faces = create_sphere_mesh(subdiv=2)  # initial sphere
    print(f"GT points: {gt_points.shape}")
    print(f"Initial mesh: {mesh_verts.shape}, {mesh_faces.shape}")

    Tensor.training = True
    optimizer = Adam([mesh_verts], lr=1e-2)
    snapshot_files = []

    gt_points_t = Tensor(np.array(gt_points, np.float32))

    # Convert faces to Tensor once
    mesh_faces_t = Tensor(np.array(mesh_faces, np.int32))
    tri_all = mesh_verts[mesh_faces_t]  # (F,3,3)

    for epoch in range(100):
        # We must re-fetch tri_all each time after mesh_verts changes:
        tri_all = mesh_verts[mesh_faces_t]

        # In main training loop:
        # ...
        min_dist, chosen_face, face_min_dist = brute_force_point_to_face_distance(
            gt_points_t, tri_all
        )
        A_f_t = triangle_area_batch(tri_all)

        # Suppose chosen_face is (N,) with the index of chosen face per point
        chosen_face_np = chosen_face.numpy()  # convert to numpy
        N = chosen_face_np.shape[0]
        F = A_f_t.shape[0]

        # Create a one-hot mask for the chosen faces
        chosen_face_oh = np.zeros((N, F), dtype=np.float32)
        chosen_face_oh[np.arange(N), chosen_face_np] = 1.0
        chosen_face_oh = Tensor(chosen_face_oh)  # a constant tensor (no grad needed)

        # Now compute A_chosen as a weighted sum:
        # A_f_t: (F,)
        # chosen_face_oh: (N,F)
        # Multiply and sum over faces to select the chosen face's area
        A_chosen = (chosen_face_oh * A_f_t.reshape(1, F)).sum(axis=1)  # (N,)

        # Now A_chosen is differentiable w.r.t. A_f_t and thus w.r.t mesh_verts
        # weighted_dist = ((min_dist + 1) * (A_chosen + 1)) - 1
        # total_loss = weighted_dist.mean()
        # A_chosen = A_f_t[chosen_face]

        weighted_dist = ((min_dist + 1) * (A_chosen + 1)) - 1
        # total_loss = weighted_dist.mean()
        total_loss = face_min_dist.mean()
        total_loss = min_dist.mean()
        # total_loss = A_chosen.mean()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss={total_loss.numpy().item()}")

        # Save snapshot every 20 epochs
        if epoch % 20 == 0:

            elev, azim = 90, 270

            filename = f"anim/snapshot_{epoch}.png"
            fig = plt.figure(figsize=(12, 6))

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
            ax1.view_init(elev=elev, azim=azim)  # Set the viewing angle

            sphere_verts_np = mesh_verts.realize().detach().numpy()
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
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
            ax2.view_init(elev=elev, azim=azim)  # Set the viewing angle

            plt.tight_layout()
            plt.savefig(filename)
            plt.close(fig)
            snapshot_files.append(filename)

    # After training, combine snapshots into a GIF
    images = []
    for f in snapshot_files:
        images.append(imageio.imread(f))
    imageio.mimsave("training_animation.gif", images, fps=2)
