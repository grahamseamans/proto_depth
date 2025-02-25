import numpy as np
import tinygrad
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
    return gt_mesh.sample(2000)


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


# def point_to_triangle_distance_batch(p, tri):
#     """
#     Modified version that always tries to project inside the triangle.
#     If the projection (u,w) is outside the valid range, add a penalty that
#     encourages the face to move/orient so that the projection becomes valid.
#     """
#     v0 = tri[:, 0, :]
#     v1 = tri[:, 1, :]
#     v2 = tri[:, 2, :]

#     e0 = v1 - v0  # (N,3)
#     e1 = v2 - v0  # (N,3)
#     v = p - v0  # (N,3)

#     dot00 = (e0 * e0).sum(axis=1)
#     dot01 = (e0 * e1).sum(axis=1)
#     dot11 = (e1 * e1).sum(axis=1)
#     dot02 = (e0 * v).sum(axis=1)
#     dot12 = (e1 * v).sum(axis=1)

#     invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-9)
#     u = (dot11 * dot02 - dot01 * dot12) * invDenom
#     w = (dot00 * dot12 - dot01 * dot02) * invDenom

#     # Closest point on the plane defined by the triangle
#     u_exp = u.reshape(-1, 1)
#     w_exp = w.reshape(-1, 1)
#     closest_pt = v0 + e0 * u_exp + e1 * w_exp

#     # distance to the inside-projected point
#     dist_inside = ((p - closest_pt) ** 2).sum(axis=1)

#     # Penalty if (u,w) is outside the range [0,1] and u+w <= 1
#     # If u < 0, penalty grows with (u^2) because we want u >= 0
#     # If u > 1, penalty grows with (u-1)^2
#     # Similarly for w.
#     # Also, if u+w > 1, add penalty as well.

#     penalty_u = (u.clip(None, 0) ** 2) + ((u.clip(1, None) - 1) ** 2)
#     # penalize u out of [0,1]
#     penalty_w = (w.clip(None, 0) ** 2) + ((w.clip(1, None) - 1) ** 2)
#     # penalize w out of [0,1]

#     # Additionally, if u+w > 1, we can add penalty for that as well.
#     # One simple approach is to consider how far (u+w) is from <=1:
#     penalty_sum = ((u + w - 1.0).clip(0, None)) ** 2

#     # Combine the penalties
#     outside_penalty = penalty_u + penalty_w + penalty_sum

#     # Final distance is the inside distance plus the penalty
#     dist = dist_inside + outside_penalty

#     return dist


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
    # obj_path = "data/obj/xyzrgb_dragon.obj"
    # obj_path = "data/obj/teapot.obj"
    # obj_path = "data/obj/10014_dolphin_v2_max2011_it2.obj"

    gt_points = load_ground_truth_pointcloud(obj_path, normalize=True)  # (N,3) np
    base_verts, mesh_faces = create_sphere_mesh(subdiv=3)  # initial sphere

    print(f"GT points: {gt_points.shape}")
    print(f"Initial mesh: {base_verts.shape}, {mesh_faces.shape}")

    optimizer = Adam([base_verts], lr=1e-3)

    # We'll store snapshots in a list of filenames
    snapshot_files = []

    Tensor.training = True

    for epoch in range(1000):

        noise = Tensor.rand(shape=base_verts.shape, requires_grad=False)
        mesh_verts = base_verts + noise * 0.1

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
        weighted_dist = dist_array
        # weighted_dist = (((dist_array + 1) * (A_pt + 1)) - 1) + A_pt
        # weighted_dist = ((dist_array + 1) * (A_pt + 1)) - 1
        # weighted_dist = (((dist_array + 1) ** 2) * (A_pt + 1)) - 1

        penalty_shrink = ((dist_array + 1) * (A_pt + 1)) - 1

        # Encourage expansion:
        # If dist is near zero, we want a negative penalty if area is too small
        # E.g. something that becomes strongly negative (a "reward") as dist->0 but area remains small.

        alpha = 0.1  # tune this
        # Something that is ~ -alpha * (1/(dist+1)) * (A_pt+1)
        #   - If dist=0, term= -alpha*(A_pt+1)
        #   - If dist is large, term ~ 0
        #   - So if you fit well (dist=0), the bigger the area A_pt is, the more negative this is — i.e. it "wants" to grow.
        # penalty_grow = -alpha * (1.0 / (dist_array + 1.0)) * (A_pt + 1.0)
        # penalty_grow = (1.0 / (dist_array + 1.0)) * (A_pt + 1.0) - 1
        alpha = 5.0  # bigger -> sharper cutoff
        # penalty_grow = (A_pt + 1.0) * Tensor.exp(-alpha * dist_array)

        #######################################################################

        # can I just make it so that when it wants to get smaller, it also wants to get more equilateral?
        # this would lead to a similar oucome as the even mesh normlaizer in areas with high loss

        #######################################################################

        loss = penalty_shrink  # + penalty_grow

        # Final loss:
        total_loss = loss.mean()
        # total_loss = (A_f_t).mean()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss={total_loss.numpy().item()}")

        # mesh_verts += Tensor.rand(shape=mesh_verts.shape, requires_grad=False)

        # Every 10 epochs, save a snapshot
        if epoch % 20 == 0:
            # save a plot

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

    plot(gt_points, mesh_verts, mesh_faces)

    # After training, combine snapshots into a GIF
    images = []
    for f in snapshot_files:
        images.append(imageio.imread(f))
    imageio.mimsave("training_animation.gif", images, fps=2)
