#!/usr/bin/env python3

"""
Demonstration of partial decimation + partial subdivision in PyMeshLab,
with a custom per-face error stored as a user-defined attribute 'nn_error'.

We:
  1) Load a sphere mesh from trimesh.
  2) Create a PyMeshLab Mesh, add a user-defined face attribute "nn_error".
  3) Decide which faces to freeze (protect) for decimation, and which to subdivide,
     using the "select_face_by_condition" filter with our custom attribute.
  4) Plot the original mesh vs. the final mesh in Matplotlib.

Requirements:
  pip install pymeshlab trimesh matplotlib
"""

import numpy as np
import pymeshlab
import matplotlib.pyplot as plt


def create_sphere_verts_faces(subdiv=2, radius=1.0):
    """
    Create an icosphere using trimesh, return (verts, faces) as NumPy arrays.
    """
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=subdiv, radius=radius)
    verts = np.array(sphere.vertices, dtype=np.float32)
    faces = np.array(sphere.faces, dtype=np.int32)
    return verts, faces


def partial_decimate_by_attribute(
    ms: pymeshlab.MeshSet,
    attribute_name: str,
    freeze_threshold: float,
    targetfacenum: int = 500,
):
    """
    1) Select faces with "fa.<attribute_name> >= freeze_threshold".
    2) Freeze them (so decimation won't touch them).
    3) Apply quadric edge collapse decimation on the rest.

    Note: we do *not* rely on 'qualityweight=True' because we don't have a standard
    face-quality array, but a user-defined attribute. So we decimate by
    "region-based" selection & freeze.
    """
    # 1) Select the faces that have "nn_error >= freeze_threshold"
    #    cond can reference attributes as "fa.<attribute_name>"
    cond_str = f"fa.{attribute_name} >= {freeze_threshold}"
    ms.apply_filter("select_face_by_condition", cond=cond_str)

    # 2) Freeze them
    ms.apply_filter("meshing_flag_faces_by_selection", flagtype="Frozen", value=True)

    # 3) Decimate the rest
    ms.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=targetfacenum,
        # 'selected=False' => acts on entire mesh,
        # but won't modify frozen faces
        preserveboundary=False,
        preservenormal=False,
        optimalplacement=True,
        planarquadric=False,
        autoclean=True,
    )
    return ms


def partial_subdivide_by_attribute(
    ms: pymeshlab.MeshSet, attribute_name: str, refine_threshold: float
):
    """
    1) Clear old selection
    2) Select faces with "fa.<attribute_name> >= refine_threshold"
    3) Subdivide only those faces (selected=True) with one iteration of Loop Subdivision.
    """
    # Clear old selection
    ms.apply_filter("select_all_faces", select=False)

    cond_str = f"fa.{attribute_name} >= {refine_threshold}"
    ms.apply_filter("select_face_by_condition", cond=cond_str)

    ms.apply_filter("meshing_loop_subdivision", iterations=1, selected=True)
    return ms


def plot_mesh_matplotlib(ax, verts, faces, title=""):
    """
    Plot a triangular mesh in an existing Matplotlib 3D axis using plot_trisurf.
    """
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        color="gray",
        edgecolor="none",
        alpha=0.8,
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)


def main():
    # 1) Create an icosphere
    original_verts, original_faces = create_sphere_verts_faces(subdiv=3, radius=1.0)
    print("[INFO] Original mesh:", original_verts.shape, original_faces.shape)

    # 2) Suppose we have a random face "error" from some NN
    face_error = np.random.rand(len(original_faces)).astype(np.float32)

    # 3) Create a PyMeshLab Mesh from these arrays
    mesh = pymeshlab.Mesh(vertex_matrix=original_verts, face_matrix=original_faces)

    # 4) Add a user-defined face scalar attribute (e.g. "nn_error").
    #    Then we can store face_error in that attribute array.
    attribute_name = "nn_error"
    mesh.add_face_scalar_attribute(attribute_name)
    arr = mesh.face_scalar_attribute_array(attribute_name)
    arr[:] = face_error  # fill it with our custom error values

    # 5) Add the mesh to a MeshSet
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh, "BaseMesh")

    # 6) PARTIAL DECIMATION: freeze faces with "nn_error >= 0.5"
    partial_decimate_by_attribute(
        ms, attribute_name=attribute_name, freeze_threshold=0.5, targetfacenum=500
    )

    # 7) PARTIAL SUBDIVISION: subdivide faces with "nn_error >= 0.8"
    partial_subdivide_by_attribute(
        ms, attribute_name=attribute_name, refine_threshold=0.8
    )

    # 8) Retrieve final mesh
    out_mesh = ms.current_mesh()
    new_verts = np.array(out_mesh.vertex_matrix(), dtype=np.float32)
    new_faces = np.array(out_mesh.face_matrix(), dtype=np.int32)
    print("[INFO] Final mesh:", new_verts.shape, new_faces.shape)

    # 9) Plot side-by-side with Matplotlib
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_mesh_matplotlib(ax1, original_verts, original_faces, title="Original")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_mesh_matplotlib(ax2, new_verts, new_faces, title="Decimated+Subdivided")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
