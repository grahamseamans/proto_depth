import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams,
    TexturesVertex,
)

# Check if plotly is available for interactive visualizations
PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except ImportError:
    print("Plotly not available. Interactive visualizations will be disabled.")


def visualize_3d_comparison(
    input_points,
    predicted_meshes,
    epoch=None,
    batch=None,
    output_dir="training_progress_3d",
    device=None,
):
    """
    Create interactive 3D visualizations of input point cloud and predicted meshes

    Args:
        input_points: List of point clouds (each of shape [N, 3]) from the depth map
        predicted_meshes: List of pytorch3d Meshes objects
        epoch: Current epoch number (optional)
        batch: Current batch number (optional)
        output_dir: Directory to save visualizations
        device: Device to run visualization on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    # Prepare filename
    if epoch is not None and batch is not None:
        base_filename = f"{output_dir}/3d_viz_epoch_{epoch}_batch_{batch}"
    else:
        base_filename = f"{output_dir}/3d_viz"

    # Create interactive Plotly visualization if available
    if PLOTLY_AVAILABLE:
        try:
            create_interactive_visualization(
                input_points[0].detach().cpu().numpy(),
                predicted_meshes[0],
                f"{base_filename}.html",
            )
            print(f"Interactive 3D visualization saved to {base_filename}.html")
        except Exception as e:
            print(f"Failed to create interactive visualization: {e}")
    else:
        print("Plotly is not available. Interactive 3D visualization skipped.")


def create_interactive_visualization(points, mesh, filename):
    """
    Create an interactive Plotly visualization

    Args:
        points: [N, 3] numpy array of points from the point cloud
        mesh: pytorch3d Mesh object
        filename: File to save the visualization to
    """
    # Extract mesh vertices and faces
    if len(mesh.verts_list()) > 0:
        verts_list = [v.detach().cpu().numpy() for v in mesh.verts_list()]
        faces_list = [f.detach().cpu().numpy() for f in mesh.faces_list()]
    else:
        verts_list = []
        faces_list = []

    # Create figure with subplots
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "scene"}]],
        subplot_titles=["3D Scene Reconstruction"],
    )

    # Calculate scene bounds from points
    if len(points) > 0:
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
    else:
        min_bound = np.array([-100, -100, -100])
        max_bound = np.array([100, 100, 100])

    # Add mesh vertex bounds if available
    for verts in verts_list:
        if len(verts) > 0:
            min_bound = np.minimum(min_bound, np.min(verts, axis=0))
            max_bound = np.maximum(max_bound, np.max(verts, axis=0))

    # Create some padding around the bounds
    center = (min_bound + max_bound) / 2
    max_range = np.max(max_bound - min_bound) * 0.6

    # Downsample points if there are too many (for performance)
    if len(points) > 5000:
        idx = np.random.choice(len(points), 5000, replace=False)
        subsample = points[idx]
    else:
        subsample = points

    # Add point cloud
    if len(subsample) > 0:
        # Color points by depth (Z value)
        z_values = subsample[:, 2]
        if np.max(z_values) > np.min(z_values):
            z_norm = (z_values - np.min(z_values)) / (
                np.max(z_values) - np.min(z_values)
            )
            colorscale = [[0, "blue"], [0.5, "green"], [1, "red"]]
        else:
            z_norm = np.zeros_like(z_values)
            colorscale = [[0, "blue"], [1, "blue"]]

        fig.add_trace(
            go.Scatter3d(
                x=subsample[:, 0],
                y=subsample[:, 1],
                z=subsample[:, 2],
                mode="markers",
                marker=dict(size=1, color=z_norm, colorscale=colorscale, opacity=0.8),
                name="Point Cloud",
            ),
            row=1,
            col=1,
        )

    # Define colors for different slots
    slot_colors = [
        "red",
        "green",
        "blue",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "brown",
    ]

    # Add meshes for each slot
    buttons = []
    visible_default = [True] * (1 + len(verts_list))  # Point cloud + meshes

    for i, (verts, faces) in enumerate(zip(verts_list, faces_list)):
        if len(verts) > 0 and len(faces) > 0:
            color = slot_colors[i % len(slot_colors)]

            # Convert to the format Plotly expects
            i_faces = faces.copy()  # Need to modify the array

            # Add mesh
            fig.add_trace(
                go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=i_faces[:, 0],
                    j=i_faces[:, 1],
                    k=i_faces[:, 2],
                    color=color,
                    opacity=0.5,
                    name=f"Slot {i + 1}",
                ),
                row=1,
                col=1,
            )

            # Create button for toggling this slot
            visible = visible_default.copy()
            visible[i + 1] = False  # +1 because the point cloud is first
            buttons.append(
                dict(
                    label=f"Toggle Slot {i + 1}",
                    method="update",
                    args=[{"visible": visible}],
                )
            )

    # Add button to show/hide point cloud
    point_cloud_visible = visible_default.copy()
    point_cloud_visible[0] = False
    buttons.append(
        dict(
            label="Toggle Point Cloud",
            method="update",
            args=[{"visible": point_cloud_visible}],
        )
    )

    # Add button to show everything
    buttons.append(
        dict(label="Show All", method="update", args=[{"visible": visible_default}])
    )

    # Add buttons to layout
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            ),
        ]
    )

    # Update layout for consistent view
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[center[0] - max_range, center[0] + max_range]),
            yaxis=dict(range=[center[1] - max_range, center[1] + max_range]),
            zaxis=dict(range=[center[2] - max_range, center[2] + max_range]),
            aspectmode="cube",
        ),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        title="Interactive 3D Visualization",
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # Add annotation about camera orientation
    fig.add_annotation(
        text="Camera at Origin Looking along +Z",
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.0,
        showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )

    # Save as interactive HTML
    pio.write_html(fig, file=filename, auto_open=False)
