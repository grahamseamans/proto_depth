import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams,
    PointLights,
    TexturesVertex,
)


class DepthVisualizer:
    def __init__(self, image_size=256, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.image_size = image_size

        # Setup camera (assuming a reasonable default view)
        self.cameras = PerspectiveCameras(
            focal_length=1.0,
            principal_point=((0.0, 0.0),),
            image_size=((image_size, image_size),),
            device=device,
        )

        # Setup rasterizer for depth rendering
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras, raster_settings=raster_settings
        )

    def render_depth(self, mesh):
        """Render a mesh to get its depth map"""
        fragments = self.rasterizer(mesh)
        depth = fragments.zbuf[..., 0]  # [B, H, W]
        return depth

    def visualize_comparison(self, input_depth, predicted_meshes, title=None):
        """
        Create a side-by-side visualization of input depth map and predicted scene
        Args:
            input_depth: [B, 3, H, W] input depth map tensor
            predicted_meshes: List of B Meshes objects from MeshTransformer
            title: Optional title for the plot
        """
        # Convert input depth to numpy (take first channel since they're all the same)
        input_depth_np = input_depth[0, 0].cpu().numpy()

        # Render predicted meshes
        with torch.no_grad():
            predicted_depth = self.render_depth(predicted_meshes[0])
            predicted_depth_np = predicted_depth[0].cpu().numpy()

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot input depth
        im1 = ax1.imshow(input_depth_np, cmap="viridis")
        ax1.set_title("Input Depth Map")
        plt.colorbar(im1, ax=ax1)

        # Plot predicted depth
        im2 = ax2.imshow(predicted_depth_np, cmap="viridis")
        ax2.set_title("Predicted Scene Depth")
        plt.colorbar(im2, ax=ax2)

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        return fig


def update_progress(epoch, batch, loss, input_depth, predicted_meshes, visualizer):
    """
    Update training progress with visualization
    """
    # Create visualization
    fig = visualizer.visualize_comparison(
        input_depth,
        predicted_meshes,
        title=f"Epoch {epoch}, Batch {batch}, Loss: {loss:.4f}",
    )

    # Save the figure
    plt.savefig(f"progress_epoch_{epoch}_batch_{batch}.png")
    plt.close(fig)
