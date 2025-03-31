import os
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

        # Using actual SYNTHIA camera parameters
        # FOCAL value from dataloader.py
        focal_length = 847.630211643
        focal_length_ndc = focal_length / (image_size / 2)  # Convert to NDC

        # Create rotation matrix to flip the camera direction
        # This makes PyTorch3D camera look along +Z (like SYNTHIA) instead of -Z
        R = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device).unsqueeze(
            0
        )

        # Setup camera with correct parameters matching the dataset
        self.cameras = PerspectiveCameras(
            focal_length=((focal_length_ndc, focal_length_ndc),),
            principal_point=((0.0, 0.0),),  # Center of image
            R=R,  # This flips camera to look along +Z
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

    def visualize_comparison(
        self, input_depth, predicted_meshes, title=None, original_depth=None
    ):
        """
        Create a side-by-side visualization of input depth map and predicted scene
        Args:
            input_depth: [B, 3, H, W] input depth map tensor (normalized 0-1)
            predicted_meshes: List of B Meshes objects from MeshTransformer
            title: Optional title for the plot
            original_depth: [B, 1, H, W] original depth map tensor in meters (unnormalized)
        """
        # Render predicted meshes to get depth in meters
        with torch.no_grad():
            predicted_depth = self.render_depth(predicted_meshes[0])
            predicted_depth_np = predicted_depth[0].cpu().numpy()

        # Use original depth in meters if provided, otherwise use normalized
        if original_depth is not None:
            input_depth_np = original_depth[0, 0].cpu().numpy()
        else:
            # Fallback to normalized depth if original not provided
            input_depth_np = input_depth[0, 0].cpu().numpy()

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Determine common colormap range for consistency
        vmin = min(input_depth_np.min(), predicted_depth_np.min())
        vmax = max(input_depth_np.max(), predicted_depth_np.max())

        # Plot input depth in meters
        im1 = ax1.imshow(input_depth_np, cmap="viridis", vmin=vmin, vmax=vmax)
        ax1.set_title("Input Depth Map (meters)")
        plt.colorbar(im1, ax=ax1, label="Depth (m)")

        # Plot predicted depth in meters
        im2 = ax2.imshow(predicted_depth_np, cmap="viridis", vmin=vmin, vmax=vmax)
        ax2.set_title("Predicted Scene Depth (meters)")
        plt.colorbar(im2, ax=ax2, label="Depth (m)")

        # Mark slot centers on the predicted depth image if we have multiple slots
        mesh = predicted_meshes[0]  # First batch item's mesh
        num_slots = len(mesh.verts_list())
        if num_slots > 0:
            img_height, img_width = predicted_depth_np.shape
            # Mark each slot center with a colored cross
            for s in range(num_slots):
                verts = mesh.verts_list()[s]
                if verts.shape[0] > 0:  # Check if the mesh has any vertices
                    # Get mean position in 3D space
                    mean_pos = torch.mean(verts, dim=0).detach().cpu().numpy()

                    # Convert 3D position to 2D image coordinates (approximate)
                    # This assumes a perspective projection similar to the renderer
                    # Z is the depth (smaller Z is closer to camera)
                    if mean_pos[2] > 0:  # Only if the Z position is valid
                        # Use proper camera projection for 3D to 2D conversion
                        # Convert world coordinates to screen coordinates
                        world_coords = torch.tensor(
                            [mean_pos], device=self.device
                        ).unsqueeze(0)  # [1,1,3]
                        projected_coords = self.cameras.transform_points_screen(
                            world_coords, image_size=((img_height, img_width),)
                        )[0, 0]  # Get x,y for the point

                        x_px, y_px = (
                            int(projected_coords[0].item()),
                            int(projected_coords[1].item()),
                        )

                        # Ensure they're in the image bounds
                        if 0 <= x_px < img_width and 0 <= y_px < img_height:
                            # Draw a colored cross for each slot
                            color = plt.cm.tab10(
                                s % 10
                            )  # Use tab10 colormap for distinct colors
                            marker_size = 100
                            ax2.scatter(
                                x_px,
                                y_px,
                                s=marker_size,
                                c=[color],
                                marker="x",
                                linewidths=2,
                                zorder=10,
                                label=f"Slot {s}",
                            )

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        return fig


def update_progress(
    epoch,
    batch,
    loss,
    input_depth,
    predicted_meshes,
    visualizer,
    original_depth=None,
    global_chamfer=None,
    per_slot_chamfer=None,
    scales=None,
    transforms=None,
    prototype_weights=None,
):
    """
    Update training progress with visualization

    Args:
        epoch: Current epoch number
        batch: Current batch number
        loss: Current loss value
        input_depth: [B, 3, H, W] input normalized depth tensor
        predicted_meshes: List of predicted meshes
        visualizer: DepthVisualizer instance
        original_depth: [B, 1, H, W] original depth tensor in meters (optional)
        global_chamfer: Global chamfer loss value (optional)
        per_slot_chamfer: Per-slot chamfer loss value (optional)
        scales: [B, num_slots, 1] scale factors (optional, for debugging)
        transforms: [B, num_slots, num_prototypes, 6] transforms (optional, for debugging)
        prototype_weights: [B, num_slots, num_prototypes] weights (optional, for debugging)
    """
    # Debug slot information
    print(f"\n======= DEBUG INFO FOR EPOCH {epoch}, BATCH {batch} =======")

    # Print slot statistics from predicted meshes
    mesh = predicted_meshes[0]  # First batch item's mesh
    num_slots = len(mesh.verts_list())

    print(f"Number of slots: {num_slots}")

    # Print mesh-derived information
    print("\nSlot vertices statistics:")
    for s in range(num_slots):
        verts = mesh.verts_list()[s]
        if verts.shape[0] > 0:  # Check if the mesh has any vertices
            mean_pos = torch.mean(verts, dim=0).detach()
            min_pos = torch.min(verts, dim=0)[0].detach()
            max_pos = torch.max(verts, dim=0)[0].detach()
            print(f"  Slot {s}: Mean position: {mean_pos.tolist()}")
            print(f"         Min position: {min_pos.tolist()}")
            print(f"         Max position: {max_pos.tolist()}")
            print(f"         Bounding box size: {(max_pos - min_pos).tolist()}")

    # Print original parameter values if provided
    if scales is not None and transforms is not None and prototype_weights is not None:
        print("\nModel output parameters:")
        print(f"  Scales: {scales[0].squeeze().detach().tolist()}")  # First batch item

        print("\n  Transforms (position components):")
        for s in range(transforms.shape[1]):  # For each slot
            positions = []
            for p in range(transforms.shape[2]):  # For each prototype
                pos = (
                    transforms[0, s, p, :3].detach().tolist()
                )  # First batch, position components
                positions.append(pos)
            print(f"    Slot {s}: {positions}")

        print("\n  Prototype weights:")
        for s in range(prototype_weights.shape[1]):  # For each slot
            weights = prototype_weights[0, s].detach().tolist()  # First batch
            print(f"    Slot {s}: {weights}")
            print(
                f"    Max weight index: {torch.argmax(prototype_weights[0, s]).detach().item()}"
            )
            print(
                f"    Max weight value: {torch.max(prototype_weights[0, s]).detach().item():.4f}"
            )

    print("=================================================\n")

    # Create title with loss information
    title = f"Epoch {epoch}, Batch {batch}, Loss: {loss:.4f}"
    if global_chamfer is not None and per_slot_chamfer is not None:
        title += f"\nGlobal Chamfer: {global_chamfer:.4f}, Per-slot Chamfer: {per_slot_chamfer:.4f}"

    # Create visualization
    fig = visualizer.visualize_comparison(
        input_depth,
        predicted_meshes,
        title=title,
        original_depth=original_depth,
    )

    # Save the figure
    plt.savefig(f"training_progress/progress_epoch_{epoch}_batch_{batch}.png")
    plt.close(fig)


def save_final_visualizations(model, data_loader, num_samples, visualizer, device):
    """
    Generate and save side-by-side visualizations for the first n samples from the data loader

    Args:
        model: The trained model
        data_loader: DataLoader containing samples to visualize
        num_samples: Number of samples to visualize
        visualizer: DepthVisualizer instance for rendering
        device: Device to run model on
    """
    # Create output directory
    os.makedirs("final_results", exist_ok=True)

    # Set model to evaluation mode
    model.eval()

    # Get the first batch from the data loader
    depth_img_3ch, points_list, original_depth = next(iter(data_loader))

    # Determine how many samples we can use from this batch
    batch_size = depth_img_3ch.shape[0]
    samples_to_process = min(batch_size, num_samples)

    print(f"\nGenerating final visualizations for {samples_to_process} samples...")

    # Process each sample
    for i in range(samples_to_process):
        # Get single sample and move to device
        input_depth = depth_img_3ch[i : i + 1].to(device)

        # Get original depth if available
        orig_depth = None
        if original_depth is not None:
            orig_depth = original_depth[i : i + 1].to(device)

        # Forward pass
        with torch.no_grad():
            scales, transforms, prototype_weights, prototype_offsets = model(
                input_depth
            )
            from .mesh_utils import MeshTransformer

            mesh_transformer = MeshTransformer(device=device)
            transformed_meshes = mesh_transformer.transform_mesh(
                scales, transforms, prototype_weights, prototype_offsets
            )

        # Create visualization with original depth
        fig = visualizer.visualize_comparison(
            input_depth,
            transformed_meshes,
            title=f"Sample {i + 1}",
            original_depth=orig_depth,
        )

        # Save the figure
        plt.savefig(f"final_results/sample_{i + 1}.png")
        plt.close(fig)

    print(f"Final visualizations saved to 'final_results/' directory")
