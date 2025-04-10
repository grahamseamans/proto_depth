"""
Visualize scene generation and optimization.
Downloads test models and generates visualizations.
"""

import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request

from src import Scene


def download_models():
    """Download test models in OBJ format"""
    models_dir = Path("3d_models")
    models_dir.mkdir(exist_ok=True)

    # URLs from alecjacobson's common-3d-test-models repo
    model_urls = {
        "bunny": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj",
        "spot": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/spot.obj",
        "armadillo": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.obj",
    }

    for name, url in model_urls.items():
        path = models_dir / f"{name}.obj"
        if not path.exists():
            print(f"Downloading {name} model...")
            urllib.request.urlretrieve(url, path)


def save_depth_maps(depth_maps, output_dir, scene):
    """Save depth maps as images"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save each depth map
    for i, depth_map in enumerate(depth_maps):
        # Get raw depth values
        depth = depth_map  # Already on GPU, no batch dimension
        print(f"\nDepth map {i}:")
        print(f"Shape: {depth.shape}")
        print(f"Range: {depth.min():.2f} to {depth.max():.2f}")
        print(f"Mean: {depth.mean():.2f}")

        # Normalize depth for visualization
        # Ignore background (far plane) when normalizing
        mask = depth < depth.max()  # Foreground pixels
        if mask.any():
            min_depth = depth[mask].min()
            max_depth = depth[mask].max()
            depth_norm = torch.zeros_like(depth)
            depth_norm[mask] = (depth[mask] - min_depth) / (
                max_depth - min_depth + 1e-8
            )
        else:
            depth_norm = torch.zeros_like(depth)

        # Save as PNG
        torchvision.utils.save_image(depth_norm, output_dir / f"depth_{i}.png")

        # Save raw values
        torch.save(depth, output_dir / f"depth_{i}_raw.pt")

    # Save combined visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, depth_map in enumerate(depth_maps):
        row = i // 4
        col = i % 4
        axes[row, col].imshow(
            depth_map.squeeze().detach().cpu().numpy(), cmap="viridis"
        )
        axes[row, col].axis("off")
        axes[row, col].set_title(f"View {i}")
    plt.tight_layout()
    plt.savefig(output_dir / "depth_maps.png")
    plt.close()

    # Save scene layout visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot camera positions
    camera_positions = torch.stack(
        [cam.extrinsics.cam_pos().squeeze() for cam in scene.cameras]
    )
    ax.scatter(
        camera_positions[:, 0].cpu(),
        camera_positions[:, 2].cpu(),
        camera_positions[:, 1].cpu(),
        c="blue",
        marker="^",
        s=100,
        label="Cameras",
    )

    # Plot object positions
    positions = scene.positions.detach()
    ax.scatter(
        positions[:, 0].cpu(),
        positions[:, 2].cpu(),
        positions[:, 1].cpu(),
        c="red",
        marker="o",
        s=100,
        label="Objects",
    )

    # Add labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.legend()
    plt.savefig(output_dir / "scene_layout.png")
    plt.close()


def main():
    """Main visualization script"""
    print("Starting visualization...")

    # Download test models
    download_models()

    # Create test scene
    print("Creating test scene...")
    scene = Scene(
        num_objects=2,  # Two objects
        device="cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
    )

    # Get scene data
    print("Getting scene data...")
    batch = scene.get_batch()

    # Visualize results
    print("Generating visualizations...")
    save_depth_maps(
        batch["depth_maps"],
        "tests/output",
        scene,
    )

    print("Done! Results saved to tests/output/")


if __name__ == "__main__":
    main()
