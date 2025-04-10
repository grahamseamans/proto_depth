"""
Visualize scene generation and optimization.
Downloads test models and generates visualizations.
"""

import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request

from core import Scene


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


def save_depth_maps(depth_maps, output_dir):
    """Save depth maps as images"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save each depth map
    for i, depth_map in enumerate(depth_maps):
        # Get raw depth values
        depth = depth_map.squeeze(0)  # Keep on GPU, remove batch dimension
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


def main():
    """Main visualization script"""
    print("Starting visualization...")

    # Download test models
    download_models()

    # Create test scene
    print("Creating test scene...")
    scene = Scene(
        num_objects=2,  # Two objects
        device="cuda",  # Use GPU
    )

    # Get scene data
    print("Getting scene data...")
    batch = scene.get_batch()

    # Visualize results
    print("Generating visualizations...")
    save_depth_maps(
        batch["depth_maps"],
        "tests/output",
    )

    print("Done! Results saved to tests/output/")


if __name__ == "__main__":
    main()
