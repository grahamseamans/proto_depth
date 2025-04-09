"""
Visualize dataloader output on GPU.
Downloads test models and generates depth map visualizations.
"""

import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
import kaolin.io.obj
import tqdm


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

    models = {}
    for name, url in model_urls.items():
        path = models_dir / f"{name}.obj"
        if not path.exists():
            print(f"Downloading {name} model...")
            urllib.request.urlretrieve(url, path)
        print(f"Loading {name} model...")
        models[name] = kaolin.io.obj.import_mesh(str(path))

    return models


def save_depth_maps(depth_maps, output_dir):
    """Save raw depth maps as images and tensors"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save each depth map separately
    for i, depth_map in enumerate(depth_maps):
        # Get raw depth values
        depth = depth_map.squeeze(0).cpu()  # Remove batch dimension
        print(f"\nDepth map {i}:")
        print(f"Shape: {depth.shape}")
        print(f"Range: {depth.min():.2f} to {depth.max():.2f}")
        print(f"Mean: {depth.mean():.2f}")

        # Normalize to [0, 1] for visualization
        # Ignore background (0.0) when normalizing
        mask = depth < 0  # Foreground pixels have negative depth
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

    # Download and load models
    models = download_models()

    from src import SceneDataset

    # Create dataset with minimal setup
    print("Creating test scene...")
    dataset = SceneDataset(
        num_scenes=1,  # One test scene
        num_frames=8,  # 8 viewpoints around scene
        num_objects=2,  # Two objects
        device="cuda",  # Use GPU
    )

    # Get scene data
    print("Getting scene data...")
    scene_data = dataset[0]

    # Visualize results
    print("Generating visualizations...")
    save_depth_maps(
        scene_data["depth_maps"],
        "tests/output",
    )

    print("Done! Results saved to tests/output/")


if __name__ == "__main__":
    main()
