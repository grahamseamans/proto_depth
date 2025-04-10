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


def save_scene_data(scene, output_dir):
    """Save depth maps and point clouds from each camera view"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get point clouds from each camera
    point_clouds = []
    for i, camera in enumerate(scene.cameras):
        # Get depth map
        depth_map = scene._render_depth(camera)
        print(f"\nCamera {i}:")
        print(f"Depth range: {depth_map.min():.2f} to {depth_map.max():.2f}")

        # Save raw depth map
        torch.save(depth_map, output_dir / f"depth_{i}_raw.pt")

        # Convert to point cloud using our actual point cloud code
        from src.core.point_cloud import depth_to_pointcloud

        points = depth_to_pointcloud(depth_map, camera)
        point_clouds.append(points)

        # Save point cloud
        points = points.squeeze().detach().cpu()
        print(f"Points shape: {points.shape}")
        torch.save(points, output_dir / f"points_{i}.pt")

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

    return point_clouds


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

    # Save scene data (depth maps and point clouds)
    print("Saving scene data...")
    save_scene_data(scene, "tests/output")

    print("Done! Results saved to tests/output/")


if __name__ == "__main__":
    main()
