import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch3d.loss
from pytorch3d.structures import Pointclouds
import os
from pathlib import Path
from glob import glob

from .model import DepthEncoder
from .mesh_utils import MeshTransformer
from .dataloader import DataHandler, transform_fn
from .visualize import DepthVisualizer, update_progress


def train(
    data_path="data/SYNTHIA-SF/SEQ1/DepthDebugLeft",
    batch_size=32,
    num_epochs=100,
    lr=1e-4,
    num_prototypes=10,
    num_slots=5,
    viz_interval=50,  # Visualize every 50 batches
):
    # Create output directory for visualizations
    os.makedirs("training_progress", exist_ok=True)

    # Initialize model, mesh transformer and visualizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DepthEncoder(num_prototypes=num_prototypes, num_slots=num_slots).to(device)
    mesh_transformer = MeshTransformer(device=device)
    visualizer = DepthVisualizer(device=device)

    # Setup data - get all PNG files in the directory
    depth_files = [(path,) for path in glob(os.path.join(data_path, "*.png"))]
    print(f"Found {len(depth_files)} depth images")
    data_handler = DataHandler(
        data=depth_files, transform_fn=transform_fn, test_ratio=0.2, num_workers=4
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    global_batch = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        while True:
            try:
                # Get batch
                print("Getting batch...", end="\r")
                depth_img_3ch, points_t, _ = data_handler.get_train_batch(batch_size)
                print(
                    f"Got batch: depth_img shape={depth_img_3ch.shape}, points shape={points_t.shape}"
                )

                # Move to device
                depth_img_3ch = depth_img_3ch.to(device)
                points_t = points_t.to(device)

                # Forward pass through encoder
                print("Running encoder...", end="\r")
                scales, transforms, prototype_weights = model(depth_img_3ch)
                print(
                    f"Encoder output: scales={scales.shape}, transforms={transforms.shape}, weights={prototype_weights.shape}"
                )

                # Transform meshes
                print("Transforming meshes...", end="\r")
                transformed_meshes = mesh_transformer.transform_mesh(
                    scales, transforms, prototype_weights
                )
                print("Meshes transformed")

                # Compute chamfer loss
                print("Computing loss...", end="\r")
                loss = mesh_transformer.compute_chamfer_loss(
                    transformed_meshes, points_t
                )
                print(f"Loss computed: {loss.item():.4f}")

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                global_batch += 1

                # Visualize progress
                if global_batch % viz_interval == 0:
                    print(f"\nBatch {num_batches}: Loss = {loss.item():.4f}")
                    update_progress(
                        epoch + 1,
                        global_batch,
                        loss.item(),
                        depth_img_3ch,
                        transformed_meshes,
                        visualizer,
                    )

            except StopIteration:
                break

        # End of epoch
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Reset data handler for next epoch
        data_handler.reset_train()


if __name__ == "__main__":
    train()
