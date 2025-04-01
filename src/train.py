import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch3d.loss
from pytorch3d.structures import Pointclouds
import os
from pathlib import Path
from glob import glob
from tqdm import tqdm
import time  # For manual timing
from torch.autograd import profiler  # For PyTorch profiling
import os


from .model import DepthEncoder
from .mesh_utils import MeshTransformer
from .dataloader import DataHandler, transform_fn, create_data_loaders
from .viz_exporter import VizExporter


def train(
    data_path="data/SYNTHIA-SF/SEQ1/DepthDebugLeft",
    batch_size=32,
    num_epochs=100,
    lr=1e-4,
    num_prototypes=10,
    num_slots=5,
    viz_interval=50,  # Visualize every 50 batches
    num_final_samples=10,  # Number of samples to visualize at the end
    # Regularization weights
    w_chamfer=1.0,
    w_edge=0.1,  # Reduced to allow more aggressive movement
    w_normal=0.01,
    w_laplacian=0.1,
    # Visualization options
    use_interactive_viz=True,
    # Memory optimization parameters (new)
    samples_per_slot=500,
    min_distance=0.5,
    global_weight=0.7,
    slot_weight=0.2,
    repulsion_weight=0.1,
    **kwargs,  # For backward compatibility
):
    # Create output directories for visualizations
    os.makedirs("training_progress", exist_ok=True)

    # Initialize visualization exporter if interactive visualization is enabled
    viz_exporter = None
    if use_interactive_viz:
        viz_exporter = VizExporter(local_mode=True)

    # Initialize model, mesh transformer and visualizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DepthEncoder(
        num_prototypes=num_prototypes, num_slots=num_slots, device=device
    ).to(device)
    mesh_transformer = MeshTransformer(device=device)
    # visualizer = DepthVisualizer(device=device)

    # Setup data - get all PNG files in the directory
    depth_files = [(path,) for path in glob(os.path.join(data_path, "*.png"))]
    print(f"Found {len(depth_files)} depth images")

    # Create PyTorch DataLoaders
    train_loader, test_loader = create_data_loaders(
        data=depth_files,
        transform_fn=transform_fn,
        batch_size=batch_size,
        test_ratio=0.2,
        num_workers=4,
    )

    # Print actual dataset sizes
    print(
        f"Training set: {len(train_loader.dataset)} samples, {len(train_loader)} batches"
    )
    print(f"Test set: {len(test_loader.dataset)} samples, {len(test_loader)} batches")

    # Optimizer - include both model parameters and prototype offsets
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Print memory optimization settings
    print(f"\nUsing memory-optimized settings:")
    print(f"  Points sampled per slot: {samples_per_slot} (was 5000)")
    print(f"  Min distance between slots: {min_distance}m")
    print(
        f"  Loss weights - Global: {global_weight}, Slot: {slot_weight}, Repulsion: {repulsion_weight}"
    )

    # Performance profiling settings
    enable_profiling = True  # Set to False to disable profiling
    profiling_warmup_batches = 2  # Skip initial batches for warmup
    profiling_batches = 3  # Number of batches to profile
    profiling_active = False
    profiling_batch_count = 0
    detailed_timing = {}  # For manual timing

    # Training loop
    global_batch = 0
    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0):
        model.train()
        total_loss = 0
        num_batches = 0
        tqdm.write(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Create progress bar for batches
        batch_pbar = tqdm(
            total=len(train_loader), desc="Batches", position=1, leave=False
        )

        # Iterate through batches using the DataLoader
        for batch_idx, (depth_img_3ch, points_list, original_depth) in enumerate(
            train_loader
        ):
            # Check if we should start or stop profiling
            if (
                enable_profiling
                and not profiling_active
                and batch_idx >= profiling_warmup_batches
            ):
                profiling_active = True
                profiling_batch_count = 0
                print(f"\n\n{'=' * 40}")
                print(
                    f"Starting performance profiling for {profiling_batches} batches..."
                )
                print(f"{'=' * 40}\n")
                # Start the PyTorch profiler
                prof = profiler.profile(use_cuda=True, record_shapes=True)
                prof.__enter__()

                # Reset timing dictionary for manual timing
                detailed_timing = {
                    "data_transfer": [],
                    "forward_pass": [],
                    "mesh_transform": [],
                    "chamfer_loss": [],
                    "proto_regularization": [],
                    "other_loss": [],
                    "backward_pass": [],
                    "optimizer_step": [],
                    "total_batch": [],
                }

            # Start batch timer
            batch_start = time.time()

            # Move to device
            t0 = time.time()
            depth_img_3ch = depth_img_3ch.to(device)
            points_list = [p.to(device) for p in points_list]
            if original_depth is not None:
                original_depth = original_depth.to(device)
            t1 = time.time()
            if profiling_active:
                detailed_timing["data_transfer"].append(t1 - t0)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass through encoder
            t0 = time.time()
            scales, transforms, prototype_weights, prototype_offsets = model(
                depth_img_3ch
            )
            t1 = time.time()
            if profiling_active:
                detailed_timing["forward_pass"].append(t1 - t0)

            # Transform meshes
            t0 = time.time()
            transformed_meshes = mesh_transformer.transform_mesh(
                scales, transforms, prototype_weights, prototype_offsets
            )
            t1 = time.time()
            if profiling_active:
                detailed_timing["mesh_transform"].append(t1 - t0)

            # Compute chamfer loss with memory-optimized hybrid approach
            t0 = time.time()
            chamfer_loss, global_chamfer_loss, per_slot_chamfer_loss = (
                mesh_transformer.compute_hybrid_chamfer_loss(
                    transformed_meshes,
                    points_list,
                    global_weight=global_weight,
                    slot_weight=slot_weight,
                    repulsion_weight=repulsion_weight,
                    samples_per_slot=samples_per_slot,
                    min_distance=min_distance,
                )
            )
            t1 = time.time()
            if profiling_active:
                detailed_timing["chamfer_loss"].append(t1 - t0)

            # Compute prototype regularization losses
            t0 = time.time()
            proto_edge_loss, proto_normal_loss, proto_laplacian_loss = (
                mesh_transformer.compute_prototype_regularization(prototype_offsets)
            )
            t1 = time.time()
            if profiling_active:
                detailed_timing["proto_regularization"].append(t1 - t0)

            # Get regularization for final meshes (zeros now)
            t0 = time.time()
            edge_loss, normal_loss, laplacian_loss = (
                mesh_transformer.compute_regularization_losses(transformed_meshes)
            )

            # Combine losses with weights
            loss = (
                w_chamfer * chamfer_loss
                + w_edge
                * proto_edge_loss  # Apply weights to prototype regularization instead
                + w_normal * proto_normal_loss
                + w_laplacian * proto_laplacian_loss
            )
            t1 = time.time()
            if profiling_active:
                detailed_timing["other_loss"].append(t1 - t0)

            # Standard backward pass
            t0 = time.time()
            loss.backward()
            t1 = time.time()
            if profiling_active:
                detailed_timing["backward_pass"].append(t1 - t0)

            t0 = time.time()
            optimizer.step()
            t1 = time.time()
            if profiling_active:
                detailed_timing["optimizer_step"].append(t1 - t0)

            # End batch timer
            batch_end = time.time()
            if profiling_active:
                detailed_timing["total_batch"].append(batch_end - batch_start)
                profiling_batch_count += 1

                # If we've profiled enough batches, stop profiling and print results
                if profiling_batch_count >= profiling_batches:
                    # Print manual timing results
                    print(f"\n{'=' * 60}")
                    print(f"PERFORMANCE PROFILING RESULTS (Manual Timing)")
                    print(f"{'=' * 60}")
                    print(f"Average times over {profiling_batches} batches:")
                    for key, times in detailed_timing.items():
                        avg_time = sum(times) / len(times)
                        percent = (
                            100
                            * avg_time
                            / sum(detailed_timing["total_batch"])
                            / len(detailed_timing["total_batch"])
                        )
                        print(f"  {key.ljust(20)}: {avg_time:.4f}s ({percent:.1f}%)")

                    # End and print PyTorch profiler results
                    prof.__exit__(None, None, None)
                    print(f"\n{'=' * 60}")
                    print(f"PERFORMANCE PROFILING RESULTS (PyTorch Profiler)")
                    print(f"{'=' * 60}")
                    print(
                        prof.key_averages().table(
                            sort_by="cuda_time_total", row_limit=20
                        )
                    )
                    print(f"\n{'=' * 60}\n")

                    profiling_active = False
                    enable_profiling = False  # Disable future profiling sessions

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            global_batch += 1

            # Update progress bar with current loss
            batch_pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "global": f"{global_chamfer_loss.item():.4f}",
                    "per-slot": f"{per_slot_chamfer_loss.item():.4f}",
                    "edge": f"{proto_edge_loss.item():.4f}",
                }
            )
            batch_pbar.update(1)

            # Visualize progress
            if global_batch % viz_interval == 0:
                tqdm.write(
                    f"\nBatch {num_batches}: Loss = {loss.item():.4f}, "
                    f"Global Chamfer = {global_chamfer_loss.item():.4f}, "
                    f"Per-slot Chamfer = {per_slot_chamfer_loss.item():.4f}"
                )
                # Standard 2D depth map visualization
                # update_progress(
                #     epoch + 1,
                #     global_batch,
                #     loss.item(),
                #     depth_img_3ch,
                #     transformed_meshes,
                #     visualizer,
                #     original_depth=original_depth,
                #     global_chamfer=global_chamfer_loss.item(),
                #     per_slot_chamfer=per_slot_chamfer_loss.item(),
                #     scales=scales,
                #     transforms=transforms,
                #     prototype_weights=prototype_weights,
                # )

                # Export data for interactive visualization
                if use_interactive_viz and viz_exporter is not None:
                    viz_exporter.export_visualization_data(
                        epoch=epoch + 1,
                        batch=global_batch,
                        depth_img=depth_img_3ch,
                        points_list=points_list,
                        transformed_meshes=transformed_meshes,
                        prototype_offsets=prototype_offsets,
                        prototype_weights=prototype_weights,
                        scales=scales,
                        transforms=transforms,
                        loss=loss.item(),
                        global_chamfer=global_chamfer_loss.item(),
                        per_slot_chamfer=per_slot_chamfer_loss.item(),
                    )

        # Close the batch progress bar
        batch_pbar.close()

        # End of epoch
        avg_loss = total_loss / max(num_batches, 1)  # Avoid division by zero
        tqdm.write(f"\nEpoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # # After training is complete, generate final visualizations
    # save_final_visualizations(
    #     model=model,
    #     data_loader=train_loader,
    #     num_samples=num_final_samples,
    #     visualizer=visualizer,
    #     device=device,
    # )


if __name__ == "__main__":
    train()
