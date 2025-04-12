"""
Visualization Data Exporter

This module exports data from the training process to the visualization server.
"""

import os
import json
import base64
import io
import datetime
import requests
import numpy as np
import torch
from PIL import Image

# Default configuration
DEFAULT_SERVER_URL = "http://localhost:5000"


class VizExporter:
    """
    Exports visualization data to the visualization server.
    """

    def __init__(self, server_url=None, local_mode=True):
        """
        Initialize the visualization exporter.

        Args:
            server_url: URL of the visualization server (default: http://localhost:5000)
            local_mode: If True, save data locally instead of sending to server (default: True)
        """
        self.server_url = server_url or DEFAULT_SERVER_URL
        self.local_mode = local_mode
        self.run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Ensure local directories exist if in local mode
        if self.local_mode:
            self.data_dir = os.path.join("viz_server", "data", self.run_id)
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(os.path.join(self.data_dir, "epochs"), exist_ok=True)

            # Create run metadata
            self.run_metadata = {"timestamp": datetime.datetime.now().isoformat()}
            self._save_run_metadata()

        print(f"Visualization exporter initialized with run_id: {self.run_id}")
        print(
            f"Mode: {'Local' if self.local_mode else 'Server'} at {self.server_url if not self.local_mode else self.data_dir}"
        )

    def _save_run_metadata(self):
        """Save run metadata to local directory."""
        if self.local_mode:
            with open(os.path.join(self.data_dir, "run_metadata.json"), "w") as f:
                json.dump(self.run_metadata, f, indent=2)

    def export_visualization_data(
        self,
        epoch,
        batch,
        depth_img,
        points_list,
        slots,  # Renamed from transformed_meshes to slots for clarity
        prototype_offsets,
        prototype_weights,
        scales,
        transforms,
        loss,
        global_chamfer=None,
        per_slot_chamfer=None,
    ):
        """
        Export visualization data to the server or local directory.

        Args:
            epoch: Current epoch number
            batch: Current batch number
            depth_img: [B, 3, H, W] input depth image tensor
            points_list: List of point clouds
            slots: SimpleMesh object containing all slot meshes
            prototype_offsets: Prototype offset tensors
            prototype_weights: Prototype weight tensors
            scales: Scale parameters
            transforms: Transform parameters
            loss: Current loss value
            global_chamfer: Global chamfer loss (optional)
            per_slot_chamfer: Per-slot chamfer loss (optional)
        """
        try:
            # Update run metadata with prototype and slot counts
            if "num_prototypes" not in self.run_metadata:
                self.run_metadata["num_prototypes"] = prototype_weights.shape[-1]
                self.run_metadata["num_slots"] = slots[0].num_verts_per_mesh().shape[0]
                self._save_run_metadata()

            # Create data structure
            epoch_id = f"epoch_{epoch}"
            batch_id = f"batch_{batch}"

            # Extract depth image (from first item in batch)
            depth_img_np = depth_img[0].detach().cpu().permute(1, 2, 0).numpy()
            depth_img_pil = Image.fromarray((depth_img_np * 255).astype(np.uint8))

            # Extract point cloud (from first item in batch)
            point_cloud = points_list[0].detach().cpu().numpy()

            # Extract meshes (get all slots, not just the first one)
            # Note: transformed_meshes is a SimpleMesh object containing all the slots
            verts_list = [v.detach().cpu().numpy() for v in slots.verts_list()]
            faces_list = [f.detach().cpu().numpy() for f in slots.faces_list()]

            # Create slot data list
            slot_data_list = []
            for i, (verts, faces) in enumerate(zip(verts_list, faces_list)):
                # Calculate bounding box and other stats
                if len(verts) > 0:
                    min_pos = np.min(verts, axis=0)
                    max_pos = np.max(verts, axis=0)
                    mean_pos = np.mean(verts, axis=0)
                    bbox_size = max_pos - min_pos
                else:
                    min_pos = max_pos = mean_pos = np.zeros(3)
                    bbox_size = np.zeros(3)

                slot_data = {
                    "vertices": verts.tolist(),
                    "faces": faces.tolist(),
                    "stats": {
                        "num_vertices": len(verts),
                        "num_faces": len(faces),
                        "mean_position": mean_pos.tolist(),
                        "min_position": min_pos.tolist(),
                        "max_position": max_pos.tolist(),
                        "bounding_box_size": bbox_size.tolist(),
                    },
                }
                slot_data_list.append(slot_data)

            # Prepare prototype data
            proto_offsets = prototype_offsets.detach().cpu().numpy()
            prototypes_data = {
                "offsets": proto_offsets.tolist(),
                "num_prototypes": proto_offsets.shape[0],
            }

            # Prepare batch metadata
            batch_metadata = {
                "epoch": epoch,
                "batch": batch,
                "loss": float(loss),
                "global_chamfer": float(global_chamfer)
                if global_chamfer is not None
                else None,
                "per_slot_chamfer": float(per_slot_chamfer)
                if per_slot_chamfer is not None
                else None,
                "scales": scales[0].detach().cpu().numpy().tolist(),
                "transforms": transforms[0].detach().cpu().numpy().tolist(),
                "prototype_weights": prototype_weights[0]
                .detach()
                .cpu()
                .numpy()
                .tolist()
                if prototype_weights is not None
                else None,
            }

            if self.local_mode:
                # Save data locally
                self._save_local(
                    epoch_id,
                    batch_id,
                    depth_img_pil,
                    point_cloud,
                    slot_data_list,
                    prototypes_data,
                    batch_metadata,
                )
            else:
                # Convert depth image to base64 for API transmission
                buffer = io.BytesIO()
                depth_img_pil.save(buffer, format="PNG")
                depth_img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                buffer = io.BytesIO()
                depth_img_pil.save(buffer, format="PNG")
                depth_img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                # Send data to server
                self._send_to_server(
                    epoch_id,
                    batch_id,
                    depth_img_base64,
                    point_cloud,
                    slot_data_list,
                    prototypes_data,
                    batch_metadata,
                )

            print(f"Exported visualization data for epoch {epoch}, batch {batch}")

        except Exception as e:
            print(f"Error exporting visualization data: {e}")

    def save_frame_json(
        self,
        out_path,
        true_camera_positions,
        true_camera_rotations,
        pred_camera_positions,
        pred_camera_rotations,
        true_object_positions,
        true_object_rotations,
        true_object_scales,
        pred_object_positions,
        pred_object_rotations,
        pred_object_scales,
        point_clouds,  # List of N arrays, each Kx3, in camera-local space
    ):
        """
        Write a single frame's data to a JSON file in the new spec.

        Args:
            out_path: Path to write the JSON file.
            *_positions, *_rotations, *_scales: Lists/arrays as described in DATA_FORMAT.md.
            point_clouds: List of N arrays (camera-local point clouds).
        """
        data = {
            "true": {
                "camera": {
                    "positions": true_camera_positions,
                    "rotations": true_camera_rotations,
                },
                "objects": {
                    "positions": true_object_positions,
                    "rotations": true_object_rotations,
                    "scales": true_object_scales,
                },
            },
            "pred": {
                "camera": {
                    "positions": pred_camera_positions,
                    "rotations": pred_camera_rotations,
                },
                "objects": {
                    "positions": pred_object_positions,
                    "rotations": pred_object_rotations,
                    "scales": pred_object_scales,
                },
            },
            "point_clouds": point_clouds,
        }

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

    def _send_to_server(
        self,
        epoch_id,
        batch_id,
        depth_img_base64,
        point_cloud,
        slot_data_list,
        prototypes_data,
        batch_metadata,
    ):
        """Send visualization data to server."""
        data = {
            "run_id": self.run_id,
            "epoch_id": epoch_id,
            "batch_id": batch_id,
            "depth_img": f"data:image/png;base64,{depth_img_base64}",
            "point_cloud": {"points": point_cloud.tolist()},
            "slots": slot_data_list,
            "prototypes": prototypes_data,
            "batch_metadata": batch_metadata,
            "run_metadata": self.run_metadata,
        }

        try:
            response = requests.post(f"{self.server_url}/api/upload", json=data)
            if response.status_code != 200:
                print(f"Error sending data to server: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Error connecting to visualization server: {e}")
