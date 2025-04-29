"""
Scene representation for 4D reality learning system.
Encapsulates state, rendering, and optimization in one place.
"""

import torch
import numpy as np
from pathlib import Path
import urllib.request
from typing import List
import kaolin.io.obj
import kaolin.metrics.pointcloud as kaolin_metrics
from kaolin.render.camera import Camera as KaolinCamera
from kaolin.math.quat import quat_unit, rot33_from_quat
from .point_cloud import render_pointcloud


class Scene:
    """
    Scene representation that handles its own state and optimization.
    Combines data generation, rendering, and energy optimization.
    """

    def __init__(
        self,
        num_objects: int = 2,
        num_frames: int = 30,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_objects = num_objects
        self.num_frames = num_frames

        # Camera positions (fixed for true, learnable for pred)
        camera_positions = torch.tensor(
            [
                [1.0, 0.3, 0.0],  # Side view
                [0.0, 0.3, 1.0],  # Side view
            ],
            device=device,
        )
        num_cameras = camera_positions.shape[0]

        # True cameras: fixed positions, look at origin, up=[0,1,0]
        self.true_camera_positions = camera_positions.clone()
        self.true_camera_ats = torch.zeros_like(camera_positions)
        self.true_camera_ups = torch.tensor(
            [[0.0, 1.0, 0.0]] * num_cameras, device=device
        )
        self.true_cameras = [
            KaolinCamera.from_args(
                eye=self.true_camera_positions[i],
                at=self.true_camera_ats[i],
                up=self.true_camera_ups[i],
                fov=60 * np.pi / 180,
                width=256,
                height=256,
                near=0.001,
                far=10.0,
                device=device,
            )
            for i in range(num_cameras)
        ]

        # Predicted cameras: learnable positions and quaternions
        noise = torch.randn_like(camera_positions) * 0.05
        self.pred_camera_positions = (
            (camera_positions + noise).clone().detach().requires_grad_(True)
        )
        self.pred_camera_ats = torch.zeros_like(camera_positions)
        self.pred_camera_ups = torch.tensor(
            [[0.0, 1.0, 0.0]] * num_cameras, device=device
        )
        self.pred_cameras = [
            KaolinCamera.from_args(
                eye=self.pred_camera_positions[i],
                at=self.pred_camera_ats[i],
                up=self.pred_camera_ups[i],
                fov=60 * np.pi / 180,
                width=256,
                height=256,
                near=0.001,
                far=10.0,
                device=device,
            )
            for i in range(num_cameras)
        ]

        # True scene state over time (what actually exists)
        self.true_positions = torch.zeros(num_frames, num_objects, 3, device=device)
        self.true_rotations = torch.zeros(
            num_frames, num_objects, 4, device=device
        )  # quaternion (x, y, z, w)
        self.true_scales = torch.ones(num_frames, num_objects, 1, device=device)

        # Generate motion paths and quaternion rotations
        for t in range(num_frames):
            time = torch.tensor(
                t / (num_frames - 1), device=device
            )  # Normalize to [0, 1]
            angle = 2 * torch.pi * time
            self.true_positions[t, 0] = torch.stack(
                [
                    0.3 * torch.cos(angle),
                    torch.tensor(0.0, device=device),
                    0.3 * torch.sin(angle),
                ]
            )
            self.true_positions[t, 1] = torch.stack(
                [
                    0.2 * torch.cos(angle),
                    torch.tensor(0.0, device=device),
                    0.2 * torch.sin(2 * angle),
                ]
            )
            theta = 2 * torch.pi * time
            axis = torch.tensor([0.0, 1.0, 0.0], device=device)
            half_theta = theta / 2
            sin_half_theta = torch.sin(half_theta)
            q_y = torch.stack(
                [
                    axis[0] * sin_half_theta,
                    axis[1] * sin_half_theta,
                    axis[2] * sin_half_theta,
                    torch.cos(half_theta),
                ]
            )
            self.true_rotations[t, 0] = quat_unit(q_y)
            self.true_rotations[t, 1] = quat_unit(q_y)

        # Load mesh once
        self.true_mesh_verts, self.true_mesh_faces = self._load_mesh("3d_models")

        # Predicted scene state (what we think exists)
        noise = torch.randn_like(self.true_positions) * 0.1
        self.pred_positions = (
            (self.true_positions + noise).clone().detach().requires_grad_(True)
        )
        self.pred_rotations = self.true_rotations.clone().detach().requires_grad_(True)
        self.pred_scales = self.true_scales.clone().detach().requires_grad_(True)

        # Initialize predicted mesh to true mesh
        self.pred_mesh_verts = self.true_mesh_verts
        self.pred_mesh_faces = self.true_mesh_faces

    @staticmethod
    def _download_models(models_dir: Path) -> None:
        models_dir = Path(models_dir)
        models_dir.mkdir(exist_ok=True)
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

    def _load_mesh(self, models_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        models_dir = Path(models_dir)
        if not models_dir.exists() or not any(models_dir.glob("*.obj")):
            self._download_models(models_dir)
        model_files = {
            "bunny": "bunny.obj",
            "spot": "spot.obj",
            "armadillo": "armadillo.obj",
        }
        for name, filename in model_files.items():
            path = models_dir / filename
            if path.exists():
                print(f"Loading {name} model...")
                mesh = kaolin.io.obj.import_mesh(str(path))
                vertices = mesh.vertices.to(dtype=torch.float32, device=self.device)
                faces = mesh.faces.to(dtype=torch.int64, device=self.device)
                return vertices, faces
        raise FileNotFoundError(f"Failed to load any models from {models_dir}")

    def get_point_cloud_pair(
        self, camera_idx: int, frame_idx: int, camera_space: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gt_camera = self.true_cameras[camera_idx]
        pred_camera = self.pred_cameras[camera_idx]
        with torch.no_grad():
            gt_points_cam = render_pointcloud(
                gt_camera,
                self.true_mesh_verts,
                self.true_mesh_faces,
                self.true_positions[frame_idx],
                self.true_rotations[frame_idx],
                self.true_scales[frame_idx],
                self.device,
            )
        pred_points_cam = render_pointcloud(
            pred_camera,
            self.pred_mesh_verts,
            self.pred_mesh_faces,
            self.pred_positions[frame_idx],
            self.pred_rotations[frame_idx],
            self.pred_scales[frame_idx],
            self.device,
        )
        if camera_space:
            return gt_points_cam, pred_points_cam
        else:
            gt_cam2world = gt_camera.extrinsics.inv_view_matrix()
            pred_cam2world = pred_camera.extrinsics.inv_view_matrix()

            # remove batch dims from extrinsics
            gt_cam2world = gt_cam2world.squeeze(0)
            pred_cam2world = pred_cam2world.squeeze(0)

            gt_points_h = torch.cat(
                [gt_points_cam, torch.ones_like(gt_points_cam[:, :1])], dim=1
            )
            pred_points_h = torch.cat(
                [pred_points_cam, torch.ones_like(pred_points_cam[:, :1])], dim=1
            )
            gt_points_world_h = gt_points_h @ gt_cam2world.T
            pred_points_world_h = pred_points_h @ pred_cam2world.T
            gt_points_world = gt_points_world_h[:, :3]
            pred_points_world = pred_points_world_h[:, :3]
            return gt_points_world, pred_points_world

    def compute_energy(self) -> torch.Tensor:
        total_energy = 0.0
        count = 0
        for frame_idx in range(self.num_frames):
            num_cameras = len(self.true_cameras)
            for camera_idx in range(num_cameras):
                gt_points_world, pred_points_world = self.get_point_cloud_pair(
                    camera_idx, frame_idx, camera_space=True
                )
                if gt_points_world.ndim == 3 and gt_points_world.shape[0] == 1:
                    gt_points_world = gt_points_world.squeeze(0)
                if pred_points_world.ndim == 3 and pred_points_world.shape[0] == 1:
                    pred_points_world = pred_points_world.squeeze(0)
                gt_points_world = gt_points_world.contiguous()
                pred_points_world = pred_points_world.contiguous()
                if gt_points_world.ndim == 2:
                    gt_points_world = gt_points_world.unsqueeze(0)
                if pred_points_world.ndim == 2:
                    pred_points_world = pred_points_world.unsqueeze(0)
                chamfer = kaolin_metrics.chamfer_distance(
                    gt_points_world, pred_points_world, w1=1.0, w2=1.0, squared=True
                )
                chamfer = chamfer.squeeze(0)
                chamfer_mean = chamfer.mean()
                total_energy += chamfer_mean
                count += 1
        return (
            total_energy / count if count > 0 else torch.tensor(0.0, device=self.device)
        )
