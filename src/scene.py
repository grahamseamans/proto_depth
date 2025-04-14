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
import kaolin.ops.mesh
import kaolin.render.mesh as mesh_render
import kaolin.metrics.pointcloud as kaolin_metrics
from kaolin.render.camera import Camera

from .point_cloud import render_depth_and_pointcloud


class Scene:
    """
    Scene representation that handles its own state and optimization.
    Combines data generation, rendering, and energy optimization.
    """

    # Camera attributes
    true_cameras: List[Camera]
    pred_cameras: List[Camera]
    camera_positions: torch.Tensor

    # Scene state tensors
    true_positions: torch.Tensor  # [num_frames, num_objects, 3]
    true_rotations: torch.Tensor  # [num_frames, num_objects, 3]
    true_scales: torch.Tensor  # [num_frames, num_objects, 1]
    pred_positions: torch.Tensor  # [num_frames, num_objects, 3]
    pred_rotations: torch.Tensor  # [num_frames, num_objects, 3]
    pred_scales: torch.Tensor  # [num_frames, num_objects, 1]

    # Mesh attributes
    true_mesh_verts: torch.Tensor  # [V, 3]
    true_mesh_faces: torch.Tensor  # [F, 3]
    pred_mesh_verts: torch.Tensor  # [V, 3]
    pred_mesh_faces: torch.Tensor  # [F, 3]

    # Other attributes
    device: torch.device
    num_objects: int
    num_frames: int

    def __init__(
        self,
        num_objects: int = 2,
        num_frames: int = 30,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize scene with random object parameters.

        Args:
            num_objects: Number of objects in scene
            num_frames: Number of time frames
            device: Device to use for computations
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_objects = num_objects
        self.num_frames = num_frames

        # Create fixed camera positions
        self.camera_positions = torch.tensor(
            [
                [1.0, 0.3, 0.0],  # Side view
                [0.0, 0.3, 1.0],  # Side view
                # [0.7, 0.3, 0.7],  # Corner view
            ],
            device=device,
        )

        # Create true cameras looking at origin
        self.true_cameras = []
        for pos in self.camera_positions:
            camera = Camera.from_args(
                eye=pos,
                at=torch.zeros(3, device=device),  # Look at origin
                up=torch.tensor([0.0, 1.0, 0.0], device=device),
                fov=60 * np.pi / 180,
                width=256,
                height=256,
                near=0.001,  # 1mm near plane
                far=10.0,  # 10m far plane
                device=device,
            )
            self.true_cameras.append(camera)

        # True scene state over time (what actually exists)
        self.true_positions = torch.zeros(num_frames, num_objects, 3, device=device)
        self.true_rotations = torch.zeros(num_frames, num_objects, 3, device=device)
        self.true_scales = torch.ones(num_frames, num_objects, 1, device=device)

        # Generate motion paths
        for t in range(num_frames):
            time = t / (num_frames - 1)  # Normalize to [0, 1]

            # Object 1: Circle motion
            angle = 2 * np.pi * time
            self.true_positions[t, 0] = torch.tensor(
                [
                    0.3 * torch.cos(torch.tensor(angle)),  # X
                    0.0,  # Y
                    0.3 * torch.sin(torch.tensor(angle)),  # Z
                ],
                device=device,
            )

            # Object 2: Figure-8 motion
            self.true_positions[t, 1] = torch.tensor(
                [
                    0.2 * torch.cos(torch.tensor(angle)),  # X
                    0.0,  # Y
                    0.2 * torch.sin(torch.tensor(2 * angle)),  # Z
                ],
                device=device,
            )

            # Simple rotation (Y-axis spin)
            self.true_rotations[t, :, 1] = 2 * np.pi * time

        # true_cameras already created above

        # Load mesh once
        self.true_mesh_verts, self.true_mesh_faces = self._load_mesh("3d_models")

        # Predicted scene state (what we think exists)
        # Initialize with noisy versions of true positions
        noise = torch.randn_like(self.true_positions) * 0.1
        self.pred_positions = self.true_positions + noise
        self.pred_rotations = self.true_rotations.clone()
        self.pred_scales = self.true_scales.clone()

        # Initialize predicted cameras with random positions and orientations
        self.pred_cameras = []
        radius = torch.norm(self.camera_positions[0])  # Use similar radius
        for _ in range(2):
            # Random position on sphere
            theta = torch.rand(1, device=device) * 2 * np.pi  # Random azimuth
            phi = (
                torch.rand(1, device=device) * np.pi / 3 + np.pi / 6
            )  # 30-60° elevation
            pos = torch.tensor(
                [
                    radius * torch.sin(phi) * torch.cos(theta),
                    radius * torch.cos(phi),  # Y is up
                    radius * torch.sin(phi) * torch.sin(theta),
                ],
                device=device,
            ).squeeze()

            # Random look-at point near origin
            look_at = torch.randn(3, device=device) * 0.3

            # Create camera
            camera = Camera.from_args(
                eye=pos,
                at=look_at,  # Look near but not exactly at origin
                up=torch.tensor([0.0, 1.0, 0.0], device=device),
                fov=60 * np.pi / 180,
                width=256,
                height=256,
                near=0.001,
                far=10.0,
                device=device,
            )
            self.pred_cameras.append(camera)

        # Initialize predicted mesh to true mesh - maybe learn this later
        self.pred_mesh_verts = self.true_mesh_verts
        self.pred_mesh_faces = self.true_mesh_faces

    @staticmethod
    def _download_models(models_dir: Path) -> None:
        """Download test models in OBJ format"""
        models_dir = Path(models_dir)
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

    def _load_mesh(self, models_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Load OBJ models"""
        models_dir = Path(models_dir)

        # Download models if they don't exist
        if not models_dir.exists() or not any(models_dir.glob("*.obj")):
            self._download_models(models_dir)

        # Try loading available models
        model_files = {
            "bunny": "bunny.obj",
            "spot": "spot.obj",
            "armadillo": "armadillo.obj",
        }

        # Use first available model
        for name, filename in model_files.items():
            path = models_dir / filename
            if path.exists():
                print(f"Loading {name} model...")
                mesh = kaolin.io.obj.import_mesh(str(path))
                vertices = mesh.vertices.to(dtype=torch.float32, device=self.device)
                faces = mesh.faces.to(dtype=torch.int64, device=self.device)
                return vertices, faces

        raise FileNotFoundError(f"Failed to load any models from {models_dir}")

    def get_ground_truth_clouds(self, frame_idx: int) -> List[torch.Tensor]:
        """
        Get point clouds from true scene using depth maps for a specific frame.
        Each point cloud is in its camera's local space.
        This is what cameras actually see.

        Args:
            frame_idx: Which frame to get clouds for
        """
        point_clouds = []
        for camera in self.true_cameras:
            # Get depth map and point cloud from true scene state
            _, points = render_depth_and_pointcloud(
                camera,
                self.true_mesh_verts,
                self.true_mesh_faces,
                self.true_positions[frame_idx],
                self.true_rotations[frame_idx],
                self.true_scales[frame_idx],
                self.device,
            )
            point_clouds.append(points)
        return point_clouds

    def compute_energy(self, frame_idx: int) -> torch.Tensor:
        """
        Compute energy between ground truth and predicted state for a frame.
        Uses sided_distance since cameras only see visible surfaces.

        Args:
            frame_idx: Which frame to compute energy for
        """
        # Get ground truth clouds for this frame
        ground_truth_clouds = self.get_ground_truth_clouds(frame_idx)

        total_energy = 0

        # For each camera view
        for i, target in enumerate(ground_truth_clouds):
            # Transform ground truth points using predicted camera position
            if target.ndim == 2:
                target = target.unsqueeze(0)  # Add batch dimension
            target_world = self.pred_cameras[i].extrinsics.transform(target)

            # Get predicted points from this camera view
            _, pred_points = render_depth_and_pointcloud(
                self.pred_cameras[i],
                self.pred_mesh_verts,
                self.pred_mesh_faces,
                self.pred_positions[frame_idx],
                self.pred_rotations[frame_idx],
                self.pred_scales[frame_idx],
                self.device,
            )
            if pred_points.ndim == 2:
                pred_points = pred_points.unsqueeze(0)

            # Make contiguous for CUDA ops
            target_world = target_world.contiguous()
            pred_points = pred_points.contiguous()

            # Use sided_distance since cameras only see visible surfaces
            dist_to_pred = kaolin_metrics.sided_distance(target_world, pred_points)[0]
            total_energy += dist_to_pred.mean()

        return total_energy / len(ground_truth_clouds)
