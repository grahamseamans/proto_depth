"""
Scene representation for 4D reality learning system.
Encapsulates state, rendering, and optimization in one place.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import kaolin.io.obj
import kaolin.ops.mesh
import kaolin.render.mesh as mesh_render
import kaolin.metrics.pointcloud as kaolin_metrics
from kaolin.render.camera import Camera

from .point_cloud import depth_to_pointcloud


class Scene:
    """
    Scene representation that handles its own state and optimization.
    Combines data generation, rendering, and energy optimization.
    """

    def __init__(self, num_objects=2, device=None):
        """
        Initialize scene with random object parameters.

        Args:
            num_objects: Number of objects in scene
            device: Device to use for computations
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_objects = num_objects

        # Load mesh data (cached since it never changes)
        self.vertices, self.faces = self._load_mesh("3d_models")

        # Initialize optimizable parameters
        self.positions = nn.Parameter(
            (torch.rand(num_objects, 3, device=device) * 2 - 1) * 1.0  # [-1, 1] meters
        )
        self.rotations = nn.Parameter(
            (torch.rand(num_objects, 3, device=device) - 0.5)
            * (torch.pi / 2)  # [-π/4, π/4] radians
        )
        self.scales = nn.Parameter(
            torch.rand(num_objects, 1, device=device) * 2.0 + 1.0  # [1.0, 3.0]
        )

        # Setup optimizer
        self.optimizer = optim.Adam(
            [self.positions, self.rotations, self.scales], lr=0.01
        )

        # Generate camera path (fixed viewpoints)
        self.cameras = self._generate_camera_path()

        # For tracking progress
        self.iteration = 0
        self.loss_history = []

    def _load_mesh(self, models_dir):
        """Load OBJ models"""
        models_dir = Path(models_dir)

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

        raise FileNotFoundError(
            f"No models found in {models_dir}. "
            "Run scripts/visualize_dataloader.py first to download models."
        )

    def _generate_camera_path(self):
        """Generate X-shaped camera path through scene"""
        cameras = []
        t = torch.linspace(-1, 1, 4, device=self.device)  # 8 total views

        # First diagonal of X: (-5,3,-5) to (5,3,5)
        for i in range(4):
            pos = torch.tensor(
                [
                    t[i] * 5,  # x: -5 to 5
                    3.0,  # fixed height
                    t[i] * 5,  # z: -5 to 5
                ],
                device=self.device,
            )

            camera = Camera.from_args(
                eye=pos,
                at=torch.zeros(3, device=self.device),
                up=torch.tensor([0.0, 1.0, 0.0], device=self.device),
                fov=60 * np.pi / 180,
                width=256,
                height=256,
                near=1e-2,
                far=100.0,
                device=self.device,
            )
            cameras.append(camera)

        # Second diagonal: (-5,3,5) to (5,3,-5)
        for i in range(4):
            pos = torch.tensor(
                [
                    t[i] * 5,  # x: -5 to 5
                    3.0,  # fixed height
                    -t[i] * 5,  # z: 5 to -5
                ],
                device=self.device,
            )

            camera = Camera.from_args(
                eye=pos,
                at=torch.zeros(3, device=self.device),
                up=torch.tensor([0.0, 1.0, 0.0], device=self.device),
                fov=60 * np.pi / 180,
                width=256,
                height=256,
                near=1e-2,
                far=100.0,
                device=self.device,
            )
            cameras.append(camera)

        return cameras

    def _transform_vertices(self, vertices, position, rotation, scale):
        """Transform mesh vertices based on position, rotation, and scale"""
        # Make transform matrix from position, rotation (euler angles), and scale
        cos_r = torch.cos(rotation)
        sin_r = torch.sin(rotation)

        # Rotation matrices
        R_x = torch.tensor(
            [[1, 0, 0], [0, cos_r[0], -sin_r[0]], [0, sin_r[0], cos_r[0]]],
            device=self.device,
        )
        R_y = torch.tensor(
            [[cos_r[1], 0, sin_r[1]], [0, 1, 0], [-sin_r[1], 0, cos_r[1]]],
            device=self.device,
        )
        R_z = torch.tensor(
            [[cos_r[2], -sin_r[2], 0], [sin_r[2], cos_r[2], 0], [0, 0, 1]],
            device=self.device,
        )

        # Combine into single transform
        R = torch.matmul(torch.matmul(R_z, R_y), R_x)
        R = R * scale  # Scale the rotation matrix

        # Apply rotation and translation
        return vertices @ R.T + position.unsqueeze(0)

    def _render_depth(self, camera):
        """Render depth map from current scene state"""
        # Start with far plane
        depth_map = torch.full((256, 256), camera.far, device=self.device)

        # Process each object
        for i in range(self.num_objects):
            # 1. Transform vertices to world space
            verts = self._transform_vertices(
                self.vertices,
                self.positions[i],
                self.rotations[i],
                self.scales[i],
            )

            # 2. Transform to camera space
            verts_camera = camera.extrinsics.transform(verts)

            # Skip if behind camera
            if verts_camera[..., 2].max() > 0:
                continue

            # 3. Get face vertices in camera space
            face_vertices_camera = kaolin.ops.mesh.index_vertices_by_faces(
                verts_camera, self.faces
            )

            # 4. Project to screen space
            verts_screen = camera.intrinsics.transform(verts_camera)
            face_vertices_screen = kaolin.ops.mesh.index_vertices_by_faces(
                verts_screen, self.faces
            )

            # 5. Rasterize with z values from camera space
            z_vals = face_vertices_camera[..., 2]
            obj_depth, _ = mesh_render.rasterize(
                256,
                256,
                face_vertices_z=z_vals,
                face_vertices_image=face_vertices_screen[..., :2],
                face_features=z_vals.unsqueeze(-1),
            )

            # Keep closest depth
            depth_map = torch.minimum(depth_map, obj_depth[..., 0])

        return depth_map

    def get_batch(self):
        """
        Generate synthetic data from current scene state.

        Returns:
            dict containing:
                depth_maps: [num_frames, H, W] depth maps
                point_clouds: [num_frames, N, 3] point clouds
        """
        depth_maps = []
        point_clouds = []

        for camera in self.cameras:
            # Get depth map
            depth_map = self._render_depth(camera)
            depth_maps.append(depth_map)

            # Convert to point cloud
            points = depth_to_pointcloud(depth_map, camera)
            point_clouds.append(points)

        return {
            "depth_maps": torch.stack(depth_maps),  # [num_frames, H, W]
            "point_clouds": torch.stack(point_clouds),  # [num_frames, N, 3]
        }

    def _chamfer_loss(self, target_points, predicted_points):
        """
        Compute Chamfer distance from target to predicted points.
        One-directional to handle occlusions - we only care that
        predicted points explain what we see.
        """
        # Compute loss for each frame
        losses = []
        for target, predicted in zip(target_points, predicted_points):
            # Get distances from target points to closest predicted points
            dist_to_pred = kaolin_metrics.sided_distance(target, predicted)[0]
            losses.append(dist_to_pred.mean())

        # Average across frames
        return torch.stack(losses).mean()

    def step(self, target_points):
        """
        Perform one optimization step.

        Args:
            target_points: [num_frames, N, 3] target point clouds

        Returns:
            loss: Scalar loss value
        """
        self.optimizer.zero_grad()

        # Get current prediction
        batch = self.get_batch()
        pred_points = batch["point_clouds"]

        # Compute loss
        loss = self._chamfer_loss(target_points, pred_points)

        # Optimize
        loss.backward()
        self.optimizer.step()

        # Track progress
        self.iteration += 1
        self.loss_history.append(loss.item())

        return loss.item()

    def optimize(self, target_points, num_iterations=1000, callback=None):
        """
        Run optimization for specified iterations.

        Args:
            target_points: [num_frames, N, 3] target point clouds
            num_iterations: Number of optimization steps
            callback: Optional callback(scene, loss, iter) for visualization

        Returns:
            loss_history: List of loss values during optimization
        """
        for i in range(num_iterations):
            loss = self.step(target_points)

            if callback is not None:
                callback(self, loss, i)

            # Optional: Early stopping
            if loss < 1e-6:
                print(f"Converged at iteration {i} with loss {loss:.6f}")
                break

        return self.loss_history
