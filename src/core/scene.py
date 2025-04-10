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

            # Keep closest depth (remove any extra dimensions)
            depth_map = torch.minimum(depth_map, obj_depth[..., 0])

        # Debug shape
        print(f"render_depth output shape: {depth_map.shape}")
        return depth_map.squeeze()  # Ensure [H, W]

    def get_scene_points(self):
        """
        Get point clouds from current scene state.
        Each point cloud is in its camera's local space.
        """
        point_clouds = []
        for camera in self.cameras:
            # Get depth map
            depth_map = self._render_depth(camera)

            # Debug depth map shape
            print(f"depth_map shape before pointcloud: {depth_map.shape}")

            # Convert to point cloud (in camera space)
            points = depth_to_pointcloud(depth_map, camera)
            print(f"points shape after pointcloud: {points.shape}")

            point_clouds.append(points)

        return point_clouds

    def _transform_to_world(self, points, camera_idx):
        """Transform points from camera space to world space using state's camera position"""
        camera = self.cameras[camera_idx]
        # Ensure points are [B, N, 3]
        if points.ndim == 2:
            points = points.unsqueeze(0)
        return camera.extrinsics.transform(points)  # Maintains batch dimension

    def _chamfer_loss(self, target_points):
        """
        Compute Chamfer distance between:
        1. Target point clouds transformed by state's camera positions
        2. State's belief of where objects are

        Args:
            target_points: List of point clouds, each in its camera's local space
        """
        total_loss = 0

        # For each camera view
        for i, target in enumerate(target_points):
            # Transform target points using state's camera position (keep gradients)
            target_world = self._transform_to_world(target, i)  # [1, N, 3]

            # Get current scene prediction from this camera
            pred_depth = self._render_depth(self.cameras[i])
            print(f"pred_depth range: {pred_depth.min():.2f} to {pred_depth.max():.2f}")
            print(f"pred_depth shape: {pred_depth.shape}")
            print(f"num valid depths: {(pred_depth < self.cameras[i].far).sum()}")

            pred_points = depth_to_pointcloud(pred_depth, self.cameras[i])  # [1, M, 3]
            print(f"target_world shape: {target_world.shape}")
            print(f"pred_points shape: {pred_points.shape}")

            # Ensure both have batch dimension
            if target_world.ndim == 2:
                target_world = target_world.unsqueeze(0)
            if pred_points.ndim == 2:
                pred_points = pred_points.unsqueeze(0)

            print(
                f"After unsqueeze - target: {target_world.shape}, pred: {pred_points.shape}"
            )

            # Make contiguous for CUDA ops
            target_world = target_world.contiguous()
            pred_points = pred_points.contiguous()

            # Compute Chamfer distance
            dist_to_pred = kaolin_metrics.sided_distance(target_world, pred_points)[0]
            total_loss += dist_to_pred.mean()

        return total_loss / len(target_points)

    def step(self, target_points):
        """
        Perform one optimization step.

        Args:
            target_points: List of point clouds, each in its camera's local space

        Returns:
            loss: Scalar loss value
        """
        self.optimizer.zero_grad()

        # Compute loss between transformed target and state
        loss = self._chamfer_loss(target_points)

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
            target_points: List of point clouds, each in its camera's local space
            num_iterations: Number of optimization steps
            callback: Optional callback(scene, loss, iter) for visualization

        Returns:
            loss_history: List of loss values during optimization
        """
        print(f"Optimizing scene for {num_iterations} iterations...")

        for i in range(num_iterations):
            loss = self.step(target_points)

            if i % 100 == 0:
                print(f"Iteration {i}: loss = {loss:.6f}")

            if callback is not None:
                callback(self, loss, i)

            # Optional: Early stopping
            if loss < 1e-6:
                print(f"Converged at iteration {i} with loss {loss:.6f}")
                break

        return self.loss_history
