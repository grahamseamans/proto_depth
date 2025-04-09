"""
Scene generation and data loading for 4D reality learning system.
Generates scenes with dragons and camera paths, producing depth maps.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import kaolin as kal
import kaolin.io.obj
import kaolin.ops.mesh
import kaolin.render.mesh as mesh_render
from kaolin.render.camera import Camera

from .state import SceneState


class SceneDataset(Dataset):
    """Generates synthetic scenes with dragons and camera paths"""

    def __init__(
        self,
        num_scenes=1000,
        num_frames=30,
        num_objects=2,
        models_dir="3d_models",
        device=None,
    ):
        """
        Initialize the scene dataset.

        Args:
            num_scenes: Number of different scenes to generate
            num_frames: Number of frames per scene (camera positions)
            num_objects: Number of objects per scene
            models_dir: Directory containing 3D models
            device: Device to use for computations
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.num_scenes = num_scenes
        self.num_frames = num_frames
        self.num_objects = num_objects

        # Load 3D models
        self.models = self._load_models(models_dir)

        # Cache mesh data since it never changes
        self.vertices = self.models["dragon"].vertices.to(
            dtype=torch.float32, device=self.device
        )
        self.faces = self.models["dragon"].faces.to(
            dtype=torch.int64, device=self.device
        )

        # Generate camera paths for all scenes
        self.camera_paths = self._generate_camera_paths()

        # Generate ground truth scene states
        self.scene_states = self._generate_scene_states()

    def _load_models(self, models_dir):
        """Load OBJ models"""
        models = {}
        models_dir = Path(models_dir)

        # Load available models
        model_files = {
            "bunny": "bunny.obj",
            "spot": "spot.obj",
            "armadillo": "armadillo.obj",
        }

        # Try each model, use the first one that loads
        for name, filename in model_files.items():
            path = models_dir / filename
            if path.exists():
                print(f"Loading {name} model...")
                models[name] = kaolin.io.obj.import_mesh(str(path))
                # Just need one model for now
                break

        if not models:
            raise FileNotFoundError(
                f"No models found in {models_dir}. "
                "Run scripts/visualize_dataloader.py first to download models."
            )

        # Use the first model for all objects
        first_model_name = next(iter(models))
        return {
            "dragon": models[first_model_name]  # Keep name for compatibility
        }

    def _generate_camera_paths(self):
        """Generate X-shaped camera paths through scene"""
        paths = []
        for _ in range(self.num_scenes):
            scene_cameras = []
            # Generate points along an X
            t = torch.linspace(-1, 1, self.num_frames // 2, device=self.device)

            # First diagonal of X: (-5,3,-5) to (5,3,5)
            for i in range(self.num_frames // 2):
                pos = torch.tensor(
                    [
                        t[i] * 5,  # x: -5 to 5
                        3.0,  # fixed height
                        t[i] * 5,  # z: -5 to 5
                    ],
                    device=self.device,
                )

                # Look at origin
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
                scene_cameras.append(camera)

            # Second diagonal: (-5,3,5) to (5,3,-5)
            for i in range(self.num_frames // 2):
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
                scene_cameras.append(camera)

            paths.append(scene_cameras)
        return paths

    def _generate_scene_states(self):
        """Generate ground truth scene states"""
        states = []
        for _ in range(self.num_scenes):
            state = SceneState(num_objects=self.num_objects, device=self.device)
            # Objects are initialized with random positions in SceneState
            states.append(state)
        return states

    def transform_vertices(self, vertices, position, rotation, scale):
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
        R = R * scale.unsqueeze(-1)  # Scale the rotation matrix

        # Apply rotation and translation
        return vertices @ R.T + position.unsqueeze(0)

    def render_depth_map(self, scene_state, camera):
        """
        Render depth map from scene state at given camera.

        Args:
            scene_state: SceneState object containing object parameters
            camera: Kaolin Camera object for rendering

        Returns:
            depth_map: [H, W] tensor of depth values
        """
        # Start with far plane
        depth_map = torch.full((256, 256), camera.far, device=self.device)

        # Process each object separately
        for i in range(self.num_objects):
            # 1. Transform vertices to world space
            verts = self.transform_vertices(
                self.vertices,
                scene_state.get_object_positions()[i],
                scene_state.get_object_rotations()[i],
                scene_state.get_object_scales()[i],
            )

            # 2. Transform to camera space first
            verts_camera = camera.extrinsics.transform(verts)

            # Print debug info
            print(f"\nObject {i}:")
            print(f"World pos: {scene_state.get_object_positions()[i]}")
            print(f"Camera pos: {camera.extrinsics.cam_pos().squeeze()}")
            print(
                f"Vertices z range: {verts_camera[..., 2].min():.2f} to {verts_camera[..., 2].max():.2f}"
            )

            # Skip if behind camera (z < 0 in camera space means in front)
            if verts_camera[..., 2].max() > 0:
                print("Some vertices behind camera, skipping")
                continue

            # 3. Get face vertices in camera space
            face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(
                verts_camera, self.faces
            )

            # 4. Project to screen space
            verts_screen = camera.intrinsics.transform(verts_camera)
            face_vertices_screen = kal.ops.mesh.index_vertices_by_faces(
                verts_screen, self.faces
            )

            # 5. Rasterize with z values from camera space
            # Use raw z values - positive is in front of camera
            z_vals = face_vertices_camera[..., 2]
            obj_depth, _ = mesh_render.rasterize(
                256,
                256,
                face_vertices_z=z_vals,
                face_vertices_image=face_vertices_screen[..., :2],
                face_features=z_vals.unsqueeze(-1),
            )

            # Keep closest depth (smallest z value)
            depth_map = torch.minimum(depth_map, obj_depth[..., 0])

        return depth_map

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx):
        """Get a scene with its camera path and depth maps"""
        scene_state = self.scene_states[idx]
        cameras = self.camera_paths[idx]

        # Generate depth maps for each camera
        depth_maps = []
        camera_positions = []
        camera_rotations = []
        for camera in cameras:
            depth_map = self.render_depth_map(scene_state, camera)
            depth_maps.append(depth_map)

            # Extract camera position and rotation for compatibility
            pos = camera.extrinsics.cam_pos().squeeze()  # Remove extra dimensions [3]
            camera_positions.append(pos)
            # Get camera rotation from view matrix
            view_matrix = camera.extrinsics.view_matrix().squeeze(
                0
            )  # Remove batch dim [4, 4]
            rotation_matrix = view_matrix[:3, :3]  # [3, 3]

            # Convert rotation matrix to Euler angles
            # Assuming rotation order is YXZ (yaw, pitch, roll)
            yaw = torch.atan2(rotation_matrix[0, 2], rotation_matrix[2, 2])
            pitch = torch.asin(-rotation_matrix[1, 2])
            roll = torch.atan2(rotation_matrix[1, 0], rotation_matrix[1, 1])
            camera_rotations.append(torch.stack([pitch, yaw, roll]))

        depth_maps = torch.stack(depth_maps).detach()  # Detach for visualization
        camera_positions = torch.stack(camera_positions)
        camera_rotations = torch.stack(camera_rotations)

        return {
            "depth_maps": depth_maps,  # [num_frames, H, W]
            "camera_positions": camera_positions,  # [num_frames, 3]
            "camera_rotations": camera_rotations,  # [num_frames, 3]
            "scene_state": scene_state,  # Ground truth state
        }
