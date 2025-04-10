"""
Energy-based scene optimization using Chamfer distance.
Direct optimization of scene parameters without neural networks.
"""

import torch
import torch.optim as optim
import kaolin.metrics.pointcloud as kaolin_metrics

from .state import SceneState
from .point_cloud import depth_to_pointcloud


class EnergyOptimizer:
    """Optimizes scene state directly through energy minimization"""

    def __init__(self, scene_state: SceneState, vertices, faces, learning_rate=0.01):
        """
        Initialize the energy optimizer.

        Args:
            scene_state: SceneState object containing parameters to optimize
            learning_rate: Learning rate for optimization
        """
        self.scene_state = scene_state
        self.device = scene_state.device

        # Cache mesh data
        self.vertices = vertices.to(device=self.device)
        self.faces = faces.to(device=self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(scene_state.parameters(), lr=learning_rate)

        # For tracking progress
        self.iteration = 0
        self.loss_history = []

    def compute_energy(self, point_cloud, predicted_points):
        """
        Compute one-directional Chamfer distance from observed to predicted points.
        For each observed point, find distance to closest predicted point.
        This ensures the state explains what we see, without forcing matches to
        invisible parts of objects.

        Args:
            point_cloud: Observed point cloud from depth map [N, 3]
            predicted_points: Predicted point cloud from current state [M, 3]

        Returns:
            energy: Scalar tensor representing the energy to minimize
        """
        # Get distances from observed points to their closest predicted points
        dist_to_state = kaolin_metrics.sided_distance(point_cloud, predicted_points)[0]

        # Mean distance - make state explain what we observe
        energy = dist_to_state.mean()

        return energy

    def step(self, point_cloud):
        """
        Perform one optimization step.

        Args:
            point_cloud: Observed point cloud tensor [N, 3]

        Returns:
            energy: The energy value after this step
        """
        # Reset gradients
        self.optimizer.zero_grad()

        # Get predicted point cloud from current state
        # This will be implemented in the scene generator/renderer
        predicted_points = self.get_predicted_points()

        # Compute energy
        energy = self.compute_energy(point_cloud, predicted_points)

        # Backpropagate
        energy.backward()

        # Update parameters
        self.optimizer.step()

        # Track progress
        self.iteration += 1
        self.loss_history.append(energy.item())

        return energy.item()

    def get_predicted_points(self):
        """
        Generate predicted point cloud from current scene state using Kaolin mesh rendering.
        Uses the same rendering pipeline as the dataloader to ensure consistency.
        """
        # Import Kaolin rendering components
        import kaolin.ops.mesh
        import kaolin.render.mesh as mesh_render
        from kaolin.render.camera import Camera

        # Get current camera
        camera = self.scene_state.camera

        # Start with far plane
        depth_map = torch.full((256, 256), camera.far, device=self.device)

        # Process each object
        for i in range(self.scene_state.num_objects):
            # 1. Transform vertices to world space
            verts = self.transform_vertices(
                self.vertices,
                self.scene_state.get_object_positions()[i],
                self.scene_state.get_object_rotations()[i],
                self.scene_state.get_object_scales()[i],
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

        # Convert depth map to point cloud using imported function
        points = depth_to_pointcloud(depth_map, camera)
        return points

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

    def optimize(self, point_cloud, num_iterations=1000, callback=None):
        """
        Run optimization for specified number of iterations.

        Args:
            point_cloud: Observed point cloud tensor [N, 3]
            num_iterations: Number of optimization steps to perform
            callback: Optional callback function for visualization

        Returns:
            loss_history: List of energy values during optimization
        """
        for i in range(num_iterations):
            energy = self.step(point_cloud)

            if callback is not None:
                callback(self.scene_state, energy, i)

            # Optional: Early stopping
            if energy < 1e-6:
                break

        return self.loss_history
