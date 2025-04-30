"""
Scene representation for 4D reality learning system - V2 with batched operations.
"""

import torch
import numpy as np
from pathlib import Path
import kaolin.metrics.pointcloud as kaolin_metrics
from kaolin.render.camera import Camera as KaolinCamera
from kaolin.math.quat import quat_unit, rot33_from_quat
from kaolin.render.mesh import rasterize
from einops import rearrange, repeat
from .utils import load_mesh


class SceneV2:
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

        # Initialize cameras
        camera_positions = torch.tensor(
            [
                [1.0, 0.3, 0.0],  # Side view
                [0.0, 0.3, 1.0],  # Side view
            ],
            device=device,
        )
        self.num_cameras = camera_positions.shape[0]

        # Create true cameras (fixed)
        self.true_cameras = []
        for i in range(self.num_cameras):
            camera = KaolinCamera.from_args(
                eye=camera_positions[i],
                at=torch.zeros(3, device=device),  # Look at origin
                up=torch.tensor([0.0, 1.0, 0.0], device=device),
                fov=60 * np.pi / 180,
                width=256,
                height=256,
                near=0.001,
                far=10.0,
                device=device,
            )
            self.true_cameras.append(camera)

        # Create predicted cameras (learnable)
        noise = torch.randn_like(camera_positions) * 0.05
        noisy_positions = camera_positions + noise
        self.pred_cameras = []
        for i in range(self.num_cameras):
            camera = KaolinCamera.from_args(
                eye=noisy_positions[i],
                at=torch.zeros(3, device=device),  # Look at origin
                up=torch.tensor([0.0, 1.0, 0.0], device=device),
                fov=60 * np.pi / 180,
                width=256,
                height=256,
                near=0.001,
                far=10.0,
                device=device,
            )
            camera.requires_grad_(True)  # Enable gradients for all camera parameters
            self.pred_cameras.append(camera)

        # Load mesh once
        self.mesh_verts, self.mesh_faces = load_mesh("3d_models", self.device)

        # Initialize scene states [num_frames, num_objects, dims]
        self.true_positions = torch.zeros(num_frames, num_objects, 3, device=device)
        self.true_rotations = torch.zeros(num_frames, num_objects, 4, device=device)
        self.true_scales = torch.ones(num_frames, num_objects, 1, device=device)

        self.pred_positions = torch.zeros(num_frames, num_objects, 3, device=device)
        self.pred_rotations = torch.zeros(num_frames, num_objects, 4, device=device)
        self.pred_scales = torch.ones(num_frames, num_objects, 1, device=device)

        # Generate motion paths for both true and pred
        for t in range(num_frames):
            time = torch.tensor(t / (num_frames - 1), device=device)
            angle = 2 * torch.pi * time

            # Generate positions
            pos_0 = torch.stack(
                [
                    0.3 * torch.cos(angle),
                    torch.tensor(0.0, device=device),
                    0.3 * torch.sin(angle),
                ]
            )
            pos_1 = torch.stack(
                [
                    0.2 * torch.cos(angle),
                    torch.tensor(0.0, device=device),
                    0.2 * torch.sin(2 * angle),
                ]
            )
            self.true_positions[t, 0] = pos_0
            self.true_positions[t, 1] = pos_1
            self.pred_positions[t, 0] = pos_0
            self.pred_positions[t, 1] = pos_1

            # Generate rotations
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
            q_y = quat_unit(q_y)
            self.true_rotations[t] = q_y
            self.pred_rotations[t] = q_y

        # Add noise to predicted state and enable gradients
        noise = torch.randn_like(self.true_positions) * 0.1
        self.pred_positions = (self.pred_positions + noise).requires_grad_(True)
        self.pred_rotations = self.pred_rotations.clone().detach().requires_grad_(True)
        self.pred_scales = self.pred_scales.clone().detach().requires_grad_(True)

    def compute_energy(self, return_points=False):
        # Stage 1: Stack true/pred states
        # Input:  [T=30 frames, O=2 objects, D=dims (3 for pos, 4 for rot, 1 for scale)]
        # Output: [P=2 states, T=30 frames, O=2 objects, D=dims]
        positions = torch.stack(
            [self.true_positions, self.pred_positions]
        )  # [P T O xyz]
        rotations = torch.stack(
            [self.true_rotations, self.pred_rotations]
        )  # [P T O xyzw]
        scales = torch.stack([self.true_scales, self.pred_scales])  # [P T O scale]

        # Stage 2: Reshape for batched processing
        # Input:  [P=2 states, T=30 frames, O=2 objects, D=dims]
        # Output: [B=60 batches, O=2 objects, D=dims]
        positions = rearrange(positions, "P T O xyz -> (P T) O xyz")
        rotations = rearrange(rotations, "P T O xyzw -> (P T) O xyzw")
        scales = rearrange(scales, "P T O s -> (P T) O s")

        # Stage 3: Transform vertices
        # Input:  verts [V=35947 vertices, xyz=3]
        #         positions [B=60, O=2, xyz=3]
        #         rotations [B=60, O=2, xyzw=4]
        #         scales [B=60, O=2, scale=1]
        # Output: transformed_verts [B=60, V=71894, xyz=3]
        batch_size = 2 * self.num_frames

        # First apply rotation
        rot_mat = rot33_from_quat(rearrange(rotations, "B O xyzw -> (B O) xyzw"))
        rot_mat = rearrange(rot_mat, "(B O) h w -> B O h w", B=batch_size)

        # Expand vertices for each batch and object
        verts = repeat(
            self.mesh_verts, "V xyz -> B O V xyz", B=batch_size, O=self.num_objects
        )
        verts = torch.matmul(verts, rot_mat.transpose(-1, -2))

        # Apply scale then translate
        verts = verts * rearrange(scales, "B O s -> B O 1 s")
        verts = verts + rearrange(positions, "B O xyz -> B O 1 xyz")

        # Merge objects with face offsets
        # Input:  verts [B=60, O=2, V=35947, xyz=3]
        #         faces [F=69451, idx=3]
        # Output: verts [B=60, V=71894, xyz=3]  # All objects' vertices
        #         faces [F=138902, idx=3]       # All objects' faces with offsets

        # Build face offsets for each object
        faces_list = []
        offset = 0
        for i in range(self.num_objects):
            f = self.mesh_faces + offset  # Add offset to face indices
            faces_list.append(f)
            offset += self.mesh_verts.shape[0]  # Increment by vertices per object
        faces = torch.cat(faces_list, dim=0)  # [F=138902, idx=3]
        faces = faces[:, [0, 2, 1]]  # Swap v1,v2 for correct winding

        # Merge object vertices
        verts = rearrange(verts, "B O V xyz -> B (O V) xyz")  # [B=60, V=71894, xyz=3]

        # Stage 4: Camera matrices
        # Input:  true_cameras List[C=2], pred_cameras List[C=2]
        # Output: view_mats [BC=120, h=4, w=4]  # Concatenated true and pred
        #         proj_mats [BC=120, h=4, w=4]  # Concatenated true and pred

        # Stack true and pred cameras separately
        true_view = torch.stack(
            [cam.extrinsics.view_matrix().squeeze(0) for cam in self.true_cameras]
        )
        pred_view = torch.stack(
            [cam.extrinsics.view_matrix().squeeze(0) for cam in self.pred_cameras]
        )
        true_proj = torch.stack(
            [cam.intrinsics.projection_matrix().squeeze(0) for cam in self.true_cameras]
        )
        pred_proj = torch.stack(
            [cam.intrinsics.projection_matrix().squeeze(0) for cam in self.pred_cameras]
        )

        # Repeat each for their respective batches (half of batch_size each)
        true_view = repeat(
            true_view, "C h w -> (B C) h w", B=batch_size // 2
        )  # [60, 4, 4]
        pred_view = repeat(
            pred_view, "C h w -> (B C) h w", B=batch_size // 2
        )  # [60, 4, 4]
        true_proj = repeat(
            true_proj, "C h w -> (B C) h w", B=batch_size // 2
        )  # [60, 4, 4]
        pred_proj = repeat(
            pred_proj, "C h w -> (B C) h w", B=batch_size // 2
        )  # [60, 4, 4]

        # Concatenate true and pred
        view_mats = torch.cat([true_view, pred_view], dim=0)  # [120, 4, 4]
        proj_mats = torch.cat([true_proj, pred_proj], dim=0)  # [120, 4, 4]

        # Stage 5: Project through cameras
        # Input:  verts [B=60, V=71894, xyz=3]
        #         view_mats [BC=120, h=4, w=4]  # Already expanded for batches
        #         proj_mats [BC=120, h=4, w=4]  # Already expanded for batches
        # Output: projected_verts [BC=120 views, V=71894, xyz=3]  # 60 batches * 2 cameras

        # Add homogeneous coordinate and expand for cameras
        verts_h = torch.cat(
            [verts, torch.ones_like(verts[..., :1])], dim=-1
        )  # [B, V, 4]
        verts_h = repeat(
            verts_h, "B V d -> (B C) V d", C=self.num_cameras
        )  # [BC, V, 4]

        # Project to camera and clip space
        verts_cam = torch.bmm(verts_h, view_mats.transpose(1, 2))
        verts_clip = torch.bmm(verts_cam, proj_mats.transpose(1, 2))
        verts_ndc = verts_clip[..., :3] / verts_clip[..., 3:]

        # Stage 6: Rasterize
        # Input:  verts_ndc [BC=120 views, V=71894, xyz=3]  # 60 batches * 2 cameras
        #         faces [F=138902 faces, idx=3]  # All objects' faces with offsets
        # Output: image_xyz [2, B=60, H=256, W=256, xyz=3]  # Split by true/pred
        #         face_idx [2, B=60, H=256, W=256]         # Split by true/pred

        # Get per-face data for all objects
        face_vs = verts_ndc[:, faces]  # [BC, F, 3, 3]  # F includes all objects' faces
        face_xy = face_vs[..., :2]  # [BC, F, 3, 2]
        face_xyz = verts_cam[:, faces, :3]  # [BC, F, 3, 3]
        face_z = face_xyz[..., 2]  # [BC, F, 3]

        # Rasterize to get point clouds
        image_xyz, face_idx = rasterize(
            256, 256, face_z, face_xy, face_xyz, backend="cuda"
        )
        # Reshape to separate true/pred states
        image_xyz = rearrange(
            image_xyz, "(P B) H W xyz -> P B H W xyz", P=2
        )  # [2, 60, 256, 256, 3]
        face_idx = rearrange(face_idx, "(P B) H W -> P B H W", P=2)  # [2, 60, 256, 256]

        # Stage 7: Extract valid points and compute chamfer distance
        # Input:  image_xyz [2, B=60, H=256, W=256, xyz=3]  # Split by true/pred
        #         face_idx [2, B=60, H=256, W=256]         # Split by true/pred
        # Output: energy scalar (mean chamfer distance across valid pairs)
        #         true_points, pred_points lists of valid point clouds if return_points=True

        true_points = []
        pred_points = []
        valid_pairs = []

        # Process each frame and camera combination
        for frame_idx in range(self.num_frames):
            for cam_idx in range(self.num_cameras):
                # Convert frame/camera indices to batch index
                b = frame_idx * self.num_cameras + cam_idx

                # Get valid points for this view
                true_valid = face_idx[0, b] >= 0  # [H, W]
                pred_valid = face_idx[1, b] >= 0  # [H, W]

                true_pc = image_xyz[0, b][true_valid]  # [N, 3]
                pred_pc = image_xyz[1, b][pred_valid]  # [M, 3]

                true_points.append(true_pc)
                pred_points.append(pred_pc)

                if len(true_pc) > 0 and len(pred_pc) > 0:
                    valid_pairs.append((true_pc, pred_pc))

        if not valid_pairs:
            # Return high energy when no valid pairs (encourages keeping objects in view)
            energy = torch.tensor(100.0, device=self.device)
        else:
            chamfer_dists = [
                kaolin_metrics.chamfer_distance(
                    t.unsqueeze(0), p.unsqueeze(0), w1=1.0, w2=1.0, squared=True
                ).mean()
                for t, p in valid_pairs
            ]
            energy = sum(chamfer_dists) / len(chamfer_dists)

        if return_points:
            # Stage 8: Convert to world space for visualization
            # Input:  true_points, pred_points lists of valid point clouds in camera space
            #         cameras List[C=2] of camera objects (true/pred separately)
            # Output: true_points_world, pred_points_world lists of point clouds in world space

            # Get camera transforms
            true_cam2world = torch.stack(
                [
                    cam.extrinsics.inv_view_matrix().squeeze(0)
                    for cam in self.true_cameras
                ]
            )  # [C, 4, 4]
            pred_cam2world = torch.stack(
                [
                    cam.extrinsics.inv_view_matrix().squeeze(0)
                    for cam in self.pred_cameras
                ]
            )  # [C, 4, 4]

            # Process points in corresponding pairs
            true_points_world = []
            pred_points_world = []
            for frame_idx in range(self.num_frames):
                for cam_idx in range(self.num_cameras):
                    idx = frame_idx * self.num_cameras + cam_idx
                    true_pc, pred_pc = true_points[idx], pred_points[idx]

                    if len(true_pc) == 0 or len(pred_pc) == 0:
                        # Skip empty point clouds but maintain list structure
                        true_points_world.append(
                            torch.empty((0, 3), device=self.device)
                        )
                        pred_points_world.append(
                            torch.empty((0, 3), device=self.device)
                        )
                        continue

                    # Add homogeneous coordinates and transform
                    true_h = torch.cat(
                        [true_pc, torch.ones_like(true_pc[:, :1])], dim=1
                    )
                    pred_h = torch.cat(
                        [pred_pc, torch.ones_like(pred_pc[:, :1])], dim=1
                    )

                    true_world = (true_h @ true_cam2world[cam_idx].T)[:, :3]
                    pred_world = (pred_h @ pred_cam2world[cam_idx].T)[:, :3]

                    true_points_world.append(true_world)
                    pred_points_world.append(pred_world)

            return energy, (true_points_world, pred_points_world)
        return energy
