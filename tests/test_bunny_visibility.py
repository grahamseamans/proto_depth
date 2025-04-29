import torch
from src.scene import Scene


def test_bunny_always_visible():
    scene = Scene(num_objects=2, num_frames=30, device="cpu")
    bunny_visible_all = True
    for frame in range(scene.num_frames):
        for cam_idx, camera in enumerate(scene.true_cameras):
            # Get bunny position for this frame
            bunny_pos = scene.true_positions[frame, 0]
            # Camera parameters
            cam_eye = scene.camera_positions[cam_idx]
            cam_at = torch.zeros(3)
            # Vector from camera to bunny
            to_bunny = bunny_pos - cam_eye
            to_look = cam_at - cam_eye
            # Angle between camera look direction and bunny
            cos_angle = torch.dot(to_bunny, to_look) / (
                to_bunny.norm() * to_look.norm()
            )
            angle_deg = torch.acos(cos_angle).item() * 180.0 / 3.14159265
            # Camera FOV is 60 deg, so half-FOV is 30 deg
            if angle_deg > 30.0:
                print(
                    f"Frame {frame}, Camera {cam_idx}: Bunny outside FOV! Angle: {angle_deg:.2f}"
                )
                bunny_visible_all = False
            else:
                print(
                    f"Frame {frame}, Camera {cam_idx}: Bunny inside FOV. Angle: {angle_deg:.2f}"
                )
    assert bunny_visible_all, "Bunny is not visible in all frames for all cameras!"


if __name__ == "__main__":
    test_bunny_always_visible()
