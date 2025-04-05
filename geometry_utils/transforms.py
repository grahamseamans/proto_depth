"""
Transformation utilities implemented in pure PyTorch

This module provides functions for 3D geometric transformations
without any dependency on PyTorch3D.
"""

import torch
import math


def euler_angles_to_matrix(angles, convention="XYZ"):
    """
    Convert Euler angles to rotation matrix using the specified convention.

    Args:
        angles: Tensor of shape [..., 3] with angles in radians
        convention: String specifying rotation order, e.g., "XYZ"

    Returns:
        Rotation matrices of shape [..., 3, 3]
    """
    orig_shape = angles.shape[:-1]
    angles = angles.reshape(-1, 3)

    # Handle batch dimensions
    batch_size = angles.shape[0]
    matrices = []

    for i in range(batch_size):
        # Extract angles for this batch
        a, b, c = angles[i]

        # Get individual rotation matrices for each axis
        rot_matrices = {
            "X": rotation_matrix_x(a),
            "Y": rotation_matrix_y(b),
            "Z": rotation_matrix_z(c),
        }

        # Compose according to convention
        R = torch.eye(3, device=angles.device, dtype=angles.dtype)
        for axis in convention:
            R = torch.matmul(rot_matrices[axis], R)

        matrices.append(R)

    # Stack and reshape to match input
    matrices = torch.stack(matrices)
    return matrices.reshape(*orig_shape, 3, 3)


def rotation_matrix_x(angle):
    """
    Create rotation matrix for rotation around X axis.

    Args:
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    R = torch.eye(3, device=angle.device, dtype=angle.dtype)
    R[1, 1] = cos
    R[1, 2] = -sin
    R[2, 1] = sin
    R[2, 2] = cos

    return R


def rotation_matrix_y(angle):
    """
    Create rotation matrix for rotation around Y axis.

    Args:
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    R = torch.eye(3, device=angle.device, dtype=angle.dtype)
    R[0, 0] = cos
    R[0, 2] = sin
    R[2, 0] = -sin
    R[2, 2] = cos

    return R


def rotation_matrix_z(angle):
    """
    Create rotation matrix for rotation around Z axis.

    Args:
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    R = torch.eye(3, device=angle.device, dtype=angle.dtype)
    R[0, 0] = cos
    R[0, 1] = -sin
    R[1, 0] = sin
    R[1, 1] = cos

    return R


def matrix_to_euler_angles(matrix, convention="XYZ"):
    """
    Convert rotation matrix to Euler angles.

    Args:
        matrix: Tensor of shape [..., 3, 3]
        convention: String specifying rotation order, e.g., "XYZ"

    Returns:
        Euler angles of shape [..., 3]
    """
    orig_shape = matrix.shape[:-2]
    matrix = matrix.reshape(-1, 3, 3)

    batch_size = matrix.shape[0]
    angles = []

    for i in range(batch_size):
        R = matrix[i]

        if convention == "XYZ":
            # Extract Euler angles (XYZ convention)
            # From https://www.geometrictools.com/Documentation/EulerAngles.pdf
            if R[2, 0] < 1.0 - 1e-10:
                if R[2, 0] > -1.0 + 1e-10:
                    # Unique solution
                    y = torch.asin(-R[2, 0])
                    x = torch.atan2(R[2, 1], R[2, 2])
                    z = torch.atan2(R[1, 0], R[0, 0])
                else:
                    # R[2, 0] = -1, gimbal lock
                    y = torch.tensor(math.pi / 2, device=R.device, dtype=R.dtype)
                    x = -torch.atan2(-R[0, 1], R[0, 2])
                    z = torch.tensor(0.0, device=R.device, dtype=R.dtype)
            else:
                # R[2, 0] = 1, gimbal lock
                y = torch.tensor(-math.pi / 2, device=R.device, dtype=R.dtype)
                x = torch.atan2(-R[0, 1], R[0, 2])
                z = torch.tensor(0.0, device=R.device, dtype=R.dtype)
        else:
            # For other conventions, you can implement similar equations
            # Here we just return zeros as placeholder
            x = y = z = torch.tensor(0.0, device=R.device, dtype=R.dtype)

        angles.append(torch.stack([x, y, z]))

    # Stack and reshape to match input
    angles = torch.stack(angles)
    return angles.reshape(*orig_shape, 3)


def transform_points(points, matrix, translate=None, scale=None):
    """
    Transform points using rotation matrix, translation vector, and scale.

    Args:
        points: Tensor of shape [..., N, 3]
        matrix: Rotation matrix of shape [..., 3, 3]
        translate: Optional translation vector of shape [..., 3]
        scale: Optional scale factor of shape [..., 1] or [..., 3]

    Returns:
        Transformed points of shape [..., N, 3]
    """
    # Apply rotation
    points_t = torch.matmul(points, matrix.transpose(-1, -2))

    # Apply scaling if provided
    if scale is not None:
        points_t = points_t * scale.unsqueeze(-2)

    # Apply translation if provided
    if translate is not None:
        points_t = points_t + translate.unsqueeze(-2)

    return points_t
