# Data Format Specification

This document describes the required JSON structure for each frame in the 4D Reality Learning System pipeline. This format is designed for maximum generality, extensibility, and compatibility with both optimization and visualization.

---

## Top-Level Structure (per frame)

```json
{
  "true": {
    "camera": {
      "positions": [[x, y, z], ...],      // N cameras
      "rotations": [[yaw, pitch, roll], ...]
    },
    "objects": {
      "positions": [[x, y, z], ...],      // M objects
      "rotations": [[yaw, pitch, roll], ...],
      "scales": [[s], ...]                // or [sx, sy, sz]
    }
  },
  "pred": {
    "camera": {
      "positions": [[x, y, z], ...],
      "rotations": [[yaw, pitch, roll], ...]
    },
    "objects": {
      "positions": [[x, y, z], ...],
      "rotations": [[yaw, pitch, roll], ...],
      "scales": [[s], ...]
    }
  },
  "point_clouds": [
    [[x, y, z], ...],   // K points per camera, in camera-local space
    ...
  ]
}
```

---

## Field Descriptions

- **true / pred**:  
  - **camera.positions**: Array of N camera positions, each `[x, y, z]`.
  - **camera.rotations**: Array of N camera rotations, each `[yaw, pitch, roll]`.
  - **objects.positions**: Array of M object positions, each `[x, y, z]`.
  - **objects.rotations**: Array of M object rotations, each `[yaw, pitch, roll]`.
  - **objects.scales**: Array of M object scales, each `[s]` or `[sx, sy, sz]`.

- **point_clouds**:  
  - Array of N arrays, one per camera.
  - Each is an array of `[x, y, z]` points, in the local coordinate system of that camera (camera at origin, -Z forward).

---

## Constraints

- All arrays must be non-empty and of matching length within their section.
- All numbers must be finite (not NaN or inf).
- The number of cameras (N) must match between true and pred.
- The number of objects (M) must match between true and pred.
- The number of point clouds (N) must match the number of cameras.

---

## Example

```json
{
  "true": {
    "camera": {
      "positions": [[0, 0, 0], [1, 0, 0]],
      "rotations": [[0, 0, 0], [0, 0, 0]]
    },
    "objects": {
      "positions": [[0.5, 0, 0], [0.2, 0, 0]],
      "rotations": [[0, 0, 0], [0, 0, 0]],
      "scales": [[1], [1]]
    }
  },
  "pred": {
    "camera": {
      "positions": [[0.1, 0, 0], [1.1, 0, 0]],
      "rotations": [[0, 0, 0], [0, 0, 0]]
    },
    "objects": {
      "positions": [[0.6, 0, 0], [0.3, 0, 0]],
      "rotations": [[0, 0, 0], [0, 0, 0]],
      "scales": [[1], [1]]
    }
  },
  "point_clouds": [
    [[0.1, 0.2, -1.0], [0.2, 0.1, -1.1]],
    [[-0.1, 0.0, -1.2], [0.0, -0.1, -1.3]]
  ]
}
```

---

## Notes

- All point clouds are stored in camera-local space. Transformations to world or predicted space are performed at runtime using the corresponding camera parameters.
- This format is extensible for additional object types, attributes, or future requirements.
