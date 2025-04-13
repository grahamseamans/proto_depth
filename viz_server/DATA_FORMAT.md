# Data Format Specification

This document describes the required JSON structure for each frame in the 4D Reality Learning System pipeline. This format is designed for maximum generality, extensibility, and compatibility with both optimization and visualization.

---

## Top-Level Structure (per frame)

```json
{
  "true": {
    "camera": {
      "transforms": [
        [
          [r11, r12, r13, tx],  // 4x4 camera-to-world transform matrix
          [r21, r22, r23, ty],  // (column-major)
          [r31, r32, r33, tz],
          [0,   0,   0,   1]
        ],
        ...  // N cameras
      ]
    },
    "objects": {
      "positions": [[x, y, z], ...],      // M objects
      "rotations": [[yaw, pitch, roll], ...],
      "scales": [[s], ...]                // or [sx, sy, sz]
    }
  },
  "pred": {
    // Same structure as "true"
  },
  "point_clouds": [
    [[x, y, z], ...],  // K points per camera, in camera-local space
    ...  // N cameras
  ]
}
```

---

## Field Descriptions

- **true / pred**:  
  - **camera.transforms**: Array of N camera-to-world transform matrices (4x4, column-major).
    Each matrix transforms from camera space (-Z forward) to world space.
  - **objects.positions**: Array of M object positions, each `[x, y, z]`.
  - **objects.rotations**: Array of M object rotations, each `[yaw, pitch, roll]`.
  - **objects.scales**: Array of M object scales, each `[s]` or `[sx, sy, sz]`.

- **point_clouds**:  
  - Array of N arrays, one per camera.
  - Each is an array of `[x, y, z]` points, in the local coordinate system of that camera (camera at origin, -Z forward).
  - Points are transformed to world space at runtime using the corresponding camera transform.

---

## Constraints

- All arrays must be non-empty and of matching length within their section.
- All numbers must be finite (not NaN or inf).
- The number of cameras (N) must match between true and pred.
- The number of objects (M) must match between true and pred.
- The number of point clouds (N) must match the number of cameras.
- All transform matrices must be 4x4 and column-major.

---

## Example

```json
{
  "true": {
    "camera": {
      "transforms": [
        [
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]
        ],
        [
          [0, 0, -1, 1],
          [0, 1, 0, 0],
          [1, 0, 0, 0],
          [0, 0, 0, 1]
        ]
      ]
    },
    "objects": {
      "positions": [[0.5, 0, 0], [0.2, 0, 0]],
      "rotations": [[0, 0, 0], [0, 0, 0]],
      "scales": [[1], [1]]
    }
  },
  "pred": {
    "camera": {
      "transforms": [
        [
          [1, 0, 0, 0.1],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]
        ],
        [
          [0, 0, -1, 1.1],
          [0, 1, 0, 0],
          [1, 0, 0, 0],
          [0, 0, 0, 1]
        ]
      ]
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

- All point clouds are stored in camera-local space. Transformations to world space are performed at runtime using the corresponding camera transform matrix.
- Camera transforms are camera-to-world (inverse view matrix), so points can be transformed directly.
- All matrices are column-major (standard in computer graphics).
- This format is extensible for additional object types, attributes, or future requirements.
