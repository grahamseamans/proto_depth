/**
 * Camera frustum visualization
 */
import * as THREE from 'three';

/**
 * Create a camera frustum visualization
 * @param {Array|THREE.Vector3} position - Position of the camera
 * @param {Array|THREE.Vector3} rotation - Rotation of the camera (euler angles)
 * @param {Object} options - Visualization options
 * @param {number} options.color - Color of the frustum (default: 0x0000ff)
 * @param {number} options.opacity - Opacity of the frustum (default: 0.3)
 * @param {number} options.fov - Field of view in degrees (default: 60)
 * @param {number} options.aspect - Aspect ratio (default: 1.0)
 * @param {number} options.near - Near plane distance (default: 0.1)
 * @param {number} options.far - Far plane distance (default: 1.0)
 * @returns {THREE.LineSegments} The created frustum visualization
 */
export function createFrustum(position, rotation, options = {}) {
    // Convert position array to Vector3 if needed
    if (Array.isArray(position)) {
        position = new THREE.Vector3(...position);
    }

    // Convert rotation array to direction vector
    let direction;
    if (Array.isArray(rotation)) {
        // Create direction vector from euler angles
        const euler = new THREE.Euler(rotation[0], rotation[1], rotation[2]);
        direction = new THREE.Vector3(0, 0, -1).applyEuler(euler);
    } else {
        // Assume rotation is already a direction vector
        direction = rotation;
    }
    const {
        color = 0x0000ff,
        opacity = 0.3,
        fov = 60,
        aspect = 1.0,
        near = 0.1,
        far = 1.0
    } = options;

    // Calculate frustum corners
    const halfHeight = Math.tan(fov * Math.PI / 360) * far;
    const halfWidth = halfHeight * aspect;

    // Create up vector (assuming Y-up)
    const up = new THREE.Vector3(0, 1, 0);

    // Calculate frustum corners
    const farCenter = new THREE.Vector3().copy(position).add(
        new THREE.Vector3().copy(direction).multiplyScalar(far)
    );

    const right = new THREE.Vector3().crossVectors(direction, up).normalize();
    const upVec = new THREE.Vector3().crossVectors(right, direction).normalize();

    const farTopLeft = new THREE.Vector3()
        .copy(farCenter)
        .add(right.clone().multiplyScalar(-halfWidth))
        .add(upVec.clone().multiplyScalar(halfHeight));

    const farTopRight = new THREE.Vector3()
        .copy(farCenter)
        .add(right.clone().multiplyScalar(halfWidth))
        .add(upVec.clone().multiplyScalar(halfHeight));

    const farBottomLeft = new THREE.Vector3()
        .copy(farCenter)
        .add(right.clone().multiplyScalar(-halfWidth))
        .add(upVec.clone().multiplyScalar(-halfHeight));

    const farBottomRight = new THREE.Vector3()
        .copy(farCenter)
        .add(right.clone().multiplyScalar(halfWidth))
        .add(upVec.clone().multiplyScalar(-halfHeight));

    // Create geometry
    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array([
        // Near plane (just show the position point)
        ...position.toArray(),
        ...position.toArray(),
        ...position.toArray(),
        ...position.toArray(),

        // Far plane
        ...farTopLeft.toArray(),
        ...farTopRight.toArray(),
        ...farBottomRight.toArray(),
        ...farBottomLeft.toArray(),

        // Connections
        ...position.toArray(), ...farTopLeft.toArray(),
        ...position.toArray(), ...farTopRight.toArray(),
        ...position.toArray(), ...farBottomRight.toArray(),
        ...position.toArray(), ...farBottomLeft.toArray(),

        // Far plane connections
        ...farTopLeft.toArray(), ...farTopRight.toArray(),
        ...farTopRight.toArray(), ...farBottomRight.toArray(),
        ...farBottomRight.toArray(), ...farBottomLeft.toArray(),
        ...farBottomLeft.toArray(), ...farTopLeft.toArray()
    ]);

    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

    // Create material
    const material = new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity
    });

    return new THREE.LineSegments(geometry, material);
}
