/**
 * Pure functions for camera frustum visualization
 */
import * as THREE from 'three';

/**
 * Add a camera frustum to a Three.js scene
 * @param {THREE.Scene} scene - The scene to add the frustum to
 * @param {Object} cameraData - Camera parameters
 * @param {Object} options - Visualization options
 * @param {number} options.color - Color of the frustum (default: 0x0000ff)
 * @param {number} options.opacity - Opacity of the frustum (default: 0.3)
 * @returns {THREE.LineSegments} The created frustum visualization
 */
export function addFrustumToScene(scene, cameraData, options = {}) {
    const {
        color = 0x0000ff,
        opacity = 0.3
    } = options;

    // Extract camera parameters
    const { position, direction, up, fov, aspect, near, far } = cameraData;

    // Calculate frustum corners
    const halfHeight = Math.tan(fov * 0.5) * far;
    const halfWidth = halfHeight * aspect;

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
        // Near plane
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

    // Create and add to scene
    const frustum = new THREE.LineSegments(geometry, material);
    scene.add(frustum);

    return frustum;
} 