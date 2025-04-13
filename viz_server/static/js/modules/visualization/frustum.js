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
        // Create direction vector from euler angles (camera looks down -Z)
        const euler = new THREE.Euler(rotation[0], rotation[1], rotation[2], 'YXZ');
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
        far = 0.5,
        lineWidth = 0.005  // Width of the frustum lines
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


    // Create geometry for the frustum outline - proper pyramid shape
    const geometry = new THREE.BufferGeometry();
    const outlineVertices = new Float32Array([
        // Lines from camera position to far corners (pyramid edges)
        ...position.toArray(), ...farTopLeft.toArray(),
        ...position.toArray(), ...farTopRight.toArray(),
        ...position.toArray(), ...farBottomRight.toArray(),
        ...position.toArray(), ...farBottomLeft.toArray(),

        // Far plane rectangle (base of pyramid)
        ...farTopLeft.toArray(), ...farTopRight.toArray(),
        ...farTopRight.toArray(), ...farBottomRight.toArray(),
        ...farBottomRight.toArray(), ...farBottomLeft.toArray(),
        ...farBottomLeft.toArray(), ...farTopLeft.toArray(),
    ]);

    geometry.setAttribute('position', new THREE.BufferAttribute(outlineVertices, 3));

    // Create tube geometry directly from our line segments
    const tubeGeometry = new THREE.BufferGeometry();
    const tubeVertices = [];
    const tubeIndices = [];

    // For each line segment (every 6 values in outlineVertices is a line: start[xyz] end[xyz])
    for (let i = 0; i < outlineVertices.length; i += 6) {
        const start = new THREE.Vector3(
            outlineVertices[i],
            outlineVertices[i + 1],
            outlineVertices[i + 2]
        );
        const end = new THREE.Vector3(
            outlineVertices[i + 3],
            outlineVertices[i + 4],
            outlineVertices[i + 5]
        );
        const direction = end.clone().sub(start).normalize();
        const perpendicular = new THREE.Vector3().crossVectors(direction, new THREE.Vector3(0, 1, 0)).normalize();
        if (perpendicular.lengthSq() === 0) {
            perpendicular.set(1, 0, 0);
        }
        const perpendicular2 = new THREE.Vector3().crossVectors(direction, perpendicular).normalize();

        // Create four vertices around each line point
        const baseIndex = tubeVertices.length / 3;
        for (const point of [start, end]) {
            for (const offset of [
                perpendicular.clone().multiplyScalar(lineWidth),
                perpendicular2.clone().multiplyScalar(lineWidth),
                perpendicular.clone().multiplyScalar(-lineWidth),
                perpendicular2.clone().multiplyScalar(-lineWidth)
            ]) {
                const vertex = point.clone().add(offset);
                tubeVertices.push(vertex.x, vertex.y, vertex.z);
            }
        }

        // Create triangles for the tube segment
        const faces = [
            0, 1, 4, 4, 1, 5,  // top
            1, 2, 5, 5, 2, 6,  // right
            2, 3, 6, 6, 3, 7,  // bottom
            3, 0, 7, 7, 0, 4   // left
        ];
        for (const index of faces) {
            tubeIndices.push(baseIndex + index);
        }
    }

    tubeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(tubeVertices, 3));
    tubeGeometry.setIndex(tubeIndices);
    tubeGeometry.computeVertexNormals();


    // Create material and mesh
    const material = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity,
        side: THREE.DoubleSide
    });

    return new THREE.Mesh(tubeGeometry, material);
}
