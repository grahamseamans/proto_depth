/**
 * Camera frustum visualization
 */
import * as THREE from 'three';

/**
 * Create a camera frustum visualization
 * @param {THREE.Matrix4} transform - Camera-to-world transform matrix
 * @param {Object} options - Visualization options
 * @param {number} options.color - Color of the frustum (default: 0x0000ff)
 * @param {number} options.opacity - Opacity of the frustum (default: 0.3)
 * @param {number} options.fov - Field of view in degrees (default: 60)
 * @param {number} options.aspect - Aspect ratio (default: 1.0)
 * @param {number} options.near - Near plane distance (default: 0.1)
 * @param {number} options.far - Far plane distance (default: 1.0)
 * @param {boolean} options.showLookAt - Whether to show look-at line (default: true)
 * @returns {THREE.Group} Group containing frustum and look-at line
 */
export function createFrustum(transform, options = {}) {
    const {
        color = 0x0000ff,
        opacity = 0.3,
        fov = 60,
        aspect = 1.0,
        near = 0.1,
        far = 0.3,  // Increased far plane to match scene scale
        lineWidth = 0.005,  // Width of the frustum lines
        showLookAt = true
    } = options;

    // Create group to hold all visualization elements
    const group = new THREE.Group();

    // Calculate frustum corners in camera space
    const halfHeight = Math.tan(fov * Math.PI / 360) * far;
    const halfWidth = halfHeight * aspect;

    // Create frustum corners in camera space (origin at 0,0,0, looking down -Z)
    const farTopLeft = new THREE.Vector3(-halfWidth, halfHeight, -far);
    const farTopRight = new THREE.Vector3(halfWidth, halfHeight, -far);
    const farBottomLeft = new THREE.Vector3(-halfWidth, -halfHeight, -far);
    const farBottomRight = new THREE.Vector3(halfWidth, -halfHeight, -far);
    const origin = new THREE.Vector3(0, 0, 0);

    // Apply camera transform to all points
    [farTopLeft, farTopRight, farBottomLeft, farBottomRight].forEach(point => {
        point.applyMatrix4(transform);
    });

    // Get camera position from transform matrix
    const position = new THREE.Vector3().setFromMatrixPosition(transform);

    // Create geometry for the frustum outline
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

    // Create tube geometry for frustum lines
    const tubeGeometry = new THREE.BufferGeometry();
    const tubeVertices = [];
    const tubeIndices = [];

    // For each line segment
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
        const segDir = end.clone().sub(start).normalize();
        const perpendicular = new THREE.Vector3().crossVectors(segDir, new THREE.Vector3(0, 1, 0)).normalize();
        if (perpendicular.lengthSq() === 0) {
            perpendicular.set(1, 0, 0);
        }
        const perpendicular2 = new THREE.Vector3().crossVectors(segDir, perpendicular).normalize();

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

    // Create material and mesh for frustum
    const material = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity,
        side: THREE.DoubleSide
    });

    const frustumMesh = new THREE.Mesh(tubeGeometry, material);
    group.add(frustumMesh);

    // Add look-at line if enabled
    if (showLookAt) {
        const lineGeometry = new THREE.BufferGeometry();
        // Create look-at point by transforming a point 1 unit down -Z in camera space
        const lookAt = new THREE.Vector3(0, 0, -1).applyMatrix4(transform);
        const lineVertices = new Float32Array([
            ...position.toArray(),
            ...lookAt.toArray()
        ]);
        lineGeometry.setAttribute('position', new THREE.BufferAttribute(lineVertices, 3));

        const lineMaterial = new THREE.LineBasicMaterial({
            color,
            transparent: true,
            opacity: opacity * 0.7,  // Slightly more transparent
            linewidth: 1
        });

        const lookAtLine = new THREE.Line(lineGeometry, lineMaterial);
        group.add(lookAtLine);

        // Add small sphere at look-at point
        const sphereGeometry = new THREE.SphereGeometry(lineWidth * 2, 8, 8);
        const sphereMaterial = new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: opacity * 0.7
        });
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        sphere.position.copy(lookAt);
        group.add(sphere);
    }

    return group;
}
