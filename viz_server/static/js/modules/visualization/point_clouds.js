/**
 * Visualization logic for rendering point clouds and time-varying scenes.
 */
import * as THREE from 'three';

/**
 * Convert from camera space (-Z forward) to Three.js space (+Z forward)
 */
export function transformToThreeSpace(points) {
    return points.map(point => [point[0], point[1], -point[2]]);
}

/**
 * Render a generic point cloud in a given scene.
 * @param {Array} points - Array of [x, y, z] points.
 * @param {Object} options - { color, size, opacity, scene }
 */
export function renderPointCloud(scene, points, options = {}) {
    const {
        color = 0xff0000,
        size = 0.02,
        opacity = 0.7
    } = options;

    // Filter out invalid points
    const validPoints = points.filter(point =>
        point.every(coord => typeof coord === "number" && !isNaN(coord) && isFinite(coord))
    );
    if (validPoints.length === 0) {
        console.warn("No valid points to render");
        return;
    }

    // Transform points to Three.js coordinate system
    const worldPoints = transformToThreeSpace(validPoints);

    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array(worldPoints.flat());
    geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));

    const material = new THREE.PointsMaterial({
        color,
        size,
        transparent: true,
        opacity
    });

    const pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);
    return pointCloud;
}

/**
 * Create a point cloud from an array of points
 * @param {Array} points - Array of point data
 * @param {Object} options - Visualization options
 * @param {number} options.color - Color of the points
 * @param {number} options.size - Size of each point
 * @param {number} options.opacity - Opacity of the points
 * @returns {THREE.Points} Three.js points object
 */
export function createPointCloud(points, options = {}) {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(points.length * 3);

    // Fill positions array
    for (let i = 0; i < points.length; i++) {
        positions[i * 3] = points[i].x;
        positions[i * 3 + 1] = points[i].y;
        positions[i * 3 + 2] = points[i].z;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
        color: options.color || 0xff0000,
        size: options.size || 0.1,
        transparent: options.opacity < 1.0,
        opacity: options.opacity || 1.0
    });

    return new THREE.Points(geometry, material);
}


/**
 * Add a point cloud to a Three.js scene
 * @param {THREE.Scene} scene - The scene to add the point cloud to
 * @param {Array} points - Array of [x, y, z] points
 * @param {Object} options - Visualization options
 * @param {number} options.color - Color of the points (default: 0xff0000)
 * @param {number} options.size - Size of each point (default: 0.02)
 * @param {number} options.opacity - Opacity of the points (default: 0.7)
 * @returns {THREE.Points} The created point cloud object
 */
export function addPointCloudToScene(scene, points, options = {}) {
    const {
        color = 0xff0000,
        size = 0.02,
        opacity = 0.7
    } = options;

    // Filter out invalid points
    const validPoints = points.filter(point =>
        point.every(coord => typeof coord === "number" && !isNaN(coord) && isFinite(coord))
    );

    if (validPoints.length === 0) {
        console.warn("No valid points to render");
        return null;
    }

    // Create geometry
    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array(validPoints.flat());
    geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));

    // Create material
    const material = new THREE.PointsMaterial({
        color,
        size,
        transparent: true,
        opacity
    });

    // Create and add to scene
    const pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);

    return pointCloud;
}
