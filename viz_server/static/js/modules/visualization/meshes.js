/**
 * Pure functions for mesh visualization
 */
import * as THREE from 'three';

/**
 * Load and parse an OBJ file
 * @param {string} path - Path to the OBJ file
 * @returns {Promise<Object>} Object with vertices and faces arrays
 */
export async function loadObjFile(path) {
    const response = await fetch(path);
    const text = await response.text();

    const vertices = [];
    const faces = [];

    // Parse OBJ file
    const lines = text.split('\n');
    for (const line of lines) {
        const parts = line.trim().split(/\s+/);
        if (parts[0] === 'v') {
            // Vertex
            vertices.push([
                parseFloat(parts[1]),
                parseFloat(parts[2]),
                parseFloat(parts[3])
            ]);
        } else if (parts[0] === 'f') {
            // Face (OBJ indices are 1-based)
            faces.push([
                parseInt(parts[1].split('/')[0]) - 1,
                parseInt(parts[2].split('/')[0]) - 1,
                parseInt(parts[3].split('/')[0]) - 1
            ]);
        }
    }

    return { vertices, faces };
}

/**
 * Create a mesh from vertex and face data
 * @param {Object} data - Mesh geometry data
 * @param {Array} data.vertices - Array of vertex positions
 * @param {Array} data.faces - Array of face indices
 * @param {Object} options - Visualization options
 * @param {number} options.color - Color of the mesh
 * @param {boolean} options.wireframe - Whether to show as wireframe
 * @param {number} options.opacity - Opacity of the mesh
 * @returns {THREE.Mesh} The created mesh
 */
export function createMesh(data, options = {}) {
    const {
        color = 0x00ff00,
        wireframe = false,
        opacity = 0.7
    } = options;

    // Create geometry
    const geometry = new THREE.BufferGeometry();

    // Set up vertices
    const positions = new Float32Array(data.vertices.flat());
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // Set up faces if available
    if (data.faces) {
        const indices = new Uint32Array(data.faces.flat());
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    }

    // Compute normals if needed
    if (!data.normals) {
        geometry.computeVertexNormals();
    } else {
        const normals = new Float32Array(data.normals.flat());
        geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    }

    // Create material
    const material = new THREE.MeshPhongMaterial({
        color,
        wireframe,
        transparent: opacity < 1.0,
        opacity,
        side: THREE.DoubleSide
    });

    // Create and return mesh
    return new THREE.Mesh(geometry, material);
}

/**
 * Add a mesh to a Three.js scene
 * @param {THREE.Scene} scene - The scene to add the mesh to
 * @param {Object} data - Mesh geometry data
 * @param {Object} options - Visualization options
 * @returns {THREE.Mesh} The created mesh
 */
export function addMeshToScene(scene, data, options = {}) {
    const mesh = createMesh(data, options);
    scene.add(mesh);
    return mesh;
}

/**
 * Create meshes for a collection of objects
 * @param {Array} objects - Array of objects with mesh data and transforms
 * @param {Object} options - Visualization options
 * @returns {Array<THREE.Mesh>} Array of created meshes
 */
export function createMeshes(objects, options = {}) {
    return objects.map((obj, index) => {
        const mesh = createMesh(obj.data, {
            color: obj.color !== undefined ? obj.color : (obj.isTrue ? 0x00ff00 : 0xff0000),
            opacity: options.opacity || 0.7,
            wireframe: options.wireframe || false
        });

        // Apply transforms
        if (obj.position) mesh.position.fromArray(obj.position);
        if (obj.rotation) {
            if (obj.rotation.length === 4) {
                // Assume quaternion [x, y, z, w]
                mesh.quaternion.fromArray(obj.rotation);
            } else {
                // Fallback: assume Euler angles [x, y, z]
                mesh.rotation.fromArray(obj.rotation);
            }
        }
        if (obj.scale) {
            let scale = obj.scale;
            if (Array.isArray(scale)) {
                if (scale.length === 1) {
                    scale = [scale[0], scale[0], scale[0]];
                } else {
                    throw new Error('Mesh scale must be a single value or array of length 1');
                }
            } else {
                scale = [scale, scale, scale];
            }
            mesh.scale.fromArray(scale);
        }

        return mesh;
    });
}
