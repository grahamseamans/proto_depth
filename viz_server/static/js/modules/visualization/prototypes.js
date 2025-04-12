/**
 * Pure functions for prototype visualization
 */
import * as THREE from 'three';

/**
 * Add a prototype mesh to a Three.js scene
 * @param {THREE.Scene} scene - The scene to add the prototype to
 * @param {Object} prototypeData - Prototype geometry data
 * @param {Object} options - Visualization options
 * @param {number} options.color - Color of the prototype (default: 0x00ff00)
 * @param {boolean} options.wireframe - Whether to show as wireframe (default: false)
 * @param {number} options.opacity - Opacity of the prototype (default: 1.0)
 * @returns {THREE.Mesh} The created prototype mesh
 */
export function addPrototypeToScene(scene, prototypeData, options = {}) {
    const {
        color = 0x00ff00,
        wireframe = false,
        opacity = 1.0
    } = options;

    // Create geometry from prototype data
    const geometry = new THREE.BufferGeometry();

    // Set up vertices
    const vertices = new Float32Array(prototypeData.vertices.flat());
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

    // Set up faces if available
    if (prototypeData.faces) {
        const indices = new Uint32Array(prototypeData.faces.flat());
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    }

    // Compute normals if needed
    if (!prototypeData.normals) {
        geometry.computeVertexNormals();
    } else {
        const normals = new Float32Array(prototypeData.normals.flat());
        geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    }

    // Create material
    const material = new THREE.MeshPhongMaterial({
        color,
        wireframe,
        transparent: opacity < 1.0,
        opacity
    });

    // Create and add to scene
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    return mesh;
} 