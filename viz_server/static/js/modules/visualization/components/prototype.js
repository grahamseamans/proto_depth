import { objects, elements } from "../../state.js";
import { addMeshToScene, removeMeshFromScene } from "../core/scene.js";
import { positionCamera } from "../core/camera.js";

/**
 * Prototype visualization functionality
 */

/**
 * Create prototype geometry
 * @param {Object} data - Prototype data
 * @returns {THREE.BufferGeometry} Prototype geometry
 */
export function createPrototypeGeometry(data) {
    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array(data.vertices.flat());
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

    if (data.normals) {
        const normals = new Float32Array(data.normals.flat());
        geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    }

    if (data.faces) {
        const indices = new Uint32Array(data.faces.flat());
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    }

    geometry.computeVertexNormals();
    return geometry;
}

/**
 * Create prototype material
 * @param {Object} options - Material options
 * @param {number} options.color - Material color in hex
 * @param {boolean} options.transparent - Whether material is transparent
 * @param {number} options.opacity - Material opacity
 * @param {boolean} options.wireframe - Whether to show wireframe
 * @returns {THREE.Material} Prototype material
 */
export function createPrototypeMaterial(options = {}) {
    const {
        color = 0xffffff,
        transparent = false,
        opacity = 1,
        wireframe = false
    } = options;

    return new THREE.MeshPhongMaterial({
        color,
        transparent,
        opacity,
        wireframe,
        side: THREE.DoubleSide
    });
}

/**
 * Create prototype mesh
 * @param {Object} data - Prototype data
 * @param {Object} materialOptions - Material options
 * @returns {THREE.Mesh} Prototype mesh
 */
export function createPrototype(data, materialOptions = {}) {
    const geometry = createPrototypeGeometry(data);
    const material = createPrototypeMaterial(materialOptions);
    return new THREE.Mesh(geometry, material);
}

/**
 * Add prototype to scene
 * @param {string} sceneKey - Key of the scene to add prototype to
 * @param {Object} data - Prototype data
 * @param {Object} materialOptions - Material options
 * @returns {THREE.Mesh} Created prototype
 */
export function addPrototypeToScene(sceneKey, data, materialOptions = {}) {
    const prototype = createPrototype(data, materialOptions);
    addMeshToScene(sceneKey, prototype);
    return prototype;
}

/**
 * Remove prototype from scene
 * @param {string} sceneKey - Key of the scene to remove prototype from
 * @param {THREE.Mesh} prototype - Prototype to remove
 */
export function removePrototypeFromScene(sceneKey, prototype) {
    removeMeshFromScene(sceneKey, prototype);
}

/**
 * Create prototype label
 * @param {string} text - Label text
 * @param {Object} options - Label options
 * @param {number} options.size - Canvas size
 * @param {string} options.font - Font style
 * @param {string} options.backgroundColor - Background color
 * @param {string} options.textColor - Text color
 * @returns {THREE.Sprite} Label sprite
 */
export function createPrototypeLabel(text, options = {}) {
    const {
        size = 128,
        font = '40px Arial',
        backgroundColor = 'rgba(0, 0, 0, 0.7)',
        textColor = 'white'
    } = options;

    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext('2d');

    // Draw background
    context.fillStyle = backgroundColor;
    context.fillRect(0, 0, size, size);

    // Draw text
    context.font = font;
    context.fillStyle = textColor;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, size / 2, size / 2);

    // Create texture and sprite
    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        opacity: 0.8
    });

    const sprite = new THREE.Sprite(material);
    sprite.scale.set(0.5, 0.5, 1);
    sprite.position.set(0, 0.4, 0);

    return sprite;
}

/**
 * Focus camera on prototype
 * @param {string} cameraKey - Key of the camera to use
 * @param {string} controlsKey - Key of the controls to update
 * @param {THREE.Mesh} prototype - Prototype to focus on
 */
export function focusOnPrototype(cameraKey, controlsKey, prototype) {
    const position = prototype.position.clone();
    position.z += 1.2; // Offset camera position
    positionCamera(cameraKey, controlsKey, position, prototype.position);
} 