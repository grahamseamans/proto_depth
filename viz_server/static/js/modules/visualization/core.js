/**
 * Core visualization functions for Three.js setup and management
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * Create a basic Three.js scene with default lighting
 * @returns {THREE.Scene} Configured Three.js scene
 */
export function createScene() {
    const scene = new THREE.Scene();

    // Add default lighting
    const ambientLight = new THREE.AmbientLight(0x404040);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(1, 1, 1);
    scene.add(ambientLight, directionalLight);

    return scene;
}

/**
 * Create a perspective camera
 * @param {Object} options - Camera configuration
 * @param {number} options.fov - Field of view (default: 75)
 * @param {number} options.aspect - Aspect ratio (default: window width/height)
 * @param {number} options.near - Near plane (default: 0.1)
 * @param {number} options.far - Far plane (default: 1000)
 * @returns {THREE.PerspectiveCamera} Configured camera
 */
export function createCamera(options = {}) {
    const {
        fov = 75,
        aspect = window.innerWidth / window.innerHeight,
        near = 0.1,
        far = 1000
    } = options;

    return new THREE.PerspectiveCamera(fov, aspect, near, far);
}

/**
 * Create a WebGL renderer
 * @param {Object} options - Renderer configuration
 * @param {boolean} options.antialias - Enable antialiasing (default: true)
 * @param {number} options.width - Width (default: window width)
 * @param {number} options.height - Height (default: window height)
 * @returns {THREE.WebGLRenderer} Configured renderer
 */
export function createRenderer(options = {}) {
    const {
        antialias = true,
        width = window.innerWidth,
        height = window.innerHeight
    } = options;

    const renderer = new THREE.WebGLRenderer({ antialias });
    renderer.setSize(width, height);
    return renderer;
}

/**
 * Create orbit controls for a camera
 * @param {THREE.Camera} camera - Camera to control
 * @param {HTMLElement} domElement - DOM element to attach controls to
 * @param {Object} options - Control configuration
 * @returns {OrbitControls} Configured controls
 */
export function createOrbitControls(camera, domElement, options = {}) {
    const controls = new OrbitControls(camera, domElement);

    // Default settings
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 0.1;
    controls.maxDistance = 100;
    controls.maxPolarAngle = Math.PI;

    // Apply any custom options
    Object.assign(controls, options);

    return controls;
}

/**
 * Set up a complete Three.js visualization environment
 * @param {HTMLElement} container - DOM element to attach the renderer to
 * @param {Object} options - Configuration options
 * @returns {Object} { scene, camera, renderer, controls }
 */
export function setupVisualization(container, options = {}) {
    const scene = createScene();
    const camera = createCamera(options.camera);
    const renderer = createRenderer(options.renderer);
    const controls = createOrbitControls(camera, renderer.domElement, options.controls);

    // Add renderer to container
    container.appendChild(renderer.domElement);

    // Setup animation loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // Handle window resize
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    return { scene, camera, renderer, controls };
}
