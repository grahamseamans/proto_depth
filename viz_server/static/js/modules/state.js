/**
 * Global state and objects for the visualization application
 */

// Application state
window.state = {
    currentRun: null,
    currentEpoch: 0,
    currentFrame: 0,
    showAllFrames: true,
    pointCloudBounds: null,
};

// Three.js scenes
window.scenes = {
    unified: new THREE.Scene(),
};

// Add lighting to main scene
scenes.unified.background = new THREE.Color(0x15191E);
const ambientLight = new THREE.AmbientLight(0x404040);
scenes.unified.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLight.position.set(1, 1, 1);
scenes.unified.add(directionalLight);

// Add grid helper
const gridHelper = new THREE.GridHelper(2, 20, 0x888888, 0x444444);
scenes.unified.add(gridHelper);

// Add axes helper
const axesHelper = new THREE.AxesHelper(0.5);
scenes.unified.add(axesHelper);

// Three.js cameras
window.cameras = {
    unified: new THREE.PerspectiveCamera(75, 1, 0.001, 10),
};

// Set initial camera position
cameras.unified.position.set(0, 1, 3);
cameras.unified.lookAt(0, 0, 0);

// Three.js renderers
window.renderers = {
    unified: new THREE.WebGLRenderer({ antialias: true }),
};

// Three.js controls
window.controls = {
    unified: null,  // Will be initialized in main.js
};

// Scene objects
window.objects = {
    pointClouds: [],  // Array of point cloud objects
    meshes: [],       // Array of mesh objects (bunnies)
    cameras: [],      // Array of camera frustums
};

// DOM elements (will be initialized in ui.js)
window.elements = {
    epochSlider: null,
    epochDisplay: null,
    resetViewBtn: null,
    runSelector: null,
    togglePointCloud: null,
    showDepthImage: null,
    modalDepthContainer: null,
    closeDepthModal: null,
    depthImageModal: null,
    timeSlider: null,
    timeDisplay: null,
    toggleAllFrames: null,
    toggleCameras: null,
    toggleObjects: null,
};

// Colors for different objects
window.colors = [
    0xff0000,  // Red
    0x00ff00,  // Green
    0x0000ff,  // Blue
    0xffff00,  // Yellow
    0xff00ff,  // Magenta
    0x00ffff,  // Cyan
];
