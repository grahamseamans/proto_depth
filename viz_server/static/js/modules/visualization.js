import { createScene, createCamera, createRenderer, createOrbitControls, setupVisualization } from './visualization/core.js';
import { renderPointCloud } from './visualization/point_clouds.js';
import { addPrototypeToScene } from './visualization/prototypes.js';

/**
 * High-level function to render a point cloud visualization
 * @param {Object} options - Configuration options
 * @param {Array} options.points - Array of point data
 * @param {string} options.containerId - ID of the container element
 * @param {Object} options.color - Color configuration
 * @param {number} options.opacity - Point opacity
 */
export function renderPointCloudVisualization(options) {
    const { points, containerId, color, opacity } = options;
    const scene = createScene();
    const camera = createCamera();
    const renderer = createRenderer();
    const controls = createOrbitControls(camera, renderer.domElement);

    setupVisualization(scene, camera, renderer, controls, containerId);
    renderPointCloud(scene, points, { color, opacity });
}

/**
 * High-level function to render prototype visualizations
 * @param {Object} options - Configuration options
 * @param {Array} options.prototypes - Array of prototype data
 * @param {string} options.containerId - ID of the container element
 * @param {Object} options.color - Color configuration
 * @param {boolean} options.wireframe - Whether to show wireframe
 */
export function renderPrototypeVisualization(options) {
    const { prototypes, containerId, color, wireframe } = options;
    const scene = createScene();
    const camera = createCamera();
    const renderer = createRenderer();
    const controls = createOrbitControls(camera, renderer.domElement);

    setupVisualization(scene, camera, renderer, controls, containerId);
    prototypes.forEach(prototype => {
        addPrototypeToScene(scene, prototype, { color, wireframe });
    });
}

/**
 * High-level function to render a time-varying scene
 * @param {Object} options - Configuration options
 * @param {Array} options.frames - Array of frame data
 * @param {string} options.containerId - ID of the container element
 * @param {Function} options.onFrameChange - Callback for frame changes
 */
export function renderTimeVaryingScene(options) {
    const { frames, containerId, onFrameChange } = options;
    const scene = createScene();
    const camera = createCamera();
    const renderer = createRenderer();
    const controls = createOrbitControls(camera, renderer.domElement);

    setupVisualization(scene, camera, renderer, controls, containerId);
    let currentFrame = 0;

    function animate() {
        requestAnimationFrame(animate);
        if (frames[currentFrame]) {
            scene.clear();
            frames[currentFrame].forEach(object => {
                if (object.type === 'pointCloud') {
                    renderPointCloud(scene, object.points, object.options);
                } else if (object.type === 'prototype') {
                    addPrototypeToScene(scene, object.data, object.options);
                }
            });
            onFrameChange?.(currentFrame);
        }
        renderer.render(scene, camera);
    }

    animate();
} 