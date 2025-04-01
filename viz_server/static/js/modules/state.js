/**
 * State management for the visualization application
 */

// Application state object
const state = {
    runs: [],
    currentRun: null,
    epochs: [],
    currentEpoch: null,
    currentBatch: null,
    currentEpochIndex: 0,
    batchData: null,
    slotVisibility: {},
    pointCloudBounds: null, // Store point cloud bounds for depth image alignment
};

// Object references
const objects = {
    pointCloud: null,
    slots: [],
    prototypes: [],
    depthImagePlane: null, // Reference to the depth image plane in 3D space
};

// Three.js components
const renderers = {
    unified: null, // Single renderer for the integrated view
    prototypes: null,
};

const scenes = {
    unified: null, // Single scene containing both point cloud and slots
    prototypes: null,
};

const cameras = {
    unified: null, // Single camera for the integrated view
    prototypes: null,
};

const controls = {
    unified: null, // Single controls for the integrated view
    prototypes: null,
};

// Charts
let weightsChart = null;
const setWeightsChart = (chart) => {
    weightsChart = chart;
};

// Colors for slots and prototypes
const colors = [
    0xff0000, // red
    0x00ff00, // green
    0x0000ff, // blue
    0xff00ff, // magenta
    0x00ffff, // cyan
    0xffff00, // yellow
    0xff8000, // orange
    0x8000ff, // purple
];
