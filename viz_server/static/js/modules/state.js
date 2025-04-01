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
};

// Object references
const objects = {
    pointCloud: null,
    slots: [],
    prototypes: [],
};

// Three.js components
const renderers = {
    pointCloud: null,
    scene: null,
    prototypes: null,
};

const scenes = {
    pointCloud: null,
    scene: null,
    prototypes: null,
};

const cameras = {
    pointCloud: null,
    scene: null,
    prototypes: null,
};

const controls = {
    pointCloud: null,
    scene: null,
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
