/**
 * DOM element references for the visualization application
 */

// Cache all DOM elements used in the application
const elements = {
    runSelector: document.getElementById('run-selector'),
    epochSlider: document.getElementById('epoch-slider'),
    epochDisplay: document.getElementById('epoch-display'),
    resetViewBtn: document.getElementById('reset-view-btn'),
    navMain: document.getElementById('nav-main'),
    navPrototypes: document.getElementById('nav-prototypes'),
    navMainMobile: document.getElementById('nav-main-mobile'),
    navPrototypesMobile: document.getElementById('nav-prototypes-mobile'),
    mainView: document.getElementById('main-view'),
    prototypesView: document.getElementById('prototypes-view'),
    depthContainer: document.getElementById('depth-container'),
    pointCloudContainer: document.getElementById('pointcloud-container'),
    sceneContainer: document.getElementById('scene-container'),
    prototypesContainer: document.getElementById('prototypes-container'),
    prototypeWeightsContainer: document.getElementById('prototype-weights-container'),
    slotToggles: document.getElementById('slot-toggles'),
    weightChart: document.getElementById('weight-chart'),
    loading: document.getElementById('loading'),
};

/**
 * Show or hide the loading overlay
 */
function showLoading(show) {
    if (show) {
        elements.loading.classList.remove('hidden');
    } else {
        elements.loading.classList.add('hidden');
    }
}
