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
    unifiedContainer: document.getElementById('unified-container'),
    prototypesContainer: document.getElementById('prototypes-container'),
    prototypeWeightsContainer: document.getElementById('prototype-weights-container'),
    slotToggles: document.getElementById('slot-toggles'),
    weightChart: document.getElementById('weight-chart'),
    loading: document.getElementById('loading'),

    // New UI controls
    togglePointCloud: document.getElementById('toggle-pointcloud'),
    showDepthImage: document.getElementById('show-depth-image'),
    depthImageModal: document.getElementById('depth-image-modal'),
    closeDepthModal: document.getElementById('close-depth-modal'),
    modalDepthContainer: document.getElementById('modal-depth-container'),
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
