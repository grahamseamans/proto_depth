/**
 * UI event handlers and interaction for the visualization application
 */

// Function reference holders to avoid circular dependencies
let _loadRun, _loadEpoch;

function setDataHandlers(loadRunFn, loadEpochFn) {
    _loadRun = loadRunFn;
    _loadEpoch = loadEpochFn;
}

/**
 * Initialize UI event listeners
 */
function initUI() {
    // Run selector
    elements.runSelector.addEventListener('change', async (e) => {
        const runId = e.target.value;
        if (runId) {
            await loadRun(runId);
        }
    });

    // Epoch slider
    elements.epochSlider.addEventListener('input', async (e) => {
        const epochIndex = parseInt(e.target.value);
        if (state.epochs && epochIndex >= 0 && epochIndex < state.epochs.length) {
            state.currentEpochIndex = epochIndex;
            await loadEpoch(state.epochs[epochIndex]);
        }
    });

    // Reset view button
    elements.resetViewBtn.addEventListener('click', () => {
        resetAllViews();
    });

    // Navigation tabs
    elements.navMain.addEventListener('click', () => switchTab('main'));
    elements.navPrototypes.addEventListener('click', () => switchTab('prototypes'));
    elements.navMainMobile.addEventListener('click', () => switchTab('main'));
    elements.navPrototypesMobile.addEventListener('click', () => switchTab('prototypes'));

    // Toggle point cloud visibility
    elements.togglePointCloud.addEventListener('change', (e) => {
        if (objects.pointCloud) {
            objects.pointCloud.visible = e.target.checked;
        }
    });

    // Show depth image button
    elements.showDepthImage.addEventListener('click', () => {
        showDepthImageModal();
    });

    // Close depth image modal
    elements.closeDepthModal.addEventListener('click', () => {
        hideDepthImageModal();
    });

    // Window resize listener
    window.addEventListener('resize', () => {
        resizeRenderers();
    });
}

/**
 * Show the depth image in a modal
 */
function showDepthImageModal() {
    if (state.currentRun && state.currentEpoch && state.currentBatch) {
        const runId = state.currentRun;
        const epochId = state.currentEpoch.id;
        const batchId = state.currentBatch;

        if (state.batchData && state.batchData.has_depth_image) {
            const imgUrl = `/api/run/${runId}/epoch/${epochId}/batch/${batchId}/depth_img.png`;
            elements.modalDepthContainer.innerHTML = `<img src="${imgUrl}" alt="Depth Image" style="max-width: 100%;">`;
        } else {
            elements.modalDepthContainer.innerHTML = '<div class="text-center p-4">No depth image available</div>';
        }
    }

    // Show the modal
    elements.depthImageModal.classList.add('modal-open');
}

/**
 * Hide the depth image modal
 */
function hideDepthImageModal() {
    elements.depthImageModal.classList.remove('modal-open');
}

/**
 * Switch between main and prototypes tabs
 */
function switchTab(tab) {
    // Update active tab styling
    elements.navMain.classList.remove('active');
    elements.navPrototypes.classList.remove('active');
    elements.navMainMobile.classList.remove('active');
    elements.navPrototypesMobile.classList.remove('active');

    if (tab === 'main') {
        elements.navMain.classList.add('active');
        elements.navMainMobile.classList.add('active');
        elements.mainView.classList.remove('hidden');
        elements.prototypesView.classList.add('hidden');

        // Ensure the main renderer is properly preserved/restored
        if (renderers.unified && renderers.unified.domElement) {
            // Make sure it's still in the DOM
            if (!renderers.unified.domElement.parentNode) {
                elements.unifiedContainer.appendChild(renderers.unified.domElement);
                console.log("Re-attached main renderer to DOM");
            }

            // If we've lost the WebGL context, try to recover
            try {
                // Force a render to check if context is alive
                renderers.unified.render(scenes.unified, cameras.unified);
            } catch (e) {
                console.warn("Main renderer WebGL context may be lost, attempting recovery");
                // If we get here, we might have lost the context - try to recover
                const oldDomElement = renderers.unified.domElement;

                // Create a new renderer with same settings
                renderers.unified = new THREE.WebGLRenderer({
                    antialias: true,
                    powerPreference: 'high-performance',
                    preserveDrawingBuffer: true
                });
                renderers.unified.setPixelRatio(window.devicePixelRatio);
                renderers.unified.isPermanent = true;

                // Replace the old DOM element
                if (oldDomElement.parentNode) {
                    oldDomElement.parentNode.replaceChild(renderers.unified.domElement, oldDomElement);
                } else {
                    elements.unifiedContainer.appendChild(renderers.unified.domElement);
                }

                // Recreate controls
                controls.unified = new THREE.OrbitControls(cameras.unified, renderers.unified.domElement);

                // Resize to match container
                const width = elements.unifiedContainer.clientWidth;
                const height = elements.unifiedContainer.clientHeight;
                renderers.unified.setSize(width, height);

                console.log("Main renderer recreated after context loss");
            }
        }
    } else {
        elements.navPrototypes.classList.add('active');
        elements.navPrototypesMobile.classList.add('active');
        elements.mainView.classList.add('hidden');
        elements.prototypesView.classList.remove('hidden');
    }

    // Resize renderers after tab switch
    setTimeout(() => resizeRenderers(), 10);
}

/**
 * Update slot toggle buttons based on current slot data
 */
function updateSlotToggles(slotsData) {
    if (!slotsData || slotsData.length === 0) {
        elements.slotToggles.innerHTML = '<div class="text-sm text-gray-500">No slots available</div>';
        return;
    }

    // Create toggle buttons
    elements.slotToggles.innerHTML = '';

    // Add "Toggle All" button
    const allButton = document.createElement('button');
    allButton.className = 'btn btn-sm btn-primary';
    allButton.textContent = 'Toggle All';
    allButton.addEventListener('click', () => {
        const allVisible = objects.slots.every(slot => slot.visible);
        toggleAllSlots(!allVisible);
    });
    elements.slotToggles.appendChild(allButton);

    // Add individual slot toggle buttons
    slotsData.forEach((slot, index) => {
        const slotId = slot.id;
        const button = document.createElement('button');
        button.className = 'btn btn-sm';
        button.style.backgroundColor = `#${colors[index % colors.length].toString(16).padStart(6, '0')}`;
        button.style.color = 'white';
        button.textContent = `Slot ${index + 1}`;

        button.addEventListener('click', () => {
            toggleSlotVisibility(slotId);
            updateToggleButtonStyles();
        });

        elements.slotToggles.appendChild(button);
    });

    updateToggleButtonStyles();
}

/**
 * Toggle visibility of a specific slot
 */
function toggleSlotVisibility(slotId) {
    state.slotVisibility[slotId] = !state.slotVisibility[slotId];

    // Update mesh visibility
    for (const mesh of objects.slots) {
        if (mesh.userData.slotId === slotId) {
            mesh.visible = state.slotVisibility[slotId];
            break;
        }
    }
}

/**
 * Toggle visibility of all slots
 */
function toggleAllSlots(visible) {
    for (const slotId in state.slotVisibility) {
        state.slotVisibility[slotId] = visible;
    }

    // Update mesh visibility
    for (const mesh of objects.slots) {
        mesh.visible = visible;
    }

    updateToggleButtonStyles();
}

/**
 * Update toggle button styles based on current visibility
 */
function updateToggleButtonStyles() {
    const buttons = elements.slotToggles.querySelectorAll('.btn:not(:first-child)');
    buttons.forEach((button, index) => {
        const slotId = `slot_${index + 1}`;
        if (state.slotVisibility[slotId]) {
            button.classList.add('opacity-100');
            button.classList.remove('opacity-50');
        } else {
            button.classList.add('opacity-50');
            button.classList.remove('opacity-100');
        }
    });
}
