/**
 * Three.js renderers and scene setup for the visualization application
 */

/**
 * Initialize Three.js renderers for all visualization panels
 */
function initializeRenderers() {
    // Unified scene for both point cloud and slots/shapes
    scenes.unified = new THREE.Scene();
    scenes.unified.background = new THREE.Color(0x15191E);
    cameras.unified = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    cameras.unified.position.set(0, 0, 0.01); // kills controls if at origin for some reason
    renderers.unified = new THREE.WebGLRenderer({ antialias: true });
    renderers.unified.setPixelRatio(window.devicePixelRatio);
    controls.unified = new THREE.OrbitControls(cameras.unified, renderers.unified.domElement);
    elements.unifiedContainer.appendChild(renderers.unified.domElement);

    // Initialize prototypes renderer (still needed for compatibility)
    scenes.prototypes = new THREE.Scene();
    scenes.prototypes.background = new THREE.Color(0x15191E);
    cameras.prototypes = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    cameras.prototypes.position.set(0, 0, 2);
    renderers.prototypes = new THREE.WebGLRenderer({ antialias: true });
    renderers.prototypes.setPixelRatio(window.devicePixelRatio);
    controls.prototypes = new THREE.OrbitControls(cameras.prototypes, renderers.prototypes.domElement);
    // Do not append to DOM since we're using a different approach now

    // Add axes helpers to all scenes
    addAxes(scenes.unified);
    addAxes(scenes.prototypes);

    // Add lighting to all scenes
    addLighting(scenes.unified);
    addLighting(scenes.prototypes);

    // Initial resize
    resizeRenderers();
}

/**
 * Add axes helpers to a Three.js scene
 */
function addAxes(scene, size = 1) {
    const axes = new THREE.AxesHelper(size);
    scene.add(axes);
}

/**
 * Add lighting to a Three.js scene
 */
function addLighting(scene) {
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);

    // Add directional lights from multiple angles
    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight1.position.set(1, 1, 1);
    scene.add(directionalLight1);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
    directionalLight2.position.set(-1, -1, -1);
    scene.add(directionalLight2);

    const directionalLight3 = new THREE.DirectionalLight(0xffffff, 0.2);
    directionalLight3.position.set(1, -1, 0);
    scene.add(directionalLight3);
}

/**
 * Resize all renderers to match their container elements
 */
function resizeRenderers() {
    // Unified view
    if (renderers.unified && elements.unifiedContainer.offsetWidth > 0) {
        const unifiedWidth = elements.unifiedContainer.clientWidth;
        const unifiedHeight = elements.unifiedContainer.clientHeight;
        renderers.unified.setSize(unifiedWidth, unifiedHeight);
        cameras.unified.aspect = unifiedWidth / unifiedHeight;
        cameras.unified.updateProjectionMatrix();
    }

    // Original prototypes renderer (for compatibility)
    if (renderers.prototypes) {
        // We need to keep it sized sensibly even if not attached to DOM
        renderers.prototypes.setSize(500, 500);
        if (cameras.prototypes) {
            cameras.prototypes.aspect = 1;
            cameras.prototypes.updateProjectionMatrix();
        }
    }

    // Individual prototype views are resized when they're created, and their
    // size is fixed to ensure a consistent grid layout
}

/**
 * Animation loop for Three.js renderers
 */
function animate() {
    requestAnimationFrame(animate);

    // Update and render the unified view
    if (controls.unified) controls.unified.update();
    if (renderers.unified) renderers.unified.render(scenes.unified, cameras.unified);

    // Update and render the old prototypes view (still needed for compatibility)
    if (controls.prototypes) controls.prototypes.update();
    if (renderers.prototypes && scenes.prototypes) {
        renderers.prototypes.render(scenes.prototypes, cameras.prototypes);
    }

    // Our individual prototype views are handled in their own animation loop
}

/**
 * Reset all camera views to default positions
 */
function resetAllViews() {
    // Reset unified view
    if (cameras.unified && controls.unified) {
        cameras.unified.position.set(0, 0, 2);
        cameras.unified.lookAt(0, 0, 0);
        controls.unified.reset();
    }

    // Reset original prototypes view (for compatibility)
    if (cameras.prototypes && controls.prototypes) {
        cameras.prototypes.position.set(0, 0, 2);
        cameras.prototypes.lookAt(0, 0, 0);
        controls.prototypes.reset();

        // Re-render this view
        if (renderers.prototypes && scenes.prototypes) {
            renderers.prototypes.render(scenes.prototypes, cameras.prototypes);
        }
    }

    // Reset all individual prototype views
    for (let i = 0; i < cameras.prototypeCameras.length; i++) {
        if (cameras.prototypeCameras[i] && controls.prototypeControls[i]) {
            cameras.prototypeCameras[i].position.set(0, 0, 1.2);
            cameras.prototypeCameras[i].lookAt(0, 0, 0);
            controls.prototypeControls[i].reset();

            // Re-render this view
            if (renderers.prototypeRenderers[i] && scenes.prototypeScenes[i]) {
                renderers.prototypeRenderers[i].render(scenes.prototypeScenes[i], cameras.prototypeCameras[i]);
            }
        }
    }
}

/**
 * Update depth image plane opacity based on slider
 */
function updateDepthImageOpacity(opacity) {
    if (objects.depthImagePlane && objects.depthImagePlane.material) {
        objects.depthImagePlane.material.opacity = opacity / 100;
    }
}
