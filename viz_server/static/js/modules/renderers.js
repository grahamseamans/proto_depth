/**
 * Three.js renderers and scene setup for the visualization application
 */

/**
 * Initialize Three.js renderers for all visualization panels
 */
function initializeRenderers() {
    // Point Cloud renderer
    scenes.pointCloud = new THREE.Scene();
    scenes.pointCloud.background = new THREE.Color(0xf0f0f0);
    cameras.pointCloud = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    cameras.pointCloud.position.set(0, 0, 2);
    renderers.pointCloud = new THREE.WebGLRenderer({ antialias: true });
    renderers.pointCloud.setPixelRatio(window.devicePixelRatio);
    controls.pointCloud = new THREE.OrbitControls(cameras.pointCloud, renderers.pointCloud.domElement);
    elements.pointCloudContainer.appendChild(renderers.pointCloud.domElement);

    // 3D Scene renderer
    scenes.scene = new THREE.Scene();
    scenes.scene.background = new THREE.Color(0xf0f0f0);
    cameras.scene = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    cameras.scene.position.set(0, 0, 2);
    renderers.scene = new THREE.WebGLRenderer({ antialias: true });
    renderers.scene.setPixelRatio(window.devicePixelRatio);
    controls.scene = new THREE.OrbitControls(cameras.scene, renderers.scene.domElement);
    elements.sceneContainer.appendChild(renderers.scene.domElement);

    // Prototypes renderer
    scenes.prototypes = new THREE.Scene();
    scenes.prototypes.background = new THREE.Color(0xf0f0f0);
    cameras.prototypes = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    cameras.prototypes.position.set(0, 0, 2);
    renderers.prototypes = new THREE.WebGLRenderer({ antialias: true });
    renderers.prototypes.setPixelRatio(window.devicePixelRatio);
    controls.prototypes = new THREE.OrbitControls(cameras.prototypes, renderers.prototypes.domElement);
    elements.prototypesContainer.appendChild(renderers.prototypes.domElement);

    // Add axes helpers to all scenes
    addAxes(scenes.pointCloud);
    addAxes(scenes.scene);
    addAxes(scenes.prototypes);

    // Add lighting to all scenes
    addLighting(scenes.pointCloud);
    addLighting(scenes.scene);
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
    // Point Cloud
    if (renderers.pointCloud && elements.pointCloudContainer.offsetWidth > 0) {
        const pointCloudWidth = elements.pointCloudContainer.clientWidth;
        const pointCloudHeight = elements.pointCloudContainer.clientHeight;
        renderers.pointCloud.setSize(pointCloudWidth, pointCloudHeight);
        cameras.pointCloud.aspect = pointCloudWidth / pointCloudHeight;
        cameras.pointCloud.updateProjectionMatrix();
    }

    // 3D Scene
    if (renderers.scene && elements.sceneContainer.offsetWidth > 0) {
        const sceneWidth = elements.sceneContainer.clientWidth;
        const sceneHeight = elements.sceneContainer.clientHeight;
        renderers.scene.setSize(sceneWidth, sceneHeight);
        cameras.scene.aspect = sceneWidth / sceneHeight;
        cameras.scene.updateProjectionMatrix();
    }

    // Prototypes
    if (renderers.prototypes && elements.prototypesContainer.offsetWidth > 0) {
        const prototypesWidth = elements.prototypesContainer.clientWidth;
        const prototypesHeight = elements.prototypesContainer.clientHeight;
        renderers.prototypes.setSize(prototypesWidth, prototypesHeight);
        cameras.prototypes.aspect = prototypesWidth / prototypesHeight;
        cameras.prototypes.updateProjectionMatrix();
    }
}

/**
 * Animation loop for Three.js renderers
 */
function animate() {
    requestAnimationFrame(animate);

    if (controls.pointCloud) controls.pointCloud.update();
    if (controls.scene) controls.scene.update();
    if (controls.prototypes) controls.prototypes.update();

    if (renderers.pointCloud) renderers.pointCloud.render(scenes.pointCloud, cameras.pointCloud);
    if (renderers.scene) renderers.scene.render(scenes.scene, cameras.scene);
    if (renderers.prototypes) renderers.prototypes.render(scenes.prototypes, cameras.prototypes);
}

/**
 * Reset all camera views to default positions
 */
function resetAllViews() {
    // Reset point cloud view
    cameras.pointCloud.position.set(0, 0, 2);
    cameras.pointCloud.lookAt(0, 0, 0);
    controls.pointCloud.reset();

    // Reset scene view
    cameras.scene.position.set(0, 0, 2);
    cameras.scene.lookAt(0, 0, 0);
    controls.scene.reset();

    // Reset prototypes view
    cameras.prototypes.position.set(0, 0, 2);
    cameras.prototypes.lookAt(0, 0, 0);
    controls.prototypes.reset();
}
