/**
 * Main application entry point and animation loop
 */

/**
 * Main application initialization and animation loop
 */

// Initialize Three.js scene and renderer
function initThreeJs() {
    try {
        // Get the main container
        const container = document.getElementById('unified-container');
        if (!container) {
            throw new Error('Main container not found');
        }

        // Set up renderer
        renderers.unified.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderers.unified.domElement);

        // Make renderer fill container
        renderers.unified.domElement.style.width = '100%';
        renderers.unified.domElement.style.height = '100%';

        // Set up orbit controls
        controls.unified = new THREE.OrbitControls(cameras.unified, renderers.unified.domElement);
        controls.unified.enableDamping = true;
        controls.unified.dampingFactor = 0.25;

        // Handle window resizing
        function onWindowResize() {
            const rect = container.getBoundingClientRect();
            cameras.unified.aspect = rect.width / rect.height;
            cameras.unified.updateProjectionMatrix();
            renderers.unified.setSize(rect.width, rect.height);
        }
        window.addEventListener('resize', onWindowResize);
        onWindowResize();  // Initial size

        console.log('Three.js initialized successfully');
        return true;
    } catch (error) {
        console.error('Error initializing Three.js:', error);
        return false;
    }
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);

    try {
        // Update controls
        if (controls.unified) {
            controls.unified.update();
        }

        // Update any animated objects here
        if (objects.meshes && state.showAllFrames) {
            // Add subtle rotation to all meshes when showing all frames
            objects.meshes.forEach(mesh => {
                mesh.rotation.y += 0.001;
            });
        }

        // Render scene
        if (renderers.unified && scenes.unified && cameras.unified) {
            renderers.unified.render(scenes.unified, cameras.unified);
        }
    } catch (error) {
        console.error('Error in animation loop:', error);
    }
}

// Initialize everything when the DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    try {
        console.log('Initializing application...');

        // Set dark theme by default
        document.documentElement.setAttribute('data-theme', 'dark');

        // Initialize Three.js first
        if (!initThreeJs()) {
            throw new Error('Failed to initialize Three.js');
        }

        // Initialize UI elements
        initUI();

        // Start animation loop
        animate();

        console.log('Application initialized successfully');
    } catch (error) {
        console.error('Error initializing application:', error);
        // Show error to user
        const container = document.getElementById('unified-container');
        if (container) {
            container.innerHTML = `
                <div class="flex items-center justify-center h-full">
                    <div class="text-error">
                        <h3 class="font-bold">Error Initializing Application</h3>
                        <p>${error.message}</p>
                    </div>
                </div>
            `;
        }
    }
});
