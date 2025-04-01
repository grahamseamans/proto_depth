/**
 * Module initialization ordering and circular dependency resolution
 */

// Initialize the application
async function initializeApplication() {
    try {
        console.log("Initializing application...");
        // Set up cross-module communication to resolve circular dependencies
        setDataHandlers(loadRun, loadEpoch);

        // Initialize UI
        setupEventListeners();

        // Initialize renderers
        initializeRenderers();

        // Load initial data
        await loadRuns();

        // Start animation loop
        animate();
    } catch (error) {
        console.error("Error during initialization:", error);
    }
}
