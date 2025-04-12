import { VisualizationManager } from "./manager.js";
import { UI } from "./ui.js";

// Initialize visualization when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    try {
        // Get the container element
        const container = document.getElementById('unified-container');
        if (!container) {
            throw new Error('Could not find unified-container element');
        }

        // Create visualization manager
        const vizManager = new VisualizationManager(container);

        // Create UI and connect it to the manager
        const ui = new UI(vizManager);

        // Set initial camera position
        vizManager.resetView();

    } catch (error) {
        console.error('Error initializing application:', error);
        const errorOverlay = document.getElementById('error-overlay');
        if (errorOverlay) {
            errorOverlay.innerHTML = `<div class="font-bold mb-1">Initialization Error:</div><div>${error.message}</div>`;
            errorOverlay.classList.remove('hidden');
        }
    }
});
