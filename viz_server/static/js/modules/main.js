import { VisualizationManager } from "./manager.js";

// Initialize visualization when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Get the container element
    const container = document.getElementById('unified-container');

    // Create visualization manager
    const vizManager = new VisualizationManager(container);

    // Example usage: render a point cloud (replace with real data as needed)
    const examplePoints = [
        [0, 0, 0],
        [1, 1, 1],
        [-1, -1, 0.5]
    ];

    vizManager.addPointCloud(examplePoints, {
        color: 0x00ff00,
        size: 0.1,
        opacity: 0.8
    });

    // Position camera
    vizManager.setCameraPosition(
        new THREE.Vector3(2, 2, 2),
        new THREE.Vector3(0, 0, 0)
    );
});
