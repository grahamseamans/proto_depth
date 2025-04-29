/**
 * Main visualization manager that handles state and coordinates between components
 */
import * as THREE from 'three';
import { setupVisualization } from './visualization/core.js';
import { addPointCloudToScene, transformToThreeSpace } from './visualization/point_clouds.js';
import { createFrustum } from './visualization/frustum.js';
import { createMeshes, loadObjFile } from './visualization/meshes.js';

export class VisualizationManager {
    constructor(container) {
        // Set up Three.js environment using core utilities
        const { scene, camera, renderer, controls } = setupVisualization(container);
        this.scene = scene;
        this.camera = camera;
        this.renderer = renderer;
        this.controls = controls;
        this.container = container;

        // Set initial background style based on current theme
        const theme = document.documentElement.getAttribute('data-theme') || 'light';
        this.setBackgroundStyle(theme);

        // Application state
        this.currentRun = null;
        this.currentIteration = 0;
        this.currentFrame = 0;
        this.runMetadata = null;

        // Visualization state
        this.pointClouds = [];
        this.frustums = [];
        this.meshes = [];
        this.showPointClouds = true;
        this.showFrustums = true;
        this.showMeshes = true;

        // Load bunny mesh data
        this.bunnyMeshData = null;
        this.loadBunnyMesh();
    }

    // Load bunny mesh data
    async loadBunnyMesh() {
        try {
            this.bunnyMeshData = await loadObjFile('/static/3d_models/bunny.obj');
            console.log('Bunny mesh loaded:', {
                vertices: this.bunnyMeshData.vertices.length,
                faces: this.bunnyMeshData.faces.length
            });
        } catch (error) {
            console.error('Error loading bunny mesh:', error);
        }
    }

    // Data loading methods
    async loadRun(runId) {
        try {
            // Load run metadata
            const metadataResponse = await fetch(`/api/run/${runId}/run_metadata.json`);
            if (!metadataResponse.ok) throw new Error('Failed to load run metadata');
            this.runMetadata = await metadataResponse.json();
            this.currentRun = runId;

            // Load iterations
            const iterResponse = await fetch(`/api/run/${runId}/epochs`);
            if (!iterResponse.ok) throw new Error('Failed to load iterations');
            const iterations = await iterResponse.json();

            // Reset state
            this.currentIteration = 0;
            this.currentFrame = 0;

            // Load initial frame
            await this.loadFrame(0);

            return {
                numIterations: iterations.length,
                numFrames: this.runMetadata.num_frames
            };
        } catch (error) {
            console.error('Error loading run:', error);
            throw error;
        }
    }

    async loadFrame(frame) {
        if (!this.currentRun) return;

        try {
            const response = await fetch(
                `/api/run/${this.currentRun}/iter/iter_${String(this.currentIteration).padStart(4, '0')}/frame_${String(frame).padStart(4, '0')}.json`
            );
            if (!response.ok) throw new Error('Failed to load frame data');
            const frameData = await response.json();

            // Clear current scene
            this.clearScene();


            // Process each camera pair
            const trueCameras = frameData.true?.camera?.transforms?.length || 0;
            const predCameras = frameData.pred?.camera?.transforms?.length || 0;

            if (trueCameras === 0 || predCameras === 0) {
                throw new Error('No cameras found in frame data');
            }
            if (trueCameras !== predCameras) {
                throw new Error(`Mismatched number of cameras: true=${trueCameras}, pred=${predCameras}`);
            }

            // Generate a color palette for cameras
            function getColorPalette(n) {
                // Evenly spaced hues in HSL, full saturation and 0.5 lightness
                return Array.from({ length: n }, (_, k) => {
                    const hue = Math.round((360 * k) / n);
                    return new THREE.Color(`hsl(${hue}, 100%, 50%)`).getHex();
                });
            }
            function lightenColor(hex, amount = 0.5) {
                // amount: 0 = no change, 1 = white
                const color = new THREE.Color(hex);
                color.lerp(new THREE.Color(0xffffff), amount);
                return color.getHex();
            }

            const cameraColors = getColorPalette(trueCameras);

            for (let i = 0; i < trueCameras; i++) {
                // Get camera transforms - now directly accessing 4x4 matrix array
                const trueMatrix = frameData.true.camera.transforms[i];
                const predMatrix = frameData.pred.camera.transforms[i];

                // Convert to column-major format for THREE.js
                const trueMatrixArray = [];
                const predMatrixArray = [];
                for (let col = 0; col < 4; col++) {
                    for (let row = 0; row < 4; row++) {
                        trueMatrixArray.push(trueMatrix[row][col]);
                        predMatrixArray.push(predMatrix[row][col]);
                    }
                }

                const trueTransform = new THREE.Matrix4().fromArray(trueMatrixArray);
                const predTransform = new THREE.Matrix4().fromArray(predMatrixArray);

                console.log(`Camera ${i} transforms:`, {
                    true: frameData.true.camera.transforms[i],
                    pred: frameData.pred.camera.transforms[i]
                });

                // Assign base color for this camera
                const baseColor = cameraColors[i];
                const predColor = lightenColor(baseColor, 0.5);

                // Add frustums
                const trueFrustum = this.addFrustum(trueTransform, {
                    color: baseColor,
                    opacity: 0.4,
                    showLookAt: true
                });
                trueFrustum.visible = this.showFrustums;
                this.frustums.push(trueFrustum);

                const predFrustum = this.addFrustum(predTransform, {
                    color: predColor,
                    opacity: 0.4,
                    showLookAt: true
                });
                predFrustum.visible = this.showFrustums;
                this.frustums.push(predFrustum);

                // Get this camera's point cloud (new format: true.point_clouds and pred.point_clouds)
                if (!frameData.true?.point_clouds || !Array.isArray(frameData.true.point_clouds)) {
                    throw new Error('No ground truth point clouds found in frame data');
                }
                if (!frameData.pred?.point_clouds || !Array.isArray(frameData.pred.point_clouds)) {
                    throw new Error('No predicted point clouds found in frame data');
                }

                const gtPoints = frameData.true.point_clouds[i];
                const predPoints = frameData.pred.point_clouds[i];

                if (!Array.isArray(gtPoints) || gtPoints.length === 0) {
                    throw new Error(`Ground truth point cloud ${i} is empty or not an array`);
                }
                if (!Array.isArray(predPoints) || predPoints.length === 0) {
                    throw new Error(`Predicted point cloud ${i} is empty or not an array`);
                }

                console.log(`Processing ground truth point cloud ${i} with ${gtPoints.length} points`);
                console.log(`Processing predicted point cloud ${i} with ${predPoints.length} points`);

                // Transform points to Three.js coordinate system
                const threeGTPoints = transformToThreeSpace(gtPoints);
                const threePredPoints = transformToThreeSpace(predPoints);

                // Points are already in world space; do not apply camera transform
                const trueWorldPoints = threeGTPoints;
                const predWorldPoints = threePredPoints;

                const trueCloud = this.addPointCloud(trueWorldPoints, {
                    color: baseColor,
                    opacity: 0.7,
                    size: 0.01
                });
                if (trueCloud) {
                    trueCloud.visible = this.showPointClouds;
                    this.pointClouds.push(trueCloud);
                }

                const predCloud = this.addPointCloud(predWorldPoints, {
                    color: predColor,
                    opacity: 0.7,
                    size: 0.01
                });
                if (predCloud) {
                    predCloud.visible = this.showPointClouds;
                    this.pointClouds.push(predCloud);
                }
            }

            // Add object meshes if bunny data is loaded (new format: true.objects and pred.objects)
            if (this.bunnyMeshData) {
                const objects = [];

                // Generate a color palette for meshes (true + pred)
                const numTrueMeshes = frameData.true?.objects?.positions?.length || 0;
                const numPredMeshes = frameData.pred?.objects?.positions?.length || 0;
                const totalMeshes = numTrueMeshes + numPredMeshes;
                const meshColors = getColorPalette(totalMeshes);

                let meshColorIdx = 0;

                // Add true objects
                if (frameData.true?.objects) {
                    console.log('True objects:', {
                        positions: frameData.true.objects.positions,
                        rotations: frameData.true.objects.rotations,
                        scales: frameData.true.objects.scales
                    });
                    frameData.true.objects.positions.forEach((pos, i) => {
                        const baseColor = meshColors[meshColorIdx++];
                        objects.push({
                            data: this.bunnyMeshData,
                            position: pos,
                            rotation: frameData.true.objects.rotations[i],
                            scale: frameData.true.objects.scales[i],
                            isTrue: true,
                            color: baseColor
                        });
                    });
                } else {
                    console.warn('No true objects in frame data');
                }

                // Add predicted objects
                if (frameData.pred?.objects) {
                    console.log('Predicted objects:', {
                        positions: frameData.pred.objects.positions,
                        rotations: frameData.pred.objects.rotations,
                        scales: frameData.pred.objects.scales
                    });
                    frameData.pred.objects.positions.forEach((pos, i) => {
                        // Use the same color index as true mesh if possible, else continue palette
                        const baseColor = meshColors[meshColorIdx - numPredMeshes + i] || meshColors[meshColorIdx++];
                        const predColor = lightenColor(baseColor, 0.5);
                        objects.push({
                            data: this.bunnyMeshData,
                            position: pos,
                            rotation: frameData.pred.objects.rotations[i],
                            scale: frameData.pred.objects.scales[i],
                            isTrue: false,
                            color: predColor
                        });
                    });
                } else {
                    console.warn('No predicted objects in frame data');
                }

                // Create and add meshes
                console.log('Creating meshes for objects:', objects);
                this.meshes = createMeshes(objects);
                this.meshes.forEach((mesh, idx) => {
                    mesh.visible = this.showMeshes;
                    this.scene.add(mesh);
                    // Debug: Log mesh world position, rotation, and scale
                    console.log(
                        `Mesh ${idx} debug:`,
                        {
                            position: mesh.position.toArray(),
                            rotation: [mesh.rotation.x, mesh.rotation.y, mesh.rotation.z],
                            quaternion: mesh.quaternion.toArray(),
                            scale: mesh.scale.toArray()
                        }
                    );
                });
                console.log('Added meshes to scene:', this.meshes.length);
            } else {
                console.warn('Bunny mesh data not loaded yet');
            }

            this.currentFrame = frame;
        } catch (error) {
            console.error('Error loading frame:', error);
            throw error;
        }
    }

    // Visualization control methods
    setVisibility(type, visible) {
        switch (type) {
            case 'pointClouds':
                this.showPointClouds = visible;
                this.pointClouds.forEach(cloud => cloud.visible = visible);
                break;
            case 'frustums':
                this.showFrustums = visible;
                this.frustums.forEach(frustum => frustum.visible = visible);
                break;
            case 'meshes':
                this.showMeshes = visible;
                this.meshes.forEach(mesh => mesh.visible = visible);
                break;
        }
    }

    setCameraPosition(position, target) {
        this.camera.position.copy(position);
        this.controls.target.copy(target);
        this.controls.update();
    }

    resetView() {
        this.camera.position.set(0, 1, 3);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    clearScene() {
        // Remove old objects
        this.pointClouds.forEach(cloud => this.scene.remove(cloud));
        this.frustums.forEach(frustum => this.scene.remove(frustum));
        this.meshes.forEach(mesh => this.scene.remove(mesh));

        // Clear arrays
        this.pointClouds = [];
        this.frustums = [];
        this.meshes = [];
    }

    // Helper methods
    addPointCloud(points, options = {}) {
        return addPointCloudToScene(this.scene, points, options);
    }

    addFrustum(transform, options = {}) {
        const frustum = createFrustum(transform, options);
        this.scene.add(frustum);
        return frustum;
    }

    /**
     * Set the 3D background color based on theme ("light" or "dark")
     * @param {string} theme
     */
    setBackgroundStyle(theme) {
        if (theme === 'dark') {
            this.renderer.setClearColor(0x111111);
            this.scene.background = new THREE.Color(0x111111);
        } else {
            this.renderer.setClearColor(0xffffff);
            this.scene.background = new THREE.Color(0xffffff);
        }
    }
}
