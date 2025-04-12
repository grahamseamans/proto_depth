/**
 * Main visualization manager that handles state and coordinates between components
 */
import * as THREE from 'three';
import { setupVisualization } from './visualization/core.js';
import { addPointCloudToScene } from './visualization/point_clouds.js';
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
            console.log('Bunny mesh loaded successfully');
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

            // Add point clouds
            if (frameData.true?.point_clouds) {
                frameData.true.point_clouds.forEach(points => {
                    const cloud = this.addPointCloud(points, {
                        color: 0x00ff00,
                        opacity: 0.7
                    });
                    if (cloud) {
                        cloud.visible = this.showPointClouds;
                        this.pointClouds.push(cloud);
                    }
                });
            }
            if (frameData.pred?.point_clouds) {
                frameData.pred.point_clouds.forEach(points => {
                    const cloud = this.addPointCloud(points, {
                        color: 0xff0000,
                        opacity: 0.7
                    });
                    if (cloud) {
                        cloud.visible = this.showPointClouds;
                        this.pointClouds.push(cloud);
                    }
                });
            }

            // Add cameras/frustums
            if (frameData.cameras) {
                frameData.cameras.positions.forEach((pos, i) => {
                    const rot = frameData.cameras.rotations[i];
                    // True camera (blue)
                    const trueFrustum = this.addFrustum(pos, rot, {
                        color: 0x0000ff,
                        opacity: 0.3
                    });
                    trueFrustum.visible = this.showFrustums;
                    this.frustums.push(trueFrustum);

                    // Predicted camera (red)
                    const predFrustum = this.addFrustum(pos, rot, {
                        color: 0xff0000,
                        opacity: 0.3
                    });
                    predFrustum.visible = this.showFrustums;
                    this.frustums.push(predFrustum);
                });
            }

            // Add object meshes if bunny data is loaded
            if (this.bunnyMeshData) {
                const objects = [];
                if (frameData.true?.positions) {
                    frameData.true.positions.forEach((pos, i) => {
                        objects.push({
                            data: this.bunnyMeshData,
                            position: pos,
                            rotation: frameData.true.rotations[i],
                            scale: frameData.true.scales[i],
                            isTrue: true
                        });
                    });
                }
                if (frameData.pred?.positions) {
                    frameData.pred.positions.forEach((pos, i) => {
                        objects.push({
                            data: this.bunnyMeshData,
                            position: pos,
                            rotation: frameData.pred.rotations[i],
                            scale: frameData.pred.scales[i],
                            isTrue: false
                        });
                    });
                }
                this.meshes = createMeshes(objects);
                this.meshes.forEach(mesh => {
                    mesh.visible = this.showMeshes;
                    this.scene.add(mesh);
                });
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

    addFrustum(position, rotation, options = {}) {
        const frustum = createFrustum(position, rotation, options);
        this.scene.add(frustum);
        return frustum;
    }
}
