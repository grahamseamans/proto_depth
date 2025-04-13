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

            // Helper to create camera transform matrix
            function createCameraTransform(pos, rot) {
                const rotMatrix = new THREE.Matrix4();
                const euler = new THREE.Euler(rot[0], rot[1], rot[2], 'YXZ');
                rotMatrix.makeRotationFromEuler(euler);

                const transMatrix = new THREE.Matrix4();
                transMatrix.makeTranslation(pos[0], pos[1], pos[2]);

                const transform = new THREE.Matrix4();
                transform.multiplyMatrices(transMatrix, rotMatrix);
                return transform;
            }

            // Helper to transform points using a camera transform
            function transformPoints(points, transform) {
                return points.map(point => {
                    const vec = new THREE.Vector3(point[0], point[1], point[2]);
                    vec.applyMatrix4(transform);
                    return [vec.x, vec.y, vec.z];
                });
            }

            // Create array of camera pairs [true, pred]
            const cameras = frameData.true?.camera?.positions.map((_, i) => ({
                true: {
                    pos: frameData.true.camera.positions[i],
                    rot: frameData.true.camera.rotations[i],
                    transform: createCameraTransform(
                        frameData.true.camera.positions[i],
                        frameData.true.camera.rotations[i]
                    ).invert()
                },
                pred: {
                    pos: frameData.pred.camera.positions[i],
                    rot: frameData.pred.camera.rotations[i],
                    transform: createCameraTransform(
                        frameData.pred.camera.positions[i],
                        frameData.pred.camera.rotations[i]
                    ).invert()
                }
            })) || [];

            // Process each camera pair
            cameras.forEach((camera, i) => {
                // Add frustums
                const trueFrustum = this.addFrustum(camera.true.pos, camera.true.rot, {
                    color: 0x0000ff,
                    opacity: 0.3
                });
                trueFrustum.visible = this.showFrustums;
                this.frustums.push(trueFrustum);

                const predFrustum = this.addFrustum(camera.pred.pos, camera.pred.rot, {
                    color: 0xff0000,
                    opacity: 0.3
                });
                predFrustum.visible = this.showFrustums;
                this.frustums.push(predFrustum);

                // Get this camera's point cloud
                const pointCloud = frameData.point_clouds?.[i];
                if (!Array.isArray(pointCloud) || pointCloud.length === 0) {
                    console.warn(`Point cloud ${i} is empty or not an array`);
                    return;
                }

                const points = pointCloud[0];
                if (!Array.isArray(points)) {
                    console.warn(`Point cloud ${i} inner data is not an array`);
                    return;
                }

                // Transform points to Three.js coordinate system
                const threePoints = transformToThreeSpace(points);

                // Show points with true transform (green)
                const trueWorldPoints = transformPoints(threePoints, camera.true.transform);
                const trueCloud = this.addPointCloud(trueWorldPoints, {
                    color: 0x00ff00,
                    opacity: 0.7,
                    size: 0.01
                });
                if (trueCloud) {
                    trueCloud.visible = this.showPointClouds;
                    this.pointClouds.push(trueCloud);
                }

                // Show points with pred transform (red)
                const predWorldPoints = transformPoints(threePoints, camera.pred.transform);
                const predCloud = this.addPointCloud(predWorldPoints, {
                    color: 0xff0000,
                    opacity: 0.7,
                    size: 0.01
                });
                if (predCloud) {
                    predCloud.visible = this.showPointClouds;
                    this.pointClouds.push(predCloud);
                }
            });

            // Add object meshes if bunny data is loaded (new format: true.objects and pred.objects)
            if (this.bunnyMeshData) {
                const objects = [];

                // Add true objects
                if (frameData.true?.objects) {
                    console.log('True objects:', {
                        positions: frameData.true.objects.positions,
                        rotations: frameData.true.objects.rotations,
                        scales: frameData.true.objects.scales
                    });
                    frameData.true.objects.positions.forEach((pos, i) => {
                        objects.push({
                            data: this.bunnyMeshData,
                            position: pos,
                            rotation: frameData.true.objects.rotations[i],
                            scale: frameData.true.objects.scales[i],
                            isTrue: true
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
                        objects.push({
                            data: this.bunnyMeshData,
                            position: pos,
                            rotation: frameData.pred.objects.rotations[i],
                            scale: frameData.pred.objects.scales[i],
                            isTrue: false
                        });
                    });
                } else {
                    console.warn('No predicted objects in frame data');
                }

                // Create and add meshes
                console.log('Creating meshes for objects:', objects);
                this.meshes = createMeshes(objects);
                this.meshes.forEach(mesh => {
                    mesh.visible = this.showMeshes;
                    this.scene.add(mesh);
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

    addFrustum(position, rotation, options = {}) {
        const frustum = createFrustum(position, rotation, options);
        this.scene.add(frustum);
        return frustum;
    }
}
