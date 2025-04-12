/**
 * Main visualization manager that handles state and coordinates between components
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

// Pure visualization functions
import { addPointCloudToScene } from './visualization/point_clouds.js';
import { createPrototype } from './visualization/components/prototype.js';
import { createFrustum } from './visualization/frustum.js';

export class VisualizationManager {
    constructor(container) {
        this.container = container;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);

        // Setup renderer
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.container.appendChild(this.renderer.domElement);

        // Setup controls
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Setup lights
        const ambientLight = new THREE.AmbientLight(0x404040);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(ambientLight, directionalLight);

        // Animation loop
        this.animate = this.animate.bind(this);
        this.animate();

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }

    animate() {
        requestAnimationFrame(this.animate);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    // Visualization methods that delegate to pure functions
    addPointCloud(points, options = {}) {
        return addPointCloudToScene(this.scene, points, options);
    }

    addPrototype(prototypeData, options = {}) {
        const prototype = createPrototype(prototypeData, {
            color: options.color || 0x00ff00,
            wireframe: options.wireframe || false
        });
        this.scene.add(prototype);
        return prototype;
    }

    addFrustum(position, direction, options = {}) {
        const frustum = createFrustum(position, direction, {
            color: options.color || 0x0000ff,
            size: options.size || 1.0
        });
        this.scene.add(frustum);
        return frustum;
    }

    // State management methods
    setCameraPosition(position, target) {
        this.camera.position.copy(position);
        this.controls.target.copy(target);
        this.controls.update();
    }

    clearScene() {
        while (this.scene.children.length > 0) {
            this.scene.remove(this.scene.children[0]);
        }
    }
} 