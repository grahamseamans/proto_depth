/**
 * 3D visualization functions for point clouds, slots, and prototypes
 */

/**
 * Render the point cloud from batch data in the unified scene
 */
function renderPointCloud(pointCloudData) {
    // Clear existing point cloud
    if (objects.pointCloud) {
        scenes.unified.remove(objects.pointCloud);
        objects.pointCloud = null;
    }

    if (!pointCloudData || !pointCloudData.points || pointCloudData.points.length === 0) {
        console.warn('No point cloud data available');
        return;
    }

    // Create point cloud geometry
    const points = pointCloudData.points;
    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);

    let minZ = Infinity;
    let maxZ = -Infinity;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (let i = 0; i < points.length; i++) {
        const p = points[i];
        vertices[i * 3] = p[0];
        vertices[i * 3 + 1] = p[1];
        vertices[i * 3 + 2] = p[2];

        minX = Math.min(minX, p[0]);
        maxX = Math.max(maxX, p[0]);
        minY = Math.min(minY, p[1]);
        maxY = Math.max(maxY, p[1]);
        minZ = Math.min(minZ, p[2]);
        maxZ = Math.max(maxZ, p[2]);
    }

    // Color points by depth
    const colorScale = (maxZ > minZ) ? (maxZ - minZ) : 1;
    for (let i = 0; i < points.length; i++) {
        const z = points[i][2];
        const t = (z - minZ) / colorScale;

        // Blue to green to red color gradient
        if (t < 0.5) {
            const u = t * 2;
            colors[i * 3] = 0;
            colors[i * 3 + 1] = u;
            colors[i * 3 + 2] = 1 - u;
        } else {
            const u = (t - 0.5) * 2;
            colors[i * 3] = u;
            colors[i * 3 + 1] = 1 - u;
            colors[i * 3 + 2] = 0;
        }
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Create point material
    const material = new THREE.PointsMaterial({
        size: 0.05,
        vertexColors: true,
    });

    // Create point cloud object
    objects.pointCloud = new THREE.Points(geometry, material);
    scenes.unified.add(objects.pointCloud);

    // Toggle visibility based on UI
    objects.pointCloud.visible = elements.togglePointCloud.checked;

    // Calculate bounding box and center camera
    geometry.computeBoundingBox();
    const boundingBox = geometry.boundingBox;

    // Center cameras on the point cloud
    const center = new THREE.Vector3();
    boundingBox.getCenter(center);

    const size = new THREE.Vector3();
    boundingBox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);

    // Position camera to view the entire point cloud
    const distance = maxDim * 2;
    cameras.unified.position.set(center.x, center.y, center.z + distance);
    cameras.unified.lookAt(center);
    controls.unified.target.copy(center);

    // Store point cloud bounds for future reference
    state.pointCloudBounds = {
        min: { x: minX, y: minY, z: minZ },
        max: { x: maxX, y: maxY, z: maxZ },
        center: center.clone(),
        size: size.clone()
    };
}

/**
 * Render slot meshes from batch data in the unified scene
 */
function renderSlots(slotsData) {
    // Clear existing slots
    for (const slot of objects.slots) {
        scenes.unified.remove(slot);
    }
    objects.slots = [];

    if (!slotsData || slotsData.length === 0) {
        console.warn('No slot data available');
        return;
    }

    slotsData.forEach((slot, index) => {
        const slotId = slot.id;
        const vertices = slot.data.vertices;
        const faces = slot.data.faces;

        if (!vertices || !faces || vertices.length === 0 || faces.length === 0) {
            console.warn(`Slot ${slotId} has no valid mesh data`);
            return;
        }

        // Create mesh geometry
        const geometry = new THREE.BufferGeometry();

        // Create vertex positions attribute
        const positions = new Float32Array(vertices.length * 3);
        for (let i = 0; i < vertices.length; i++) {
            positions[i * 3] = vertices[i][0];
            positions[i * 3 + 1] = vertices[i][1];
            positions[i * 3 + 2] = vertices[i][2];
        }
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        // Create faces
        const indices = new Uint32Array(faces.length * 3);
        for (let i = 0; i < faces.length; i++) {
            indices[i * 3] = faces[i][0];
            indices[i * 3 + 1] = faces[i][1];
            indices[i * 3 + 2] = faces[i][2];
        }
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));

        // Compute normals for proper lighting
        geometry.computeVertexNormals();

        // Create mesh material
        const color = colors[index % colors.length];
        const material = new THREE.MeshPhongMaterial({
            color: color,
            transparent: true,
            opacity: 0.7,
            side: THREE.DoubleSide,
        });

        // Create mesh object
        const mesh = new THREE.Mesh(geometry, material);
        mesh.userData = { slotId, index };

        // Add to scene
        scenes.unified.add(mesh);
        objects.slots.push(mesh);
    });
}

/**
 * Render prototype visualizations in individual 3D views - simplified version
 */
function renderPrototypes(prototypesData) {
    console.log("Rendering prototypes with data:", prototypesData);

    // Initialize or clear global arrays - avoid relying on existing state
    window.prototypeScenes = window.prototypeScenes || [];
    window.prototypeCameras = window.prototypeCameras || [];
    window.prototypeControls = window.prototypeControls || [];
    window.prototypeRenderers = window.prototypeRenderers || [];
    window.prototypeMeshes = window.prototypeMeshes || [];

    // Clear DOM elements
    if (elements.prototypesGrid) {
        elements.prototypesGrid.innerHTML = "";
    } else {
        console.error("Prototype grid element not found");
        return;
    }

    // Clean up existing renderers and their DOM elements
    window.prototypeRenderers.forEach(renderer => {
        if (renderer && renderer.domElement && renderer.domElement.parentNode) {
            renderer.domElement.parentNode.removeChild(renderer.domElement);
            renderer.dispose();
        }
    });

    // Clear arrays
    window.prototypeScenes = [];
    window.prototypeCameras = [];
    window.prototypeControls = [];
    window.prototypeRenderers = [];
    window.prototypeMeshes = [];

    // Also clear existing prototypes in the original objects array
    if (objects.prototypes) {
        while (objects.prototypes.length > 0) {
            const prototype = objects.prototypes.pop();
            if (prototype && prototype.parent) {
                prototype.parent.remove(prototype);
            }
        }
    } else {
        // If objects.prototypes doesn't exist, create it
        objects.prototypes = [];
    }

    // Check if we have valid data
    if (!prototypesData || !prototypesData.offsets || prototypesData.offsets.length === 0) {
        console.warn('No prototype data available');
        return;
    }

    const numPrototypes = prototypesData.num_prototypes;
    console.log(`Creating ${numPrototypes} prototype views`);

    // Base geometry for all prototypes
    const baseSphereGeometry = new THREE.IcosahedronGeometry(0.2, 4);

    // Create a grid of prototype views
    for (let i = 0; i < numPrototypes; i++) {
        // Create container elements
        const container = document.createElement('div');
        container.className = 'card bg-base-100 shadow-xl';
        elements.prototypesGrid.appendChild(container);

        const cardBody = document.createElement('div');
        cardBody.className = 'card-body p-2';
        container.appendChild(cardBody);

        const title = document.createElement('h3');
        title.className = 'card-title text-sm justify-center';
        title.textContent = `Prototype ${i + 1}`;
        cardBody.appendChild(title);

        const viewContainer = document.createElement('div');
        viewContainer.className = 'panel-3d aspect-square';
        viewContainer.style.height = '150px';
        cardBody.appendChild(viewContainer);

        // Create the THREE.js components for this view
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x15191E);
        window.prototypeScenes.push(scene);

        const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        camera.position.set(0, 0, 1.2);
        window.prototypeCameras.push(camera);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(viewContainer.clientWidth, viewContainer.clientHeight);
        viewContainer.appendChild(renderer.domElement);
        window.prototypeRenderers.push(renderer);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;
        window.prototypeControls.push(controls);

        // Add lighting
        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);

        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight1.position.set(1, 1, 1);
        scene.add(directionalLight1);

        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        directionalLight2.position.set(-1, -1, -1);
        scene.add(directionalLight2);

        // Add axes helper
        const axes = new THREE.AxesHelper(0.3);
        scene.add(axes);

        // Create the mesh for this prototype
        const protoGeometry = baseSphereGeometry.clone();
        const vertices = protoGeometry.attributes.position.array;

        // Apply offsets if available
        if (prototypesData.offsets && prototypesData.offsets[i]) {
            const protoOffsets = prototypesData.offsets[i];
            for (let v = 0; v < vertices.length / 3; v++) {
                if (v < protoOffsets.length) {
                    vertices[v * 3] += protoOffsets[v][0];
                    vertices[v * 3 + 1] += protoOffsets[v][1];
                    vertices[v * 3 + 2] += protoOffsets[v][2];
                }
            }
        }

        // Update geometry and compute normals
        protoGeometry.attributes.position.needsUpdate = true;
        protoGeometry.computeVertexNormals();

        // Create material and mesh
        const color = colors[i % colors.length];
        const material = new THREE.MeshPhongMaterial({
            color,
            wireframe: false,
            side: THREE.DoubleSide
        });

        const mesh = new THREE.Mesh(protoGeometry, material);
        mesh.userData = { protoIndex: i };
        scene.add(mesh);

        // Store in our arrays
        window.prototypeMeshes.push(mesh);
        objects.prototypes.push(mesh); // For compatibility

        // Initial render
        renderer.render(scene, camera);
    }

    // Setup animation
    if (!window.simplifiedAnimationActive) {
        window.simplifiedAnimationActive = true;
        animateSimplified();
    }
}

/**
 * Simplified animation loop for prototype views
 */
function animateSimplified() {
    requestAnimationFrame(animateSimplified);

    // Only animate if we have renderers
    if (!window.prototypeRenderers || window.prototypeRenderers.length === 0) {
        return;
    }

    // Update each view
    for (let i = 0; i < window.prototypeRenderers.length; i++) {
        if (window.prototypeControls[i] && window.prototypeRenderers[i] &&
            window.prototypeScenes[i] && window.prototypeCameras[i]) {
            window.prototypeControls[i].update();
            window.prototypeRenderers[i].render(
                window.prototypeScenes[i],
                window.prototypeCameras[i]
            );
        }
    }
}

/**
 * Add lighting to a prototype scene
 */
function addLightingToScene(scene) {
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);

    // Add directional lights from multiple angles for good lighting
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
 * Animation loop for prototype views
 */
function animatePrototypes() {
    requestAnimationFrame(animatePrototypes);

    // Update controls and render each prototype view
    for (let i = 0; i < renderers.prototypeRenderers.length; i++) {
        if (controls.prototypeControls[i] && renderers.prototypeRenderers[i]) {
            controls.prototypeControls[i].update();
            renderers.prototypeRenderers[i].render(scenes.prototypeScenes[i], cameras.prototypeCameras[i]);
        }
    }
}
