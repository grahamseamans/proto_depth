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
 * Render all prototypes in a single unified 3D scene with navigation controls
 */
function renderPrototypes(prototypesData) {
    console.log("Rendering prototypes with unified approach:", prototypesData);

    // Access new DOM elements
    const prototypeNavigation = document.getElementById('prototype-navigation');
    const prototypeViewport = document.getElementById('prototype-viewport');

    if (!prototypeNavigation || !prototypeViewport) {
        console.error("Prototype navigation or viewport elements not found");
        return;
    }

    // Store reference to navigation element for later use
    elements.prototypeNavigation = prototypeNavigation;

    // Clear navigation container
    prototypeNavigation.innerHTML = "";

    // Clear viewport container
    prototypeViewport.innerHTML = "";

    // Clear existing prototypes from the array and scene
    if (objects.prototypes) {
        while (objects.prototypes.length > 0) {
            const prototype = objects.prototypes.pop();
            if (prototype && prototype.parent) {
                prototype.parent.remove(prototype);
            }
        }
    } else {
        objects.prototypes = [];
    }

    // Check if we have valid data
    if (!prototypesData || !prototypesData.offsets || prototypesData.offsets.length === 0) {
        console.warn('No prototype data available');
        prototypeViewport.innerHTML = '<div class="flex items-center justify-center h-full">No prototype data available</div>';
        return;
    }

    const numPrototypes = prototypesData.num_prototypes;
    console.log(`Creating ${numPrototypes} prototypes in unified scene`);

    // Create a new scene specifically for prototypes
    if (!scenes.prototypesUnified) {
        scenes.prototypesUnified = new THREE.Scene();
        scenes.prototypesUnified.background = new THREE.Color(0x15191E);

        // Add lighting to the scene
        const ambientLight = new THREE.AmbientLight(0x404040);
        scenes.prototypesUnified.add(ambientLight);

        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight1.position.set(1, 1, 1);
        scenes.prototypesUnified.add(directionalLight1);

        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        directionalLight2.position.set(-1, -1, -1);
        scenes.prototypesUnified.add(directionalLight2);

        // Add grid helper for better spatial understanding
        const gridHelper = new THREE.GridHelper(10, 10, 0x888888, 0x444444);
        scenes.prototypesUnified.add(gridHelper);

        // Add axes helper
        const axesHelper = new THREE.AxesHelper(1);
        scenes.prototypesUnified.add(axesHelper);
    } else {
        // Clear existing objects except lights and helpers
        const toRemove = [];
        scenes.prototypesUnified.traverse(object => {
            if (object.isMesh) {
                toRemove.push(object);
            }
        });

        toRemove.forEach(object => {
            scenes.prototypesUnified.remove(object);
        });
    }

    // Create camera if it doesn't exist
    if (!cameras.prototypesUnified) {
        cameras.prototypesUnified = new THREE.PerspectiveCamera(75, 1, 0.1, 100);
        cameras.prototypesUnified.position.set(0, 0, 5);
    }

    // Create renderer if it doesn't exist
    if (!renderers.prototypesUnified) {
        renderers.prototypesUnified = new THREE.WebGLRenderer({ antialias: true });
        renderers.prototypesUnified.setPixelRatio(window.devicePixelRatio);
    }

    // Create orbit controls if they don't exist
    if (!controls.prototypesUnified) {
        controls.prototypesUnified = new THREE.OrbitControls(cameras.prototypesUnified, renderers.prototypesUnified.domElement);
        controls.prototypesUnified.enableDamping = true;
        controls.prototypesUnified.dampingFactor = 0.25;
    }

    // Attach renderer to the viewport
    prototypeViewport.appendChild(renderers.prototypesUnified.domElement);

    // Make renderer canvas fill parent container with 100% width and height
    renderers.prototypesUnified.domElement.style.display = 'block';
    renderers.prototypesUnified.domElement.style.width = '100%';
    renderers.prototypesUnified.domElement.style.height = '100%';

    // Update renderer size to match container dimensions
    const rect = prototypeViewport.getBoundingClientRect();
    renderers.prototypesUnified.setSize(rect.width, rect.height);

    // Update camera aspect ratio
    cameras.prototypesUnified.aspect = rect.width / rect.height;
    cameras.prototypesUnified.updateProjectionMatrix();

    // Calculate grid layout dimensions based on number of prototypes
    const gridSize = Math.ceil(Math.sqrt(numPrototypes));
    const spacing = 1.0; // Space between prototypes

    // Base geometry for all prototypes
    const baseSphereGeometry = new THREE.IcosahedronGeometry(0.2, 4);

    // Create navigation UI
    createPrototypeNavigationUI(numPrototypes, gridSize, spacing);

    // Create prototypes in a grid layout
    for (let i = 0; i < numPrototypes; i++) {
        // Calculate grid position
        const row = Math.floor(i / gridSize);
        const col = i % gridSize;

        // Calculate 3D position in a grid
        const x = (col - gridSize / 2) * spacing;
        const y = (gridSize / 2 - row) * spacing; // Invert Y to match standard grid layout
        const z = 0;

        // Create prototype geometry
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
        mesh.position.set(x, y, z);
        mesh.userData = {
            protoIndex: i,
            gridPosition: { row, col }
        };

        // Create label for the prototype
        addPrototypeLabel(mesh, `P${i + 1}`, x, y, z);

        // Add to scene
        scenes.prototypesUnified.add(mesh);
        objects.prototypes.push(mesh);
    }

    // Set up animation loop if not already active
    if (!window.unifiedAnimationActive) {
        window.unifiedAnimationActive = true;
        animateUnifiedScene();
    }

    // Add resize handler if not already added
    if (!window.unifiedResizeHandlerAdded) {
        window.addEventListener('resize', function () {
            if (renderers.prototypesUnified && renderers.prototypesUnified.domElement) {
                const parent = renderers.prototypesUnified.domElement.parentElement;
                if (parent) {
                    const width = parent.clientWidth;
                    const height = parent.clientHeight;
                    renderers.prototypesUnified.setSize(width, height);
                    cameras.prototypesUnified.aspect = width / height;
                    cameras.prototypesUnified.updateProjectionMatrix();
                }
            }
        });
        window.unifiedResizeHandlerAdded = true;
    }

    // Initial render
    renderers.prototypesUnified.render(scenes.prototypesUnified, cameras.prototypesUnified);
}

/**
 * Create a simple prototype label using a small sprite
 */
function addPrototypeLabel(prototype, text, x, y, z) {
    // Create a canvas for the label
    const canvas = document.createElement('canvas');
    const size = 128;
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext('2d');

    // Draw background
    context.fillStyle = 'rgba(0, 0, 0, 0.7)';
    context.fillRect(0, 0, size, size);

    // Draw text
    context.font = '40px Arial';
    context.fillStyle = 'white';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, size / 2, size / 2);

    // Create texture from canvas
    const texture = new THREE.CanvasTexture(canvas);

    // Create sprite material and sprite
    const spriteMaterial = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        opacity: 0.8
    });

    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(0.5, 0.5, 1);
    sprite.position.set(0, 0.4, 0); // Position above the prototype

    // Add sprite to prototype
    prototype.add(sprite);
}

/**
 * Create navigation UI for prototypes
 */
function createPrototypeNavigationUI(numPrototypes, gridSize, spacing) {
    // Navigation container is now created in renderPrototypes
    if (!elements.prototypeNavigation) {
        console.error("Navigation container not found");
        return;
    }

    // Clear existing content
    elements.prototypeNavigation.innerHTML = '';

    // Add overview button - keep Daisy classes but without sizing
    const overviewBtn = document.createElement('button');
    overviewBtn.textContent = 'Overview';
    overviewBtn.className = 'btn btn-sm btn-primary';
    overviewBtn.style.marginBottom = '10px';
    overviewBtn.style.width = '100%';
    overviewBtn.onclick = () => {
        // Reset camera to show all prototypes
        cameras.prototypesUnified.position.set(0, 0, gridSize * spacing * 1.5);
        controls.prototypesUnified.target.set(0, 0, 0);
        controls.prototypesUnified.update();

        // Remove highlight from all buttons
        const buttons = elements.prototypeNavigation.querySelectorAll('button');
        for (let i = 1; i < buttons.length; i++) {
            buttons[i].classList.remove('ring');
        }
    };
    elements.prototypeNavigation.appendChild(overviewBtn);

    // Add simple buttons for each prototype directly in the cell
    for (let i = 0; i < numPrototypes; i++) {
        const button = document.createElement('button');
        button.textContent = `P${i + 1}`;
        button.className = 'btn btn-sm';
        button.style.backgroundColor = `#${colors[i % colors.length].toString(16).padStart(6, '0')}`;
        button.style.color = 'white';
        button.style.marginBottom = '5px';
        button.style.width = '100%';
        button.style.textAlign = 'left';

        // Simple click handler
        button.onclick = () => {
            focusOnPrototype(i, gridSize, spacing);
        };

        elements.prototypeNavigation.appendChild(button);

        // Add a line break for cleaner layout
        if (i < numPrototypes - 1) {
            elements.prototypeNavigation.appendChild(document.createElement('br'));
        }
    }
}

/**
 * Focus camera on a specific prototype
 */
function focusOnPrototype(index, gridSize, spacing) {
    const prototype = objects.prototypes[index];
    if (!prototype) return;

    // Highlight the selected button - buttons are now direct children of navigation container
    // The first button (index 0) is 'Overview', so prototype buttons start at index 1
    const buttons = elements.prototypeNavigation.querySelectorAll('button');
    for (let i = 0; i < buttons.length; i++) {
        if (i === index + 1) { // +1 to account for Overview button
            buttons[i].classList.add('ring');
        } else {
            buttons[i].classList.remove('ring');
        }
    }

    // Get prototype position
    const position = prototype.position.clone();

    // Animate camera to look at this prototype
    cameras.prototypesUnified.position.set(position.x, position.y, position.z + 1.2);
    controls.prototypesUnified.target.copy(position);
    controls.prototypesUnified.update();
}

/**
 * Animation loop for unified scene
 */
function animateUnifiedScene() {
    requestAnimationFrame(animateUnifiedScene);

    // Update controls
    if (controls.prototypesUnified) {
        controls.prototypesUnified.update();
    }

    // Render scene
    if (renderers.prototypesUnified && scenes.prototypesUnified && cameras.prototypesUnified) {
        renderers.prototypesUnified.render(scenes.prototypesUnified, cameras.prototypesUnified);
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
