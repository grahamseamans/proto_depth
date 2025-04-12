/**
 * 3D visualization functions for point clouds, slots, and prototypes
 */

/**
 * Render the point cloud from batch data in the unified scene
 */
function transformToThreeSpace(points) {
    // Convert from camera space (-Z forward) to Three.js space (+Z forward)
    return points.map(point => [point[0], point[1], -point[2]]);
}

function renderTimeVaryingScene(frameDataArray, showAllFrames) {
    console.log('Rendering scene with data:', frameDataArray);

    // Clear existing objects
    if (objects.pointClouds) {
        objects.pointClouds.forEach(cloud => {
            if (cloud.parent) cloud.parent.remove(cloud);
        });
    }
    if (objects.meshes) {
        objects.meshes.forEach(mesh => {
            if (mesh.parent) mesh.parent.remove(mesh);
        });
    }
    objects.pointClouds = [];
    objects.meshes = [];

    // Create materials
    console.log('Creating materials with colors - true: 0x00ff00 (green), predicted: 0x0000ff (blue)');
    const trueMaterial = new THREE.MeshPhongMaterial({
        color: 0x00ff00,
        transparent: true,
        opacity: showAllFrames ? 0.3 : 0.7,
        side: THREE.DoubleSide
    });
    const predMaterial = new THREE.MeshPhongMaterial({
        color: 0x0000ff,
        transparent: true,
        opacity: showAllFrames ? 0.3 : 0.7,
        side: THREE.DoubleSide
    });
    console.log('Materials created:', {
        trueMaterial: trueMaterial.color.getHexString(),
        predMaterial: predMaterial.color.getHexString()
    });

    // Create point cloud materials
    const truePointsMaterial = new THREE.PointsMaterial({
        color: 0xff0000,
        size: 0.02,
        transparent: true,
        opacity: showAllFrames ? 0.3 : 0.7
    });
    const predPointsMaterial = new THREE.PointsMaterial({
        color: 0x0000ff,
        size: 0.02,
        transparent: true,
        opacity: showAllFrames ? 0.3 : 0.7
    });

    // Process each frame
    frameDataArray.forEach(frameData => {
        console.log('Processing frame:', frameData);

        // Add point clouds
        if (frameData.point_clouds) {
            frameData.point_clouds.forEach(cloud => {
                console.log('Adding point cloud:', cloud.is_true ? 'true' : 'predicted');
                // Filter out any points with NaN values
                const validPoints = cloud.points.filter(point =>
                    point.every(coord => typeof coord === 'number' && !isNaN(coord) && isFinite(coord))
                );
                if (validPoints.length === 0) {
                    console.warn('No valid points in cloud');
                    return;
                }

                // Transform points to Three.js coordinate system
                const worldPoints = transformToThreeSpace(validPoints);

                const geometry = new THREE.BufferGeometry();
                const vertices = new Float32Array(worldPoints.flat());
                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                const material = cloud.is_true ? truePointsMaterial : predPointsMaterial;
                const pointCloud = new THREE.Points(geometry, material);
                scenes.unified.add(pointCloud);
                objects.pointClouds.push(pointCloud);
            });
        }

        // Add objects
        if (frameData.objects) {
            const bunnyGeometry = new THREE.SphereGeometry(0.5, 32, 32);  // Made bigger
            frameData.objects.forEach(obj => {
                console.log('Adding object:', obj.is_true ? 'true' : 'predicted');
                // Validate transform
                if (!obj.position.every(val => typeof val === 'number' && !isNaN(val) && isFinite(val)) ||
                    !obj.rotation.every(val => typeof val === 'number' && !isNaN(val) && isFinite(val)) ||
                    !obj.scale.every(val => typeof val === 'number' && !isNaN(val) && isFinite(val))) {
                    console.warn('Invalid transform:', obj);
                    return;
                }

                const material = obj.is_true ? trueMaterial : predMaterial;
                console.log('Using material:', obj.is_true ? 'true (green)' : 'predicted (blue)', 'for object:', obj);
                const mesh = new THREE.Mesh(bunnyGeometry, material);
                mesh.position.set(...obj.position);
                mesh.rotation.set(...obj.rotation);
                mesh.scale.set(...obj.scale);
                scenes.unified.add(mesh);
                objects.meshes.push(mesh);
            });
        }

        // Add camera frustums
        if (frameData.cameras) {
            frameData.cameras.forEach(cam => {
                console.log('Adding camera frustum:', cam.is_true ? 'true' : 'predicted');
                const frustum = createCameraFrustum(
                    new THREE.Vector3(...cam.position),
                    new THREE.Vector3(...cam.rotation),
                    0.2,  // Size of frustum
                    cam.is_true ? 0xffff00 : 0x00ffff  // Yellow for true, cyan for predicted
                );
                scenes.unified.add(frustum);
            });
        }
    });

    // Center camera
    const center = new THREE.Vector3(0, 0, 0);
    const distance = 3;
    cameras.unified.position.set(0, 1, -distance);  // Move camera to -Z
    cameras.unified.lookAt(center);
    controls.unified.target.copy(center);

    console.log('Scene rendering complete');
}

function createCameraFrustum(position, rotation, size, color) {
    const geometry = new THREE.BufferGeometry();

    // Create frustum vertices
    const vertices = new Float32Array([
        0, 0, 0,  // apex
        -size, -size, -size * 2,  // near plane corners
        size, -size, -size * 2,
        size, size, -size * 2,
        -size, size, -size * 2
    ]);

    // Create lines
    const indices = new Uint16Array([
        0, 1,  // lines from apex to corners
        0, 2,
        0, 3,
        0, 4,
        1, 2,  // lines around near plane
        2, 3,
        3, 4,
        4, 1
    ]);

    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));

    const material = new THREE.LineBasicMaterial({ color: color });
    const frustum = new THREE.LineSegments(geometry, material);

    // Position and rotate
    frustum.position.copy(position);
    frustum.rotation.setFromVector3(rotation);

    return frustum;
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
 * Render camera views with point clouds
 */
function renderCameraViews(cameraData) {
    console.log("Rendering camera views:", cameraData);

    // Access DOM elements
    const cameraNavigation = document.getElementById('camera-navigation');
    const cameraViewport = document.getElementById('camera-viewport');

    if (!cameraNavigation || !cameraViewport) {
        console.error("Camera navigation or viewport elements not found");
        return;
    }

    // Store reference to navigation element
    elements.cameraNavigation = cameraNavigation;

    // Clear navigation container
    cameraNavigation.innerHTML = "";

    // Clear viewport container
    cameraViewport.innerHTML = "";

    // Clear existing point clouds from the array and scene
    if (objects.cameraPointClouds) {
        while (objects.cameraPointClouds.length > 0) {
            const cloud = objects.cameraPointClouds.pop();
            if (cloud && cloud.parent) {
                cloud.parent.remove(cloud);
            }
        }
    } else {
        objects.cameraPointClouds = [];
    }

    // Check if we have valid data
    if (!cameraData || !cameraData.points || cameraData.points.length === 0) {
        console.warn('No camera view data available');
        cameraViewport.innerHTML = '<div class="flex items-center justify-center h-full">No camera view data available</div>';
        return;
    }

    const numViews = cameraData.points.length;
    console.log(`Creating ${numViews} camera views`);

    // Create a new scene specifically for camera views
    if (!scenes.camerasUnified) {
        scenes.camerasUnified = new THREE.Scene();
        scenes.camerasUnified.background = new THREE.Color(0x15191E);

        // Add lighting to the scene
        const ambientLight = new THREE.AmbientLight(0x404040);
        scenes.camerasUnified.add(ambientLight);

        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight1.position.set(1, 1, 1);
        scenes.camerasUnified.add(directionalLight1);

        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        directionalLight2.position.set(-1, -1, -1);
        scenes.camerasUnified.add(directionalLight2);

        // Add grid helper for better spatial understanding
        const gridHelper = new THREE.GridHelper(10, 10, 0x888888, 0x444444);
        scenes.camerasUnified.add(gridHelper);

        // Add axes helper
        const axesHelper = new THREE.AxesHelper(1);
        scenes.camerasUnified.add(axesHelper);
    } else {
        // Clear existing objects except lights and helpers
        const toRemove = [];
        scenes.camerasUnified.traverse(object => {
            if (object instanceof THREE.Points) {
                toRemove.push(object);
            }
        });

        toRemove.forEach(object => {
            scenes.camerasUnified.remove(object);
        });
    }

    // Create camera if it doesn't exist
    if (!cameras.camerasUnified) {
        cameras.camerasUnified = new THREE.PerspectiveCamera(75, 1, 0.1, 100);
        cameras.camerasUnified.position.set(0, 0, 5);
    }

    // Create renderer if it doesn't exist
    if (!renderers.camerasUnified) {
        renderers.camerasUnified = new THREE.WebGLRenderer({ antialias: true });
        renderers.camerasUnified.setPixelRatio(window.devicePixelRatio);
    }

    // Create orbit controls if they don't exist
    if (!controls.camerasUnified) {
        controls.camerasUnified = new THREE.OrbitControls(cameras.camerasUnified, renderers.camerasUnified.domElement);
        controls.camerasUnified.enableDamping = true;
        controls.camerasUnified.dampingFactor = 0.25;
    }

    // Attach renderer to the viewport
    cameraViewport.appendChild(renderers.camerasUnified.domElement);

    // Make renderer canvas fill parent container
    renderers.camerasUnified.domElement.style.display = 'block';
    renderers.camerasUnified.domElement.style.width = '100%';
    renderers.camerasUnified.domElement.style.height = '100%';

    // Update renderer size
    const rect = cameraViewport.getBoundingClientRect();
    renderers.camerasUnified.setSize(rect.width, rect.height);

    // Update camera aspect ratio
    cameras.camerasUnified.aspect = rect.width / rect.height;
    cameras.camerasUnified.updateProjectionMatrix();

    // Create navigation UI
    createCameraNavigationUI(numViews);

    // Create point clouds for each camera view
    for (let i = 0; i < numViews; i++) {
        const points = cameraData.points[i];

        // Create point cloud geometry
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array(points.length * 3);
        const colors = new Float32Array(points.length * 3);

        let minZ = Infinity;
        let maxZ = -Infinity;

        // Set vertices
        for (let j = 0; j < points.length; j++) {
            const p = points[j];
            vertices[j * 3] = p[0];
            vertices[j * 3 + 1] = p[1];
            vertices[j * 3 + 2] = p[2];
            minZ = Math.min(minZ, p[2]);
            maxZ = Math.max(maxZ, p[2]);
        }

        // Color points by depth
        const colorScale = (maxZ > minZ) ? (maxZ - minZ) : 1;
        for (let j = 0; j < points.length; j++) {
            const z = points[j][2];
            const t = (z - minZ) / colorScale;

            // Blue to green to red color gradient
            if (t < 0.5) {
                const u = t * 2;
                colors[j * 3] = 0;
                colors[j * 3 + 1] = u;
                colors[j * 3 + 2] = 1 - u;
            } else {
                const u = (t - 0.5) * 2;
                colors[j * 3] = u;
                colors[j * 3 + 1] = 1 - u;
                colors[j * 3 + 2] = 0;
            }
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        // Create point material
        const material = new THREE.PointsMaterial({
            size: 0.05,
            vertexColors: true
        });

        // Create point cloud object
        const pointCloud = new THREE.Points(geometry, material);
        pointCloud.visible = false; // Initially hidden
        pointCloud.userData = { cameraIndex: i };

        // Add to scene
        scenes.camerasUnified.add(pointCloud);
        objects.cameraPointClouds.push(pointCloud);
    }

    // Set up animation loop if not already active
    if (!window.camerasAnimationActive) {
        window.camerasAnimationActive = true;
        animateCameraScene();
    }

    // Add resize handler if not already added
    if (!window.camerasResizeHandlerAdded) {
        window.addEventListener('resize', function () {
            if (renderers.camerasUnified && renderers.camerasUnified.domElement) {
                const parent = renderers.camerasUnified.domElement.parentElement;
                if (parent) {
                    const width = parent.clientWidth;
                    const height = parent.clientHeight;
                    renderers.camerasUnified.setSize(width, height);
                    cameras.camerasUnified.aspect = width / height;
                    cameras.camerasUnified.updateProjectionMatrix();
                }
            }
        });
        window.camerasResizeHandlerAdded = true;
    }

    // Initial render
    renderers.camerasUnified.render(scenes.camerasUnified, cameras.camerasUnified);
}

/**
 * Create navigation UI for camera views
 */
function createCameraNavigationUI(numViews) {
    if (!elements.cameraNavigation) {
        console.error("Navigation container not found");
        return;
    }

    // Clear existing content
    elements.cameraNavigation.innerHTML = '';

    // Add overview button
    const overviewBtn = document.createElement('button');
    overviewBtn.textContent = 'Overview';
    overviewBtn.className = 'btn btn-sm btn-primary';
    overviewBtn.style.marginBottom = '10px';
    overviewBtn.style.width = '100%';
    overviewBtn.onclick = () => {
        // Reset camera to origin
        cameras.camerasUnified.position.set(0, 0, 5);
        controls.camerasUnified.target.set(0, 0, 0);
        controls.camerasUnified.update();

        // Hide all point clouds
        objects.cameraPointClouds.forEach(cloud => {
            cloud.visible = false;
        });

        // Remove highlight from all buttons
        const buttons = elements.cameraNavigation.querySelectorAll('button');
        for (let i = 1; i < buttons.length; i++) {
            buttons[i].classList.remove('ring');
        }
    };
    elements.cameraNavigation.appendChild(overviewBtn);

    // Add buttons for each camera view
    for (let i = 0; i < numViews; i++) {
        const button = document.createElement('button');
        button.textContent = `Camera ${i + 1}`;
        button.className = 'btn btn-sm';
        button.style.backgroundColor = `#${colors[i % colors.length].toString(16).padStart(6, '0')}`;
        button.style.color = 'white';
        button.style.marginBottom = '5px';
        button.style.width = '100%';
        button.style.textAlign = 'left';

        // Click handler
        button.onclick = () => {
            focusOnCameraView(i);
        };

        elements.cameraNavigation.appendChild(button);

        // Add line break
        if (i < numViews - 1) {
            elements.cameraNavigation.appendChild(document.createElement('br'));
        }
    }
}

/**
 * Focus on a specific camera view
 */
function focusOnCameraView(index) {
    const pointCloud = objects.cameraPointClouds[index];
    if (!pointCloud) return;

    // Hide all point clouds except the selected one
    objects.cameraPointClouds.forEach((cloud, i) => {
        cloud.visible = (i === index);
    });

    // Highlight the selected button
    const buttons = elements.cameraNavigation.querySelectorAll('button');
    for (let i = 0; i < buttons.length; i++) {
        if (i === index + 1) { // +1 to account for Overview button
            buttons[i].classList.add('ring');
        } else {
            buttons[i].classList.remove('ring');
        }
    }

    // Position camera at origin looking down -Z axis
    cameras.camerasUnified.position.set(0, 0, 0);
    controls.camerasUnified.target.set(0, 0, -1);
    controls.camerasUnified.update();
}

/**
 * Animation loop for camera views
 */
function animateCameraScene() {
    requestAnimationFrame(animateCameraScene);

    // Update controls
    if (controls.camerasUnified) {
        controls.camerasUnified.update();
    }

    // Render scene
    if (renderers.camerasUnified && scenes.camerasUnified && cameras.camerasUnified) {
        renderers.camerasUnified.render(scenes.camerasUnified, cameras.camerasUnified);
    }
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
