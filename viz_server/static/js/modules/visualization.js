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
 * Render prototype visualizations
 */
function renderPrototypes(prototypesData) {
    // Clear existing prototypes
    for (const proto of objects.prototypes) {
        scenes.prototypes.remove(proto);
    }
    objects.prototypes = [];

    if (!prototypesData || !prototypesData.offsets || prototypesData.offsets.length === 0) {
        console.warn('No prototype data available');
        return;
    }

    // Create a grid layout for prototypes
    const numPrototypes = prototypesData.num_prototypes;
    const gridSize = Math.ceil(Math.sqrt(numPrototypes));
    const spacing = 1.5;

    // For now, just visualize prototype positions as spheres
    const sphereGeometry = new THREE.SphereGeometry(0.2, 32, 32);

    for (let i = 0; i < numPrototypes; i++) {
        const row = Math.floor(i / gridSize);
        const col = i % gridSize;

        const color = colors[i % colors.length];
        const material = new THREE.MeshPhongMaterial({ color });

        const sphere = new THREE.Mesh(sphereGeometry, material);
        sphere.position.set(
            (col - gridSize / 2) * spacing,
            (row - gridSize / 2) * spacing,
            0
        );

        // Add prototype number label
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, 128, 128);
        ctx.fillStyle = 'black';
        ctx.font = '80px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(i + 1, 64, 64);

        const texture = new THREE.CanvasTexture(canvas);
        const labelMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(labelMaterial);
        sprite.scale.set(0.5, 0.5, 0.5);
        sprite.position.set(0, 0, 0.3);

        sphere.add(sprite);
        sphere.userData = { protoIndex: i };

        scenes.prototypes.add(sphere);
        objects.prototypes.push(sphere);
    }

    // Position camera to view all prototypes
    const cameraDistance = gridSize * spacing;
    cameras.prototypes.position.set(0, 0, cameraDistance * 1.5);
    cameras.prototypes.lookAt(0, 0, 0);
    controls.prototypes.target.set(0, 0, 0);
}
