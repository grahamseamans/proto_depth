/**
 * Visualization logic for rendering slot meshes in the unified scene.
 */
import { scenes, objects, colors } from "../../state.js";

export function renderSlots(slotsData) {
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
