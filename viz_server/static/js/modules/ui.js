/**
 * UI controls and event handlers
 */

// Import validator
import { validateFrameData } from "./validator.js";

// Initialize UI elements
function initUI() {
    try {
        // Get references to all UI elements
        const elementIds = {
            'epoch-slider': 'epochSlider',
            'epoch-display': 'epochDisplay',
            'reset-view-btn': 'resetViewBtn',
            'run-selector': 'runSelector',
            'toggle-pointcloud': 'togglePointCloud',
            'toggle-cameras': 'toggleCameras',
            'toggle-objects': 'toggleObjects',
            'time-slider': 'timeSlider',
            'time-display': 'timeDisplay'
        };

        // Get elements and store in global elements object
        for (const [id, key] of Object.entries(elementIds)) {
            const element = document.getElementById(id);
            if (!element) {
                console.warn(`Element with id '${id}' not found`);
                continue;
            }
            elements[key] = element;
        }

        // Add event listeners only if elements exist
        const listeners = [
            { element: 'epochSlider', event: 'input', handler: onEpochSliderChange },
            { element: 'resetViewBtn', event: 'click', handler: onResetView },
            { element: 'runSelector', event: 'change', handler: onRunSelect },
            { element: 'togglePointCloud', event: 'change', handler: onTogglePointCloud },
            { element: 'toggleCameras', event: 'change', handler: onToggleCameras },
            { element: 'toggleObjects', event: 'change', handler: onToggleObjects },
            { element: 'timeSlider', event: 'input', handler: onTimeSliderChange }
        ];

        for (const { element, event, handler } of listeners) {
            if (elements[element]) {
                elements[element].addEventListener(event, handler);
            }
        }

        // Load available runs
        loadRuns();

        console.log('UI initialized successfully');
    } catch (error) {
        console.error('Error initializing UI:', error);
    }
}

// New event handlers for camera and object toggles
function onToggleCameras() {
    if (objects.cameras) {
        objects.cameras.forEach(camera => {
            camera.visible = elements.toggleCameras.checked;
        });
    }
}

function onToggleObjects() {
    if (objects.meshes) {
        objects.meshes.forEach(mesh => {
            mesh.visible = elements.toggleObjects.checked;
        });
    }
}

// Event Handlers
async function onEpochSliderChange() {
    const epoch = elements.epochSlider.value;
    elements.epochDisplay.textContent = `Epoch: ${epoch}/${elements.epochSlider.max}`;
    await loadEpochData(epoch);
}

function onResetView() {
    if (cameras.unified && controls.unified) {
        cameras.unified.position.set(0, 1, 3);
        controls.unified.target.set(0, 0, 0);
        controls.unified.update();
    }
}

async function onRunSelect() {
    const runId = elements.runSelector.value;
    if (!runId) return;

    await loadRunData(runId);
}

function onTogglePointCloud() {
    if (objects.pointClouds) {
        objects.pointClouds.forEach(cloud => {
            cloud.visible = elements.togglePointCloud.checked;
        });
    }
}

async function onTimeSliderChange() {
    const frame = elements.timeSlider.value;
    elements.timeDisplay.textContent = `Frame: ${frame}/${elements.timeSlider.max}`;
    await loadFrameData(frame);
}

function showErrorOverlay(errors, warnings) {
    const overlay = document.getElementById("error-overlay");
    if (!overlay) return;
    if ((!errors || errors.length === 0) && (!warnings || warnings.length === 0)) {
        overlay.classList.add("hidden");
        overlay.innerHTML = "";
        return;
    }
    let html = "";
    if (errors && errors.length > 0) {
        html += `<div class="font-bold mb-1">Errors:</div><ul class="mb-2">` +
            errors.map(e => `<li>• ${e}</li>`).join("") + "</ul>";
    }
    if (warnings && warnings.length > 0) {
        html += `<div class="font-bold mb-1">Warnings:</div><ul>` +
            warnings.map(w => `<li>• ${w}</li>`).join("") + "</ul>";
    }
    overlay.innerHTML = html;
    overlay.classList.remove("hidden");
}

// Data Loading Functions
async function loadRuns() {
    try {
        const response = await fetch('/api/runs');
        const runs = await response.json();

        elements.runSelector.innerHTML = '<option value="" disabled selected>Select a run...</option>';
        runs.forEach(run => {
            const option = document.createElement('option');
            option.value = run.id;
            option.textContent = `${run.id} (${new Date(run.timestamp * 1000).toLocaleString()})`;
            elements.runSelector.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading runs:', error);
    }
}

async function loadRunData(runId) {
    try {
        // First get run metadata for frame count
        const metadataResponse = await fetch(`/api/run/${runId}/run_metadata.json`);
        const metadata = await metadataResponse.json();
        console.log('Run metadata:', metadata);

        // Update time slider based on num_frames
        if (elements.timeSlider) {
            elements.timeSlider.max = metadata.num_frames - 1;
            elements.timeSlider.value = 0;
            elements.timeDisplay.textContent = `0/${metadata.num_frames - 1}`;
        }

        const response = await fetch(`/api/run/${runId}/epochs`);
        const iterations = await response.json();
        console.log('Iterations:', iterations);

        // Update epoch slider (now showing iterations)
        elements.epochSlider.max = iterations.length - 1;
        elements.epochSlider.value = 0;
        elements.epochSlider.disabled = false;  // Enable slider
        elements.epochDisplay.textContent = `Iteration: 0/${iterations.length - 1}`;

        // Load initial iteration
        await loadEpochData(0);
    } catch (error) {
        console.error('Error loading run data:', error);
    }
}

async function loadEpochData(iteration) {
    try {
        const runId = elements.runSelector.value;
        if (!runId) return;

        // Load just the current frame
        const frame = elements.timeSlider ? elements.timeSlider.value : 0;
        await loadFrameData(frame);
    } catch (error) {
        console.error('Error loading iteration data:', error);
    }
}

async function loadFrameData(frame) {
    try {
        const runId = elements.runSelector.value;
        const iteration = elements.epochSlider.value;
        if (!runId) return;

        console.log('Loading frame', frame, 'from iteration', iteration);
        const response = await fetch(`/api/run/${runId}/iter/iter_${String(iteration).padStart(4, '0')}/frame_${String(frame).padStart(4, '0')}.json`);
        const frameData = await response.json();
        console.log('Frame data:', frameData);

        // Validate frame data and show errors/warnings
        const { errors, warnings, summary } = validateFrameData(frameData);
        showErrorOverlay(errors, warnings);

        // Transform data into expected format
        const transformedData = {
            point_clouds: [],
            objects: [],
            cameras: []
        };

        // Add point clouds
        if (frameData.true && frameData.true.point_clouds) {
            frameData.true.point_clouds.forEach(points => {
                transformedData.point_clouds.push({
                    points: points,
                    is_true: true
                });
            });
        }
        if (frameData.pred && frameData.pred.point_clouds) {
            frameData.pred.point_clouds.forEach(points => {
                transformedData.point_clouds.push({
                    points: points,
                    is_true: false
                });
            });
        }

        // Add objects
        if (frameData.true && frameData.true.positions) {
            console.log('Adding true objects:', frameData.true.positions.length);
            frameData.true.positions.forEach((pos, i) => {
                const obj = {
                    position: pos,
                    rotation: frameData.true.rotations[i],
                    scale: frameData.true.scales[i],
                    is_true: true
                };
                console.log('True object:', obj);
                transformedData.objects.push(obj);
            });
        }
        if (frameData.pred && frameData.pred.positions) {
            console.log('Adding predicted objects:', frameData.pred.positions.length);
            frameData.pred.positions.forEach((pos, i) => {
                const obj = {
                    position: pos,
                    rotation: frameData.pred.rotations[i],
                    scale: frameData.pred.scales[i],
                    is_true: false
                };
                console.log('Predicted object:', obj);
                transformedData.objects.push(obj);
            });
        }

        // Add cameras - each camera appears twice, once for true and once for predicted
        if (frameData.cameras) {
            // True cameras
            frameData.cameras.positions.forEach((pos, i) => {
                transformedData.cameras.push({
                    position: pos,
                    rotation: frameData.cameras.rotations[i],
                    is_true: true
                });
            });
            // Predicted cameras (same positions for now)
            frameData.cameras.positions.forEach((pos, i) => {
                transformedData.cameras.push({
                    position: pos,
                    rotation: frameData.cameras.rotations[i],
                    is_true: false
                });
            });
        }

        console.log('Transformed data:', transformedData);
        renderTimeVaryingScene([transformedData], false);
    } catch (error) {
        console.error('Error loading frame data:', error);
        showErrorOverlay([error.message], []);
    }
}

export { initUI };
