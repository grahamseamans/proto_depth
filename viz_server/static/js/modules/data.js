/**
 * Data loading and run/epoch/frame management for the visualization app.
 */

/**
 * Load available runs from the server
 */
async function loadRuns() {
    try {
        showLoading(true);
        const response = await fetch('/api/runs');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const runs = await response.json();
        state.runs = runs;

        console.log("Available runs:", runs);  // Debug log to see available runs

        // Populate run selector
        elements.runSelector.innerHTML = '<option value="" disabled selected>Select a run...</option>';
        runs.forEach(run => {
            const option = document.createElement('option');
            option.value = run.id;
            option.textContent = `${run.id} (${run.timestamp})`;
            elements.runSelector.appendChild(option);
        });

        // If there's only one run, select it automatically
        if (runs.length === 1) {
            elements.runSelector.value = runs[0].id;
            await loadRun(runs[0].id);
        } else if (runs.length > 0) {
            // For debugging, log all runs
            console.log("Multiple runs found, please select one from the dropdown");
        } else {
            console.warn("No runs found. Check server data directory.");
        }
    } catch (error) {
        console.error('Error loading runs:', error);
        alert('Error loading runs. Check the console for details.');
    } finally {
        showLoading(false);
    }
}

/**
 * Load a specific run and its epochs
 */
async function loadRun(runId) {
    try {
        showLoading(true);
        const response = await fetch(`/api/run/${runId}/epochs`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const epochs = await response.json();
        state.currentRun = runId;
        state.epochs = epochs;

        // Update epoch slider
        elements.epochSlider.min = 0;
        elements.epochSlider.max = epochs.length - 1;
        elements.epochSlider.value = 0;
        elements.epochSlider.disabled = epochs.length === 0;

        // Load first epoch if available
        if (epochs.length > 0) {
            state.currentEpochIndex = 0;
            await loadEpoch(epochs[0]);
        }
    } catch (error) {
        console.error('Error loading run:', error);
        alert('Error loading run. Check the console for details.');
    } finally {
        showLoading(false);
    }
}

/**
 * Load a specific epoch and its first batch
 */
async function loadEpoch(epoch) {
    try {
        showLoading(true);
        state.currentEpoch = epoch;

        // Update epoch display
        const totalEpochs = state.epochs.length;
        elements.epochDisplay.textContent = `Epoch: ${epoch.number}/${totalEpochs}`;

        // Load first batch if available
        if (epoch.batches && epoch.batches.length > 0) {
            const batch = epoch.batches[0];
            await loadBatch(state.currentRun, epoch.id, batch.id);
        } else {
            console.warn('No batches found for this epoch');
        }
    } catch (error) {
        console.error('Error loading epoch:', error);
        alert('Error loading epoch. Check the console for details.');
    } finally {
        showLoading(false);
    }
}

/**
 * Load a specific batch and render its data
 */
async function loadBatch(runId, epochId, batchId) {
    try {
        showLoading(true);

        // Fetch batch data
        const response = await fetch(`/api/run/${runId}/epoch/${epochId}/batch/${batchId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const batchData = await response.json();
        state.batchData = batchData;
        state.currentBatch = batchId;

        // No need to pre-load the depth image - it will be loaded when the modal is opened

        // Render 3D visualizations
        // Using globally accessible functions (no imports/exports)
        renderPointCloud(batchData.point_cloud);
        renderSlots(batchData.slots);
        renderPrototypes(batchData.prototypes);
        createWeightCharts(batchData);

        // Update slot toggle buttons
        updateSlotToggles(batchData.slots);
    } catch (error) {
        console.error('Error loading batch:', error);
        alert('Error loading batch. Check the console for details.');
    } finally {
        showLoading(false);
    }
}
