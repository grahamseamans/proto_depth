/**
 * UI controls and event handlers
 */
export class UI {
    constructor(manager) {
        this.manager = manager;
        this.elements = {};
        this.initUI();
    }

    initUI() {
        try {
            // Get references to all UI elements
            const elementIds = {
                'epoch-slider': 'epochSlider',
                'epoch-display': 'epochDisplay',
                'reset-view-btn': 'resetViewBtn',
                'run-selector': 'runSelector',
                'toggle-pointcloud': 'togglePointCloud',
                'toggle-cameras': 'toggleCameras',
                'toggle-meshes': 'toggleMeshes',
                'time-slider': 'timeSlider',
                'time-display': 'timeDisplay',
                'play-btn': 'playBtn',
                'play-icon': 'playIcon',
                'pause-icon': 'pauseIcon',
                'speed-slider': 'speedSlider',
                'speed-display': 'speedDisplay'
            };

            // Get elements
            for (const [id, key] of Object.entries(elementIds)) {
                const element = document.getElementById(id);
                if (!element) {
                    console.warn(`Element with id '${id}' not found`);
                    continue;
                }
                this.elements[key] = element;
            }

            // Add event listeners
            this.setupEventListeners();

            // Playback state
            this.isPlaying = false;
            this.playInterval = null;
            this.playSpeed = 1; // multiplier, 1x by default

            // Play button event
            if (this.elements.playBtn) {
                this.elements.playBtn.addEventListener('click', () => this.togglePlay());
            }
            // Speed slider event
            if (this.elements.speedSlider) {
                this.elements.speedSlider.addEventListener('input', () => this.onSpeedChange());
                this.onSpeedChange(); // set initial display
            }

            // Theme toggle: sync 3D background with theme
            const themeToggle = document.getElementById('theme-toggle');
            if (themeToggle) {
                themeToggle.addEventListener('change', (e) => {
                    const newTheme = e.target.checked ? 'dark' : 'light';
                    this.manager.setBackgroundStyle(newTheme);
                });
            }
            // Also set initial background to match current theme
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
            this.manager.setBackgroundStyle(currentTheme);

            // Load available runs
            this.loadRuns();

            console.log('UI initialized successfully');
        } catch (error) {
            console.error('Error initializing UI:', error);
        }
    }

    setupEventListeners() {
        const listeners = [
            { element: 'epochSlider', event: 'input', handler: this.onEpochSliderChange.bind(this) },
            { element: 'resetViewBtn', event: 'click', handler: this.onResetView.bind(this) },
            { element: 'runSelector', event: 'change', handler: this.onRunSelect.bind(this) },
            { element: 'togglePointCloud', event: 'change', handler: this.onTogglePointCloud.bind(this) },
            { element: 'toggleCameras', event: 'change', handler: this.onToggleCameras.bind(this) },
            { element: 'toggleMeshes', event: 'change', handler: this.onToggleMeshes.bind(this) },
            { element: 'timeSlider', event: 'input', handler: this.onTimeSliderChange.bind(this) }
        ];

        for (const { element, event, handler } of listeners) {
            if (this.elements[element]) {
                this.elements[element].addEventListener(event, handler);
            }
        }
    }

    // --- Playback Logic ---

    togglePlay() {
        this.isPlaying = !this.isPlaying;
        // Update icons
        if (this.elements.playIcon && this.elements.pauseIcon) {
            if (this.isPlaying) {
                this.elements.playIcon.classList.add("hidden");
                this.elements.pauseIcon.classList.remove("hidden");
            } else {
                this.elements.playIcon.classList.remove("hidden");
                this.elements.pauseIcon.classList.add("hidden");
            }
        }
        if (this.isPlaying) {
            this.startPlayback();
        } else {
            this.stopPlayback();
        }
    }

    startPlayback() {
        if (this.playInterval) clearInterval(this.playInterval);
        // Calculate interval in ms: map speedSlider (1-10) to 1000ms (slow) to 50ms (fast)
        const speedValue = this.elements.speedSlider ? parseInt(this.elements.speedSlider.value) : 5;
        const minMs = 50, maxMs = 1000;
        const intervalMs = maxMs - ((speedValue - 1) / 9) * (maxMs - minMs);
        this.playInterval = setInterval(() => {
            let frame = parseInt(this.elements.timeSlider.value);
            const maxFrame = parseInt(this.elements.timeSlider.max);
            frame = (frame + 1) > maxFrame ? 0 : (frame + 1);
            this.elements.timeSlider.value = frame;
            this.elements.timeDisplay.textContent = `Frame: ${frame}/${maxFrame}`;
            this.loadFrame();
        }, intervalMs);
    }

    stopPlayback() {
        if (this.playInterval) {
            clearInterval(this.playInterval);
            this.playInterval = null;
        }
    }

    onSpeedChange() {
        const speedValue = this.elements.speedSlider ? parseInt(this.elements.speedSlider.value) : 5;
        // Map 1-10 to 0.1x to 2x (log scale for better control)
        const minX = 0.1, maxX = 2.0;
        const speed = minX + ((speedValue - 1) / 9) * (maxX - minX);
        this.playSpeed = speed;
        if (this.elements.speedDisplay) {
            this.elements.speedDisplay.textContent = `${speed.toFixed(1)}x`;
        }
        // If playing, restart interval with new speed
        if (this.isPlaying) {
            this.stopPlayback();
            this.startPlayback();
        }
    }

    // Event Handlers
    async onEpochSliderChange() {
        const epoch = parseInt(this.elements.epochSlider.value);
        this.elements.epochDisplay.textContent = `Epoch: ${epoch}/${this.elements.epochSlider.max}`;
        this.manager.currentIteration = epoch;
        await this.loadFrame();
    }

    onResetView() {
        this.manager.resetView();
    }

    async onRunSelect() {
        const runId = this.elements.runSelector.value;
        if (!runId) return;

        try {
            const { numIterations, numFrames } = await this.manager.loadRun(runId);

            // Update UI controls
            this.elements.epochSlider.max = numIterations - 1;
            this.elements.epochSlider.value = 0;
            this.elements.epochSlider.disabled = false;
            this.elements.epochDisplay.textContent = `Iteration: 0/${numIterations - 1}`;

            this.elements.timeSlider.max = numFrames - 1;
            this.elements.timeSlider.value = 0;
            this.elements.timeDisplay.textContent = `0/${numFrames - 1}`;
        } catch (error) {
            console.error('Error loading run:', error);
            this.showErrorOverlay([error.message], []);
        }
    }

    onTogglePointCloud() {
        this.manager.setVisibility('pointClouds', this.elements.togglePointCloud.checked);
    }

    onToggleCameras() {
        this.manager.setVisibility('frustums', this.elements.toggleCameras.checked);
    }

    onToggleMeshes() {
        this.manager.setVisibility('meshes', this.elements.toggleMeshes.checked);
    }

    async onTimeSliderChange() {
        // If user interacts with slider, pause playback
        if (this.isPlaying) {
            this.isPlaying = false;
            this.stopPlayback();
            if (this.elements.playIcon && this.elements.pauseIcon) {
                this.elements.playIcon.classList.remove("hidden");
                this.elements.pauseIcon.classList.add("hidden");
            }
        }
        const frame = this.elements.timeSlider.value;
        this.elements.timeDisplay.textContent = `Frame: ${frame}/${this.elements.timeSlider.max}`;
        await this.loadFrame();
    }

    // Data Loading
    async loadRuns() {
        try {
            const response = await fetch('/api/runs');
            const runs = await response.json();

            this.elements.runSelector.innerHTML = '<option value="" disabled selected>Select a run...</option>';
            runs.forEach(run => {
                const option = document.createElement('option');
                option.value = run.id;
                option.textContent = `${run.id} (${new Date(run.timestamp * 1000).toLocaleString()})`;
                this.elements.runSelector.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading runs:', error);
            this.showErrorOverlay([error.message], []);
        }
    }

    async loadFrame() {
        const frame = parseInt(this.elements.timeSlider.value);
        try {
            await this.manager.loadFrame(frame);
        } catch (error) {
            console.error('Error loading frame:', error);
            this.showErrorOverlay([error.message], []);
        }
    }

    // UI Helpers
    showErrorOverlay(errors, warnings) {
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
}
