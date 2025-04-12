import { elements, colors } from "../../state.js";
import { positionCamera } from "../core/camera.js";

/**
 * Navigation UI functionality
 */

/**
 * Create navigation container
 * @param {HTMLElement} parent - Parent element to attach navigation to
 * @returns {HTMLElement} Navigation container
 */
export function createNavigationContainer(parent) {
    const container = document.createElement('div');
    container.className = 'navigation-container';
    container.style.position = 'absolute';
    container.style.top = '10px';
    container.style.right = '10px';
    container.style.zIndex = '1000';
    parent.appendChild(container);
    return container;
}

/**
 * Create overview button
 * @param {Function} onClick - Click handler
 * @returns {HTMLButtonElement} Overview button
 */
export function createOverviewButton(onClick) {
    const button = document.createElement('button');
    button.textContent = 'Overview';
    button.className = 'btn btn-sm btn-primary';
    button.style.marginBottom = '10px';
    button.style.width = '100%';
    button.onclick = onClick;
    return button;
}

/**
 * Create prototype button
 * @param {number} index - Prototype index
 * @param {Function} onClick - Click handler
 * @returns {HTMLButtonElement} Prototype button
 */
export function createPrototypeButton(index, onClick) {
    const button = document.createElement('button');
    button.textContent = `P${index + 1}`;
    button.className = 'btn btn-sm';
    button.style.backgroundColor = `#${colors[index % colors.length].toString(16).padStart(6, '0')}`;
    button.style.color = 'white';
    button.style.marginBottom = '5px';
    button.style.width = '100%';
    button.style.textAlign = 'left';
    button.onclick = onClick;
    return button;
}

/**
 * Create navigation UI for prototypes
 * @param {number} numPrototypes - Number of prototypes
 * @param {Function} onOverviewClick - Overview button click handler
 * @param {Function} onPrototypeClick - Prototype button click handler
 */
export function createPrototypeNavigation(numPrototypes, onOverviewClick, onPrototypeClick) {
    if (!elements.prototypeNavigation) {
        console.error("Navigation container not found");
        return;
    }

    // Clear existing content
    elements.prototypeNavigation.innerHTML = '';

    // Add overview button
    const overviewBtn = createOverviewButton(onOverviewClick);
    elements.prototypeNavigation.appendChild(overviewBtn);

    // Add prototype buttons
    for (let i = 0; i < numPrototypes; i++) {
        const button = createPrototypeButton(i, () => onPrototypeClick(i));
        elements.prototypeNavigation.appendChild(button);

        if (i < numPrototypes - 1) {
            elements.prototypeNavigation.appendChild(document.createElement('br'));
        }
    }
}

/**
 * Update button highlights
 * @param {number} selectedIndex - Index of selected prototype
 */
export function updateButtonHighlights(selectedIndex) {
    const buttons = elements.prototypeNavigation.querySelectorAll('button');
    for (let i = 0; i < buttons.length; i++) {
        if (i === selectedIndex + 1) { // +1 to account for Overview button
            buttons[i].classList.add('ring');
        } else {
            buttons[i].classList.remove('ring');
        }
    }
}

/**
 * Reset camera to overview position
 * @param {string} cameraKey - Key of the camera to use
 * @param {string} controlsKey - Key of the controls to update
 * @param {number} gridSize - Grid size
 * @param {number} spacing - Grid spacing
 */
export function resetToOverview(cameraKey, controlsKey, gridSize, spacing) {
    positionCamera(
        cameraKey,
        controlsKey,
        new THREE.Vector3(0, 0, gridSize * spacing * 1.5),
        new THREE.Vector3(0, 0, 0)
    );
} 