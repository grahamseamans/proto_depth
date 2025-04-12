# Visualization Server

## Overview
A 3D visualization server for comparing ground truth and predicted camera poses, point clouds, and bunny models in a scene.

## Core Features
- Display of scene elements:
  - Ground truth camera frustums
  - Predicted camera frustums
  - Ground truth point clouds
  - Predicted point clouds (ground truth clouds transformed by predicted camera poses)
  - Ground truth bunnies in their actual scene positions
  - Predicted bunnies in their predicted scene positions
- Time-based visualization:
  - Scrubbing through timesteps for each iteration
  - Option to view all timesteps simultaneously
- Run selection and management
- Interactive 3D viewport controls

## Architecture

### Core Components

1. Data Ingest Layer
   - Handles loading and parsing of run data
   - Manages data format conversion for visualization
   - Simple error reporting via console for debugging

2. UI Management Layer
   - Handles user interactions and viewport controls
   - Manages view states and transitions
   - Controls visualization options and toggles

3. Manager (Central Controller)
   - Coordinates between Data and UI layers
   - Maintains application state
   - Dispatches visualization commands
   - Handles timestep and iteration management

4. Visualization Functions
   - Stateless rendering functions for each element type:
     - displayPointCloud(container, pointData, options)
     - displayFrustum(container, cameraData, options)
     - displayBunny(container, modelData, options)
   - Each function accepts:
     - THREE.js scene/container object
     - Element-specific data
     - Visualization options (color, opacity, visibility)

### Data Flow
1. User selects a run
2. Data ingest layer loads run data
3. Manager processes data and maintains state, skipping invalid elements
4. UI layer triggers visualization updates
5. Visualization functions render scene elements with debug warnings when needed
6. User can interact with timeline and toggles

## Implementation Notes
- Uses Three.js for 3D visualization
- Maintains clean separation between data, UI, and visualization logic
- Stateless visualization functions for maintainability
- Centralized state management through Manager component
- Simple console-based debugging for research visualization
