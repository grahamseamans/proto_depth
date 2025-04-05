# Energy-Based Scene Optimization

This module implements the energy-based scene optimization approach as described. It uses a KD-tree to efficiently find the closest triangle (formed from prototypes and slots) to each point in the point cloud, computes an L2 squared distance loss, and backpropagates gradients through both the slot parameters and prototype data to iteratively refine the scene.

## Key Features

- Direct optimization of scene parameters (no neural network)
- KD-tree for efficient nearest neighbor search
- Backpropagation through both slot parameters and prototype vertices
- Visualization integration with existing visualization server

## Usage

To run the energy-based optimization:

```bash
# Run with default parameters
./run_ebm.sh

# Run with custom parameters
python run_energy_optimization.py --num_iterations 1000 --num_slots 64 --num_prototypes 10
```

## Command Line Arguments

- `--data_path`: Path to depth image data (default: data/SYNTHIA-SF/SEQ1/DepthDebugLeft)
- `--num_iterations`: Number of optimization iterations (default: 1000)
- `--num_slots`: Number of slots (objects) in the scene (default: 64)
- `--num_prototypes`: Number of prototype archetypes (default: 10)
- `--slot_lr`: Learning rate for slot parameters (default: 0.01)
- `--prototype_lr`: Learning rate for prototype parameters (default: 0.001)
- `--noise_std`: Standard deviation of noise for energy-based model (default: 0.0)
- `--viz_interval`: Interval for visualization updates (default: 50)
- `--image_index`: Index of the image to use (-1 for random) (default: -1)
- `--device`: Device to use (cuda/cpu, default: auto-detect)

## Technical Details

### Optimization Process

The optimization process follows these steps:

1. **Triangle Location Computation**:
   - Apply slot parameters (position, orientation, scale) to prototype vertices
   - Form triangles from transformed vertices

2. **KD-Tree Construction**:
   - Build a KD-tree on triangle centroids using SciPy's `cKDTree`

3. **Closest Triangle Assignment**:
   - Query point cloud against the KD-tree to find closest triangle for each point

4. **Loss Calculation**:
   - Compute L2 squared distance loss between points and their closest triangles

5. **Backpropagation**:
   - Backpropagate gradients through slot parameters and prototype vertices

6. **Update**:
   - Update parameters using Adam optimizer
   - Optionally add noise for energy-based stochasticity

7. **Iterate** until convergence or fixed number of steps

### Parameters

- **Slot Parameters**: Position (x,y,z), orientation (yaw,pitch,roll), and scale for each slot
- **Prototype Offsets**: Vertex offsets for each prototype archetype

### Visualization

The optimization progress can be visualized using the existing visualization server:

```bash
# First run the optimization
./run_ebm.sh

# Then run the visualization server
./run_viz_server.sh
```

## Implementation Notes

- The implementation uses PyTorch for automatic differentiation
- SciPy's `cKDTree` is used for efficient nearest neighbor search
- The non-differentiable KD-tree query is handled by re-establishing the graph connection

## Extension Ideas

1. Add support for point-to-triangle distance computation (currently uses centroid approximation)
2. Implement hierarchical KD-tree for better performance with large point clouds
3. Add regularization terms for more realistic shapes
4. Explore different noise schedules for better energy-based optimization
