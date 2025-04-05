#!/bin/bash

# Run energy-based optimization with GPU-accelerated nearest neighbor search using Kaolin
uv run run_energy_native.py --data_path data/SYNTHIA-SF/SEQ1/DepthLeft --num_iterations 10 --viz_interval 5 --image_index -1 --ico_level 4
