#!/bin/bash

# Extract SSH connection details from runpod_address.txt
CONNECTION_STRING=$(cat runpod_address.txt)
# Parse port and host
if [[ $CONNECTION_STRING =~ root@([^[:space:]]+)[[:space:]]-p[[:space:]]*([0-9]+)[[:space:]]-i[[:space:]]*([^[:space:]]+) ]]; then
    REMOTE_HOST="${BASH_REMATCH[1]}"
    REMOTE_PORT="${BASH_REMATCH[2]}"
    SSH_KEY="${BASH_REMATCH[3]}"
else
    echo "Error: Could not parse connection string from runpod_address.txt"
    exit 1
fi
REMOTE_DIR="/root/proto_depth"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "Preparing to build CUDA extension on $REMOTE_HOST (port $REMOTE_PORT)"

# Sync the entire project to ensure all dependencies are available
echo "Syncing code to remote machine..."
rsync -avz -e "ssh -p $REMOTE_PORT $SSH_OPTS -i $SSH_KEY" \
  --exclude='data' --exclude='.git' --exclude='.venv' --exclude='.conda' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='progress_images' \
  ./ "$REMOTE_HOST:$REMOTE_DIR/"

# Build the CUDA extension on the remote machine
echo "Building CUDA extension on remote machine..."
ssh -p $REMOTE_PORT $SSH_OPTS -i $SSH_KEY $REMOTE_HOST << EOF
  cd $REMOTE_DIR
  
  # Print CUDA info
  python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
  python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
  
  # Install Kaolin for PyTorch 2.4.0 and CUDA 124
  echo "Installing Kaolin for PyTorch 2.4.0 and CUDA 124..."
  pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu124.html
  python -c "import kaolin; print(f'Kaolin version: {kaolin.__version__}')"
  
  # Make sure we have the necessary build tools
  pip install setuptools ninja
  
  # Go to the spatial hash directory and build the extension
  cd geometry_utils/spatial_hash
  python setup.py build_ext --inplace
  
  # Verify the build
  ls -la *.so
  
  # Try importing (will raise an error if build failed)
  cd ../..
  python -c "from geometry_utils.spatial_hash import HierarchicalGrid; print('Hierarchical grid extension built successfully!')"
EOF

# Sync built extension back to local machine
echo "Syncing built extension back to local machine..."
rsync -avz -e "ssh -p $REMOTE_PORT $SSH_OPTS -i $SSH_KEY" \
  "$REMOTE_HOST:$REMOTE_DIR/geometry_utils/spatial_hash/*.so" \
  "./geometry_utils/spatial_hash/"

echo "CUDA extension build completed!"
echo "You can now run the optimization with: ./run_on_runpod.sh"
