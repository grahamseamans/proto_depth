#!/bin/bash

# Define SSH connection details - matching your existing scripts
REMOTE_HOST="root@sshg.jarvislabs.ai"
REMOTE_PORT="11014"
REMOTE_DIR="/root/proto_depth"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "Preparing to build CUDA extension on $REMOTE_HOST (port $REMOTE_PORT)"

# Sync the entire project to ensure all dependencies are available
echo "Syncing code to remote machine..."
rsync -avz -e "ssh -p $REMOTE_PORT $SSH_OPTS" \
  --exclude='data' --exclude='.git' --exclude='.venv' --exclude='.conda' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='progress_images' \
  ./ "$REMOTE_HOST:$REMOTE_DIR/"

# Build the CUDA extension on the remote machine
echo "Building CUDA extension on remote machine..."
ssh -p $REMOTE_PORT $SSH_OPTS $REMOTE_HOST << EOF
  cd $REMOTE_DIR
  
  # Print CUDA info
  python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
  python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
  
  # Make sure we have the necessary build tools
  pip install setuptools ninja
  
  # Go to the spatial hash directory and build the extension
  cd geometry_utils/spatial_hash
  python setup.py build_ext --inplace
  
  # Verify the build
  ls -la *.so
  
  # Try importing (will raise an error if build failed)
  cd ../..
  python -c "from geometry_utils.spatial_hash import create_spatial_hash; print('Spatial hash extension built successfully!')"
EOF

# Sync built extension back to local machine
echo "Syncing built extension back to local machine..."
rsync -avz -e "ssh -p $REMOTE_PORT $SSH_OPTS" \
  "$REMOTE_HOST:$REMOTE_DIR/geometry_utils/spatial_hash/*.so" \
  "./geometry_utils/spatial_hash/"

echo "CUDA extension build completed!"
echo "You can now run the optimization with: ./run_on_jarvis.sh"
