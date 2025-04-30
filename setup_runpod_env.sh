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

echo "Setting up environment on $REMOTE_HOST (port $REMOTE_PORT)"

# Show the command we're about to run
echo "Running command: ssh -t root@$REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY"

# Create remote directory structure and ensure rsync is installed
ssh -t root@$REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY << EOF
  # Check NVIDIA driver and GPU visibility FIRST
  nvidia-smi || { echo "nvidia-smi failed: NVIDIA driver or GPU not found"; exit 1; }

  mkdir -p $REMOTE_DIR/data
  
  # Install rsync if not already installed
  if ! command -v rsync &> /dev/null; then
    echo "Installing rsync..."
    apt-get update && apt-get install -y rsync
  else
    echo "rsync is already installed."
  fi
EOF

# Copy the project files to the remote machine
rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
  --exclude='.git' --exclude='.venv' --exclude='.conda' \
  --exclude='data' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='kaolin'  --exclude='Open3D' --exclude='nvdiffrast' \
  ./ "root@$REMOTE_HOST:$REMOTE_DIR/"

# Install dependencies and verify installations
ssh -t root@$REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY << EOF
  cd $REMOTE_DIR
  
  # Install dependencies
  echo "Installing dependencies..."
  pip install --ignore-installed kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu124.html
  pip install matplotlib
  pip install einops
  pip install ninja
  git clone https://github.com/NVlabs/nvdiffrast 
  cd nvdiffrast 
  pip install .
  cd ..

  # Check NVIDIA driver and GPU visibility
  nvidia-smi || echo "nvidia-smi failed: NVIDIA driver or GPU not found"

  # Verify torch and CUDA
  echo "Verifying installations..."
  python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
  python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
  
  echo "Environment setup completed successfully!"
EOF

echo "Remote environment setup completed"
echo "You can now run ./run_on_runpod.sh to run the GPU-accelerated optimization"
