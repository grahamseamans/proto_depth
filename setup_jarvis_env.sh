#!/bin/bash

# Extract SSH connection details from instance_address.txt
CONNECTION_STRING=$(cat instance_address.txt)
# Parse port and host
if [[ $CONNECTION_STRING =~ -p[[:space:]]+([0-9]+)[[:space:]]+(.+)$ ]]; then
    REMOTE_PORT="${BASH_REMATCH[1]}"
    REMOTE_HOST="${BASH_REMATCH[2]}"
else
    echo "Error: Could not parse connection string from instance_address.txt"
    exit 1
fi
REMOTE_DIR="/root/proto_depth"

echo "Setting up environment on $REMOTE_HOST (port $REMOTE_PORT)"

# Create remote directory structure and ensure rsync is installed
ssh -o StrictHostKeyChecking=no -p $REMOTE_PORT $REMOTE_HOST << EOF
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
rsync -avz -e "ssh -o StrictHostKeyChecking=no -p $REMOTE_PORT" \
  --exclude='.git' --exclude='.venv' --exclude='.conda' \
  --exclude='data' --exclude='__pycache__' --exclude='*.pyc' \
  ./ "$REMOTE_HOST:$REMOTE_DIR/"

# Install dependencies and verify installations
ssh -o StrictHostKeyChecking=no -p $REMOTE_PORT $REMOTE_HOST << EOF
  cd $REMOTE_DIR
  
  # Install all dependencies
  echo "Installing dependencies..."
  pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
  pip install matplotlib
  
  # Install our package in development mode
  echo "Installing package in development mode..."
  pip install -e .
  
  # Verify all installations
  echo "Verifying installations..."
  python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
  python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
  python -c "import kaolin; print(f'Kaolin version: {kaolin.__version__}')"
  python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"
  python -c "import src; print('Package installed successfully')"
  
  echo "Environment setup completed successfully!"
EOF

echo "Remote environment setup completed"
echo "You can now run ./run_on_jarvis.sh to run the GPU-accelerated optimization"
