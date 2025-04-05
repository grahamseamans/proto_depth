#!/bin/bash

# Define SSH connection details
# Use direct SSH connection (based on previous testing)
REMOTE_HOST="root@sshg.jarvislabs.ai" 
REMOTE_PORT="11014"
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

# Verify the system already has PyTorch and Kaolin installed
ssh -o StrictHostKeyChecking=no -p $REMOTE_PORT $REMOTE_HOST << EOF
  cd $REMOTE_DIR
  
  # Verify torch and kaolin are available
  echo "Verifying system PyTorch and Kaolin installations..."
  python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
  python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
  python -c "import kaolin; print(f'Kaolin version: {kaolin.__version__}')"
  
  echo "Environment setup completed successfully!"
EOF

echo "Remote environment setup completed"
echo "You can now run ./run_on_jarvis.sh to run the GPU-accelerated optimization"
