#!/bin/bash

# Configuration
REMOTE_HOST="root@ssha.jarvislabs.ai"
REMOTE_PORT="11614"  # Using the port from your initial message
REMOTE_DIR="/root/proto_depth"

# Step 1: Check if remote machine has rsync, install if needed
echo "Checking if rsync is installed on remote machine..."
RSYNC_INSTALLED=$(ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no $REMOTE_HOST "command -v rsync > /dev/null && echo 'yes' || echo 'no'")

if [ "$RSYNC_INSTALLED" = "no" ]; then
  echo "Installing rsync on remote machine..."
  ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no $REMOTE_HOST "apt-get update && apt-get install -y rsync"
fi

# Step 2: Create remote directory structure
echo "Creating remote directory structure..."
ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no $REMOTE_HOST "mkdir -p $REMOTE_DIR/data"

# Step 3: Sync code to remote machine using rsync
echo "Syncing code to remote machine..."
rsync -avz -e "ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no" \
  --exclude='data' --exclude='.git' --exclude='.venv' --exclude='.conda' \
  --exclude='__pycache__' --exclude='*.pyc' \
  ./ "$REMOTE_HOST:$REMOTE_DIR/"

# Step 4: Sync the small dataset to the remote machine using rsync
echo "Syncing small dataset to remote machine using rsync..."
ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no $REMOTE_HOST "mkdir -p $REMOTE_DIR/data/SYNTHIA-SF/SEQ1/DepthLeft"
rsync -avz -e "ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no" data/SYNTHIA-SF/SEQ1/DepthLeft/*.png "$REMOTE_HOST:$REMOTE_DIR/data/SYNTHIA-SF/SEQ1/DepthLeft/"

# Step 5: SSH into remote machine and run setup + training
echo "Setting up environment and starting training on remote machine..."
ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no $REMOTE_HOST << EOF
  cd $REMOTE_DIR
  
  # Make setup script executable
  chmod +x remote_setup.sh
  
  # Check the dataset structure
  echo "Checking dataset structure..."
  find data/SYNTHIA-SF -type d | sort
  
  # Count depth images
  echo "Found $(find data/SYNTHIA-SF/SEQ1/DepthLeft -name "*.png" | wc -l) depth images in data/SYNTHIA-SF/SEQ1/DepthLeft"
  
  # Set up conda environment
  echo "Setting up conda environment..."
  
  # Download and install Miniconda if not already installed
  if [ ! -d ".conda" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $REMOTE_DIR/.conda
    rm miniconda.sh
  fi
  
  # Add conda to path
  export PATH="$REMOTE_DIR/.conda/bin:$PATH"
  
  # Initialize conda for bash
  conda init bash
  source ~/.bashrc
  
    # Create a new conda environment with Python 3.9
    conda create -y -n pytorch3d python=3.9
    conda activate pytorch3d
    
    # Install PyTorch 1.13.0 with CUDA 11.7 (compatible with PyTorch3D)
    conda install -y pytorch=1.13.0 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
    
    # Install additional dependencies with specific versions
    conda install -y matplotlib
    conda install -y numpy=1.24.3  # Downgrade NumPy to avoid compatibility issues
    
    # Install PyTorch3D from conda-forge
    conda install -y pytorch3d -c pytorch3d -c conda-forge
    
    # If conda install fails, try installing from source
    if [ $? -ne 0 ]; then
      echo "Conda install of PyTorch3D failed, trying to install from source..."
      pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    fi
  
  echo "Conda environment setup complete"
  
  # Make sure train_model.py has the correct data path
  sed -i "s|data_path\": \"data/SYNTHIA-SF/SEQ1/DepthDebugLeft\"|data_path\": \"data/SYNTHIA-SF/SEQ1/DepthLeft\"|g" train_model.py
  
  # Run training with the Python from the conda environment
  conda run -n pytorch3d python train_model.py
EOF

echo "Script completed!"
