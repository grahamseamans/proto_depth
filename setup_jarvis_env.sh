#!/bin/bash

# Parse SSH connection details from instance_address.txt
SSH_STRING=$(cat instance_address.txt)
PORT=$(echo "$SSH_STRING" | sed -E 's/.*-p ([0-9]+).*/\1/')
USER_HOST=$(echo "$SSH_STRING" | awk '{print $NF}')
USER=$(echo "$USER_HOST" | cut -d'@' -f1)
HOST=$(echo "$USER_HOST" | cut -d'@' -f2)
REMOTE_HOST="$USER_HOST"
REMOTE_PORT="$PORT"
REMOTE_DIR="/root/proto_depth"

echo "Setting up environment on $REMOTE_HOST (port $REMOTE_PORT)"

# Create remote directory structure and ensure rsync is installed
ssh -p $REMOTE_PORT -o StrictHostKeyChecking=accept-new $REMOTE_HOST << EOF
  mkdir -p $REMOTE_DIR/data
  
  # Install rsync if not already installed
  if ! command -v rsync &> /dev/null; then
    echo "Installing rsync..."
    apt-get update && apt-get install -y rsync
  else
    echo "rsync is already installed."
  fi
EOF

# Copy the setup files to the remote machine
rsync -avz -e "ssh -p $REMOTE_PORT -o StrictHostKeyChecking=accept-new" \
  --include='pyproject.toml' \
  --include='requirements.txt' \
  --exclude='*' \
  ./ "$REMOTE_HOST:$REMOTE_DIR/"

# Setup the environment on the remote machine
ssh -p $REMOTE_PORT -o StrictHostKeyChecking=accept-new $REMOTE_HOST << EOF
  cd $REMOTE_DIR
  
  # Create conda environment
  if [ ! -d ".conda" ]; then
    echo "Miniconda not found. Installing..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $REMOTE_DIR/.conda
    rm miniconda.sh
  fi
  
  # Add conda to path
  export PATH="$REMOTE_DIR/.conda/bin:$PATH"
  
  # Check if environment exists
  ENV_EXISTS=\$(conda env list | grep pytorch3d | wc -l)
  
  if [ "\$ENV_EXISTS" -eq "0" ]; then
    echo "Setting up conda environment..."
    
    # Initialize conda for bash
    conda init bash
    source ~/.bashrc
    
    # Create a new conda environment with Python 3.9
    conda create -y -n pytorch3d python=3.9
    
    # Activate environment
    source $REMOTE_DIR/.conda/bin/activate pytorch3d
    
    # Install PyTorch 1.13.0 with CUDA 11.7 (compatible with PyTorch3D)
    conda install -y pytorch=1.13.0 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
    
    # Install additional dependencies
    conda install -y matplotlib tqdm
    conda install -y numpy=1.24.3  # Specific version for compatibility
    conda install -y plotly         # For interactive 3D visualizations
    
    # Install PyTorch3D from conda-forge
    conda install -y pytorch3d -c pytorch3d -c conda-forge
    
    # If conda install fails, try installing from source
    if [ \$? -ne 0 ]; then
      echo "Conda install of PyTorch3D failed, trying to install from source..."
      pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    fi
    
    echo "Conda environment setup complete"
  else
    echo "Pytorch3D environment already exists"
  fi
  
  echo "Environment setup completed successfully!"
EOF

echo "Remote environment setup completed"
echo "You can now run ./run_on_jarvis.sh to train the model"
