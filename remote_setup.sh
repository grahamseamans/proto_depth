#!/bin/bash

REMOTE_DIR="/root/proto_depth"
DATA_DIR="$REMOTE_DIR/data/SYNTHIA-SF"

# Create directories
mkdir -p "$DATA_DIR"

# Setup Python environment
if [ ! -d "$REMOTE_DIR/.conda" ]; then
    echo "Setting up conda environment..."
    cd "$REMOTE_DIR"
    
    echo "Installing Miniconda..."
    wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O "miniconda.sh"
    bash miniconda.sh -b -p "$REMOTE_DIR/.conda"
    rm miniconda.sh
    
    # Setup conda environment
    source "$REMOTE_DIR/.conda/bin/activate"
    
    echo "Creating conda environment..."
    conda create -y --prefix "$REMOTE_DIR/.conda" python=3.9 pytorch torchvision -c pytorch
    conda activate "$REMOTE_DIR/.conda"
    
    echo "Installing additional packages..."
    pip install 'git+https://github.com/facebookresearch/pytorch3d.git'
    pip install matplotlib
fi

echo "Setup complete! Environment is ready for training."
echo "To start training, run:"
echo "source $REMOTE_DIR/.conda/bin/activate"
echo "python train_model.py"
