#!/bin/bash

REMOTE_DIR="/root/proto_depth"
DATA_DIR="$REMOTE_DIR/data/SYNTHIA-SF"

# Function to check if a process is still running
check_pid() {
    if ps -p $1 > /dev/null; then
        return 0  # Process is running
    else
        return 1  # Process is not running
    fi
}

# Function to download with progress and wait for completion
download_and_wait() {
    local url="$1"
    local output="$2"
    
    # Start download if not already completed
    if [ ! -f "$output" ] || [ "$(stat -f%z "$output")" -eq 0 ]; then
        echo "Starting download of $output..."
        wget -c "$url" -O "$output" > "$output.log" 2>&1 &
        local pid=$!
        echo "Download started (PID: $pid)"
        
        # Monitor progress
        while check_pid $pid; do
            echo -n "."
            sleep 5
        done
        echo "Download completed!"
    else
        echo "$output already exists and is not empty"
    fi
}

# Create directories
mkdir -p "$DATA_DIR"

# Download and extract SYNTHIA dataset
if [ ! -d "$DATA_DIR/SEQ1" ]; then
    cd "$DATA_DIR"
    echo "Downloading SYNTHIA dataset..."
    download_and_wait "http://synthia-dataset.net/download/808/" "synthia.zip"
    
    echo "Extracting dataset..."
    unzip -o synthia.zip
    rm synthia.zip
    echo "Dataset extraction complete"
fi

# Setup Python environment
if [ ! -d "$REMOTE_DIR/.conda" ]; then
    echo "Setting up conda environment..."
    cd "$REMOTE_DIR"
    download_and_wait "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" "miniconda.sh"
    
    echo "Installing Miniconda..."
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