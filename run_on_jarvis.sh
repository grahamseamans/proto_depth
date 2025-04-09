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

# Define SSH options to be used consistently throughout the script
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "Preparing to run GPU-accelerated optimization on $REMOTE_HOST (port $REMOTE_PORT)"

# Sync code to remote machine (excluding big directories and temporary files)
echo "Syncing code to remote machine..."
rsync -avz -e "ssh -p $REMOTE_PORT $SSH_OPTS" \
  --exclude='data' --exclude='.git' --exclude='.venv' --exclude='.conda' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='progress_images' \
  ./ "$REMOTE_HOST:$REMOTE_DIR/"

# Sync the dataset to the remote machine
echo "Syncing dataset to remote machine..."
ssh -p $REMOTE_PORT $SSH_OPTS $REMOTE_HOST "mkdir -p $REMOTE_DIR/data/SYNTHIA-SF/SEQ1/DepthLeft"
rsync -avz -e "ssh -p $REMOTE_PORT $SSH_OPTS" \
  data/SYNTHIA-SF/SEQ1/DepthLeft/*.png "$REMOTE_HOST:$REMOTE_DIR/data/SYNTHIA-SF/SEQ1/DepthLeft/"

# Run the GPU-accelerated optimization on the remote machine
echo "Running GPU-accelerated optimization..."
ssh -p $REMOTE_PORT $SSH_OPTS $REMOTE_HOST << EOF
  cd $REMOTE_DIR
  
  # Install Kaolin for PyTorch 2.4.0 and CUDA 121
  echo "Installing Kaolin for PyTorch 2.4.0 and CUDA 121..."
  pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
  python -c "import kaolin; print(f'Kaolin version: {kaolin.__version__}')"
  
  # Print device info
  echo "Using device: \$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
  
  # Print dataset info
  echo "Found \$(find data/SYNTHIA-SF/SEQ1/DepthLeft -name "*.png" | wc -l) depth images"
  
  # Run the optimization with profiling flags
  # Using a reduced point cloud size initially to debug
  python run_energy_native.py --data_path data/SYNTHIA-SF/SEQ1/DepthLeft \
                             --num_iterations 100 \
                             --viz_interval 10 \
                             --image_index -1 \
                             --ico_level 3 \
                             --point_cloud_size 100000 \
                             --iterations_per_epoch 5
  
  # Check results
  echo "Optimization completed. Results saved to viz_server/data/"
EOF

# Sync results back to local machine
echo "Syncing results back to local machine..."
mkdir -p viz_server/data

rsync -avz -e "ssh -p $REMOTE_PORT $SSH_OPTS" \
  "$REMOTE_HOST:$REMOTE_DIR/viz_server/data/" "./viz_server/data/"

echo "Run completed successfully!"
echo "Interactive visualizations are available via 'sh run_viz_server.sh'"
