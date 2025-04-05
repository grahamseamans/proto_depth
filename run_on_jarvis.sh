#!/bin/bash

# Define SSH connection details
REMOTE_HOST="root@sshg.jarvislabs.ai"
REMOTE_PORT="11014"
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
  
  # Print device info
  echo "Using device: \$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
  
  # Print dataset info
  echo "Found \$(find data/SYNTHIA-SF/SEQ1/DepthLeft -name "*.png" | wc -l) depth images"
  
  # Run the optimization
  # For a meaningful optimization with more iterations
  python run_energy_native.py --data_path data/SYNTHIA-SF/SEQ1/DepthLeft --num_iterations 100 --viz_interval 10 --image_index -1 --ico_level 4
  
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
