#!/bin/bash

# Parse SSH connection details from instance_address.txt
SSH_STRING=$(cat instance_address.txt)
PORT=$(echo "$SSH_STRING" | sed -E 's/.*-p ([0-9]+).*/\1/')
USER_HOST=$(echo "$SSH_STRING" | awk '{print $NF}')
REMOTE_HOST="$USER_HOST"
REMOTE_PORT="$PORT"
REMOTE_DIR="/root/proto_depth"

# Define SSH options to be used consistently throughout the script
SSH_OPTS="-o StrictHostKeyChecking=accept-new"

echo "Preparing to run model on $REMOTE_HOST (port $REMOTE_PORT)"

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


# Run the training on the remote machine
echo "Running model training..."
ssh -p $REMOTE_PORT $SSH_OPTS $REMOTE_HOST << EOF
  cd $REMOTE_DIR
  
  # Clear previous progress images if they exist
  rm -rf training_progress/*
  
  # Activate conda environment
  export PATH="$REMOTE_DIR/.conda/bin:$PATH"
  source $REMOTE_DIR/.conda/bin/activate pytorch3d
  
  # Print device info
  echo "Using device: \$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
  
  # Print dataset info
  echo "Found \$(find data/SYNTHIA-SF/SEQ1/DepthLeft -name "*.png" | wc -l) depth images"
  
  # Run the training
  python train_model.py
  
  # Check results
  echo "Training completed. Results saved to training_progress/"
EOF

# Sync results back to local machine
echo "Syncing results back to local machine..."
mkdir -p viz_server/data

rsync -avz -e "ssh -p $REMOTE_PORT $SSH_OPTS" \
  "$REMOTE_HOST:$REMOTE_DIR/viz_server/data/" "./viz_server/data/"

echo "Run completed successfully!"
echo "Interactive visualizations are available via 'python run_viz_server.py'"
