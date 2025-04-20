#!/bin/bash

# Check if script argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_path>"
    echo "Example: $0 scripts/visualize_dataloader.py"
    exit 1
fi

SCRIPT_PATH=$1

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

# Define SSH options to be used consistently throughout the script
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "Preparing to run $SCRIPT_PATH on $REMOTE_HOST (port $REMOTE_PORT)"

# Sync code to remote machine (excluding big directories and temporary files)
echo "Syncing code to remote machine..."
rsync -avz -e "ssh -p $REMOTE_PORT $SSH_OPTS -i $SSH_KEY" \
  --exclude='data' --exclude='.git' --exclude='.venv' --exclude='.conda' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='progress_images' \
  --exclude='kaolin'  --exclude='Open3D' --exclude='nvdiffrast' \
  ./ "root@$REMOTE_HOST:$REMOTE_DIR/"

# Run the script on the remote machine
echo "Running script..."
ssh -t root@$REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY << EOF
  cd $REMOTE_DIR
  PYTHONPATH=$REMOTE_DIR python $SCRIPT_PATH
EOF

# Sync results back to local machine
echo "Syncing results back to local machine..."

# Create local directories
mkdir -p tests/output
mkdir -p viz_server/data

# Sync both output directories
rsync -avz -e "ssh -p $REMOTE_PORT $SSH_OPTS -i $SSH_KEY" \
  "root@$REMOTE_HOST:$REMOTE_DIR/tests/output/" "./tests/output/"

rsync -avz -e "ssh -p $REMOTE_PORT $SSH_OPTS -i $SSH_KEY" \
  "root@$REMOTE_HOST:$REMOTE_DIR/viz_server/data/" "./viz_server/data/"

echo "Run completed successfully!"
