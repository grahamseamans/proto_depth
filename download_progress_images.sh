#!/bin/bash

# Configuration
REMOTE_HOST="root@ssha.jarvislabs.ai"
REMOTE_PORT="11614"  # Using the port from external_server.txt
REMOTE_DIR="/root/proto_depth"
LOCAL_DIR="./progress_images"

# Create local directory to store images
mkdir -p "$LOCAL_DIR"

# Download progress images using scp
echo "Downloading progress images from remote server..."
scp -P $REMOTE_PORT -o StrictHostKeyChecking=no "$REMOTE_HOST:$REMOTE_DIR/progress_epoch_*.png" "$LOCAL_DIR/"

echo "Download complete. Images saved to $LOCAL_DIR/"
