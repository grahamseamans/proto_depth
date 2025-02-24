#!/bin/bash

# Remote machine details
REMOTE_HOST="root@ssha.jarvislabs.ai"
REMOTE_PORT="11414"
REMOTE_DIR="/root/proto_depth"

# Sync code to remote machine (excluding unnecessary directories)
echo "Syncing code to remote machine..."
/usr/bin/rsync -avz -e "ssh -p $REMOTE_PORT" \
    --exclude 'data/' \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.conda/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'Miniconda3-latest-MacOSX-arm64.sh' \
    --exclude '.DS_Store' \
    . "$REMOTE_HOST:$REMOTE_DIR"

# Make setup script executable and run it
echo "Starting remote setup..."
ssh -p $REMOTE_PORT $REMOTE_HOST "cd $REMOTE_DIR && chmod +x remote_setup.sh && nohup ./remote_setup.sh > setup.log 2>&1 &"

echo "Setup started in background on remote machine"
echo "To check progress, run:"
echo "ssh -p $REMOTE_PORT $REMOTE_HOST 'tail -f $REMOTE_DIR/setup.log'" 