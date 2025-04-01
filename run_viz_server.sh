#!/bin/bash

# Visualization Server Launcher
# This script starts the Flask visualization server for the Proto-Depth model with live reloading,
# excluding the data directory from triggering reloads.

# Default values
HOST="0.0.0.0"
PORT=5000
DEBUG=true

# Check if viz_server directory exists
if [ ! -d "viz_server" ]; then
  echo "Error: viz_server directory not found. Please make sure the setup is correct."
  exit 1
fi

# Set up Flask environment
export FLASK_APP="viz_server/app.py"
if [ "$DEBUG" = true ]; then
  export FLASK_ENV="development"
else
  export FLASK_ENV="production"
fi

echo "Starting visualization server on http://localhost:$PORT"
echo "Live reloading enabled (ignoring changes in data directory)"

# Start Flask with debug mode and exclude data directory from triggering reloads
python -m flask run --host=$HOST --port=$PORT --debug 
