#!/bin/bash

# Configuration
REMOTE_USER_HOST="horia.mercan@fep8.grid.pub.ro"
REMOTE_BASE_DIR="/export/home/acs/stud/h/horia.mercan/phantora"
LOCAL_BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# List of files and directories to sync (paths are relative to the Phantora base direction)
# You can modify this array to add or remove paths you'd like to sync.
ITEMS=(
    "phantora/phantora/"
    "phantora/netsim/"
    "phantora/cuda_call/"
    "phantora/visualizer/"
    # "stub/"
    # "include/"
    "tests/"
    # "Makefile"
    "run_phantora_deepspeed.sh"
    "run_phantora_torchtitan.sh"
    # "patch_custom_model_load.py"
    "custom_model_results.json"
)

# Print Usage
usage() {
    echo "Usage: $0 [up|down]"
    echo "  up   : Sync from local host TO server (local -> remote)"
    echo "  down : Sync from server TO local host (remote -> local)"
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

DIRECTION=$1

# Change to local base directory to make relative syncing easy
cd "$LOCAL_BASE_DIR" || { echo "Failed to cd to $LOCAL_BASE_DIR"; exit 1; }

if [ "$DIRECTION" == "up" ]; then
    echo "Direction: Local -> Server"
    echo "Target: $REMOTE_USER_HOST:$REMOTE_BASE_DIR"
    
    # Ensure the remote base directory exists
    ssh "$REMOTE_USER_HOST" "mkdir -p $REMOTE_BASE_DIR/Phantora"
    
    # Use rsync with --relative to maintain the directory structure
    for item in "${ITEMS[@]}"; do
        echo "-----------------------------------"
        echo "Syncing $item ..."
        if [[ "$item" == run_phantora_*.sh ]]; then
            TARGET_DIR="$REMOTE_BASE_DIR"
        else
            TARGET_DIR="$REMOTE_BASE_DIR/Phantora"
        fi
        rsync -avz --progress --relative "$item" "$REMOTE_USER_HOST:$TARGET_DIR/"
    done
    
    echo "-----------------------------------"
    echo "Sync (up) complete!"

elif [ "$DIRECTION" == "down" ]; then
    echo "Direction: Server -> Local"
    echo "Source: $REMOTE_USER_HOST:$REMOTE_BASE_DIR"
    
    for item in "${ITEMS[@]}"; do
        echo "-----------------------------------"
        echo "Syncing $item..."
        if [[ "$item" == run_phantora_*.sh ]]; then
            SOURCE_DIR="$REMOTE_BASE_DIR"
        else
            SOURCE_DIR="$REMOTE_BASE_DIR/Phantora"
        fi
        # The /./ syntax tells rsync where the relative path starts on the remote string
        rsync -avz --progress --relative "$REMOTE_USER_HOST:$SOURCE_DIR/./$item" "$LOCAL_BASE_DIR/"
    done
    
    echo "-----------------------------------"
    echo "Sync (down) complete!"

else
    echo "Invalid direction: $DIRECTION"
    usage
fi
