#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-10}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/dual_chat_eval.log}"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to run the Python script
run_script() {
    log "Starting chat_eval.py with arguments: $*"
    python3 "$SCRIPT_DIR/chat_dual.py" "$@"
    return $?
}

# Show usage if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [arguments]"
    echo ""
    echo "Required arguments:"
    echo "  --dataset redial_eval|opendialkg_eval    Dataset to use"
    echo "  --crs_model mcts_dual                  CRS model to use"
    echo ""
    echo "Optional arguments:"
    echo "  Set OPENAI_API_KEY in the environment before running."
    echo "  --turn_num NUM                           Number of turns (default: 10)"
    echo "  --seed NUM                               Random seed (default: 100)"
    echo "  --debug                                  Enable debug mode"
    echo "  --mcts_iterations NUM                    Iterations from turn 3 onward (default: 3)"
    echo "  --simulation_workers NUM                 Concurrent rollout workers (default: 1)"
    echo "  --trace                                  Open a live tree/reward/feedback view"
    echo "  --no_trace_open                          Write trace HTML without opening it"
    echo ""
    echo "Example:"
    echo "  $0 --dataset redial_eval --kg_dataset redial --crs_model mcts_dual"
    exit 1
fi

# Main execution
attempt=1
while [ $attempt -le $MAX_RETRIES ]; do
    log "Attempt $attempt of $MAX_RETRIES"

    if run_script "$@"; then
        log "chat_dual.py completed successfully"
        exit 0
    else
        log "chat_dual.py failed with exit code $?"

        if [ $attempt -lt $MAX_RETRIES ]; then
            log "Waiting $RETRY_DELAY seconds before retrying..."
            sleep $RETRY_DELAY
        fi
    fi

    attempt=$((attempt + 1))
done

log "Maximum retry attempts reached. Exiting."
exit 1
