#!/bin/bash
set -e
cd "$(dirname "$0")"
set -a
source .env
set +a
exec .venv/bin/python -u code/web_demo.py --dataset redial --mcts_iterations 3 --simulation_workers 3 "$@"
