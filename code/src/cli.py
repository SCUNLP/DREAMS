"""Command-line arguments shared by the two conversation entry points."""

import argparse
from pathlib import Path

from .model.demo import DemoTrace


def create_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--api_key', help='Deprecated; prefer the OPENAI_API_KEY environment variable')
    parser.add_argument('--dataset', required=True, choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--kg_dataset', required=True, choices=['redial', 'opendialkg'])
    parser.add_argument('--turn_num', type=int, default=10)
    parser.add_argument('--crs_model', default='mcts_dual', choices=['mcts_dual'])
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--mcts_iterations',
        type=int,
        default=3,
        help='Number of MCTS iterations from turn 3 onward (default: 3)',
    )
    parser.add_argument(
        '--simulation_workers',
        type=int,
        default=1,
        help='Concurrent rollout workers; 1 preserves sequential MCTS (default: 1)',
    )
    parser.add_argument('--trace', action='store_true', help='Write a live MCTS tree and event timeline')
    parser.add_argument('--trace_output', help='HTML path for --trace (default: demo/mcts_trace.html)')
    parser.add_argument('--no_trace_open', action='store_true', help='Do not open the trace page automatically')
    parser.add_argument('--ignore_warnings', action='store_true')
    return parser


def create_trace(args, repo_root):
    if not args.trace:
        return None
    output_path = Path(args.trace_output) if args.trace_output else Path(repo_root) / 'demo' / 'mcts_trace.html'
    return DemoTrace(output_path, auto_open=not args.no_trace_open)
