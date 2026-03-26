"""
MsPacman Game Server Launcher

Starts the MsPacman game with pygame visual window and embedded Flask RPC server.
The RPC server listens on http://127.0.0.1:5001

Usage:
    python pacman/run_pacman_game.py [--port PORT] [--scale SCALE]
"""

import sys
import os
import argparse

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pacman.game.game_main import run_game


def main():
    parser = argparse.ArgumentParser(description="MsPacman Game Server with pygame visual")
    parser.add_argument("--host", default="127.0.0.1", help="RPC server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5001, help="RPC server port (default: 5001)")
    args = parser.parse_args()

    print(f"Starting MsPacman Game Server on port {args.port}...")
    print(f"RPC endpoint: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")

    run_game(rpc_host=args.host, rpc_port=args.port)


if __name__ == "__main__":
    main()
