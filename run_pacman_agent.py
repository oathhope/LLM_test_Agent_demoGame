"""
MsPacman LLM Agent Launcher

Starts the LLM agent that connects to the MsPacman game server and plays the game.
Requires the game server (run_pacman_game.py) to be running first.

Usage:
    python pacman/run_pacman_agent.py [OPTIONS]

Options:
    --provider    LLM provider: anthropic (default: anthropic)
    --model       Model name (default: claude-sonnet-4-5)
    --server-url  Game server URL (default: http://127.0.0.1:5001)
    --max-turns   Max turns to play (default: 0 = unlimited)
    --delay       Delay between turns in seconds (default: 0.5)
    --mock        Use mock LLM (no API calls, rule-based decisions)
"""

import sys
import os
import argparse

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pacman.agent.pacman_agent import PacmanAgent


def main():
    parser = argparse.ArgumentParser(description="MsPacman LLM Agent")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "mock"],
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--model", default="claude-sonnet-4-5",
                        help="Model name (default: claude-sonnet-4-5)")
    parser.add_argument("--server-url", default="http://127.0.0.1:5001",
                        help="Game server URL (default: http://127.0.0.1:5001)")
    parser.add_argument("--max-turns", type=int, default=0,
                        help="Max turns (0 = unlimited, default: 0)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between turns in seconds (default: 0.5)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM (no API calls)")
    args = parser.parse_args()

    provider = "mock" if args.mock else args.provider

    print(f"Starting MsPacman LLM Agent")
    print(f"  Provider: {provider}")
    if provider != "mock":
        print(f"  Model: {args.model}")
    print(f"  Game Server: {args.server_url}")
    print(f"  Max Turns: {'unlimited' if args.max_turns == 0 else args.max_turns}")
    print(f"  Turn Delay: {args.delay}s")
    print("Press Ctrl+C to stop.\n")

    agent = PacmanAgent(
        game_url=args.server_url,
        provider=provider,
        model=args.model,
        max_steps=args.max_turns if args.max_turns > 0 else 10000,
        interval=args.delay,
    )
    agent.run()


if __name__ == "__main__":
    main()
