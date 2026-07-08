"""`toolsmith` console-script entry point: `toolsmith serve --mcp` and `toolsmith run --task`."""

from __future__ import annotations

import argparse
import sys


def _run_episode_cli(argv: list[str]) -> int:
    """Delegate to the P1-T23 episode-runner CLI (python -m toolsmith.env)."""
    from toolsmith.env.__main__ import main as env_main

    return env_main(argv)


def _run_mcp_server() -> None:
    """Delegate to the FastMCP server's stdio run loop."""
    from toolsmith.serve.mcp_server import mcp

    mcp.run()


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the `serve` or `run` subcommand."""
    parser = argparse.ArgumentParser(prog="toolsmith")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run the ToolSmith MCP server")
    serve_parser.add_argument(
        "--mcp", action="store_true", required=True, help="Serve over MCP (stdio transport)"
    )

    run_parser = subparsers.add_parser("run", help="Run one task through the episode runner")
    run_parser.add_argument("--task", required=True, help="Path to a task JSON file")
    run_parser.add_argument("--mode", default="sandbox", choices=["sandbox", "real"])

    args = parser.parse_args(argv)

    if args.command == "serve":
        _run_mcp_server()
        return 0

    return _run_episode_cli(["--task", args.task, "--mode", args.mode])


if __name__ == "__main__":
    sys.exit(main())
