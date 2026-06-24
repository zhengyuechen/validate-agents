"""Launch the web UI: python -m valagents.web."""
from __future__ import annotations

import argparse

import uvicorn

from valagents.web.app import create_app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="valagents.web", description="Validate-agents web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind (default: 8001).")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--results", default="results", help="Results directory.")
    args = parser.parse_args(argv)
    uvicorn.run(create_app(config_path=args.config, results_base=args.results),
                host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
