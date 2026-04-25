"""Server entry point for the web API."""

from __future__ import annotations

import sys

from lingxi.web.app import create_app  # re-exported for tests and external use

__all__ = ["create_app"]


def main() -> None:
    """Start the uvicorn server."""
    try:
        import uvicorn
    except ImportError:
        print("需要安装 API 依赖: pip install persona-agent[api]")
        sys.exit(1)

    import argparse

    parser = argparse.ArgumentParser(description="Persona Agent API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"\n  Persona Agent API Server")
    print(f"  http://{args.host}:{args.port}")
    print(f"  API docs: http://{args.host}:{args.port}/docs\n")

    uvicorn.run(
        "lingxi.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
