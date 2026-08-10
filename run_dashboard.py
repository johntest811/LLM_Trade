"""
run_dashboard.py
Standalone launcher for the monitoring dashboard.

Usage:
    python run_dashboard.py           # starts on http://localhost:8080
    python run_dashboard.py --port 9090

The dashboard works without the trading engine running.
When the engine is active, it writes to ui.state.dashboard_state
and the WebSocket broadcast loop picks up the changes automatically.

Optimized for Ryzen 5 5600 / RTX 3060 12 GB / 32 GB RAM:
  - Single uvicorn worker (trading loop is fully async, no WSGI forking)
  - loop = asyncio default (fastest on Windows with ProactorEventLoop)
  - log_level = 'warning' to minimise logging overhead during inference
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="LLM Trading Dashboard")
    parser.add_argument("--port",  default=8080, type=int, help="Port (default 8080)")
    parser.add_argument("--debug", action="store_true",   help="Enable uvicorn debug logging")
    args = parser.parse_args()

    print(f"\n  LLM Trading Terminal")
    print(f"  Dashboard -> http://localhost:{args.port}\n")

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=args.port,
        log_level="debug" if args.debug else "warning",
        reload=False,       # disabled in production — no file watcher overhead
        workers=1,          # single worker: shares asyncio loop with trading engine
        access_log=False,   # suppress per-request noise in 24/7 mode
        loop="auto",        # uses WindowsProactorEventLoop on Windows
    )


if __name__ == "__main__":
    main()
