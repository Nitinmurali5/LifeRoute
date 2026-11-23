"""LifeRoute Flask Development Server Entry Point.

Starts the Flask development server on host 127.0.0.1 and port 5000.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import create_app

app = create_app()

if __name__ == "__main__":
    host = os.environ.get("LIFEROUTE_HOST", "127.0.0.1")
    port = int(os.environ.get("LIFEROUTE_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    print(f"Starting LifeRoute Development Server at http://{host}:{port}/health")
    app.run(host=host, port=port, debug=debug)

