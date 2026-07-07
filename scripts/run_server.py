"""
Signal server startup script
Run this with: python scripts/run_server.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    print("Starting Trade RL Signal Server...")
    print("Press Ctrl+C to stop")
    print()

    from mars_lite.server.metrics_server import run_server

    run_server(host="0.0.0.0", port=8001)
