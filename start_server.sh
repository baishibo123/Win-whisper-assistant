#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_server.sh — start the Whisper FastAPI server in WSL
# Run from the project root: ./start_server.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

# ── Resolve Python: prefer venv, fall back to system python3 ─────────────────
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    echo "WARNING: .venv not found. Run bash setup_wsl.sh first."
    PYTHON="python3"
fi

# Read port from config.json (fallback to 8765 if parsing fails)
PORT=$($PYTHON -c "
import json
try:
    print(json.load(open('config.json')).get('server_port', 8765))
except Exception:
    print(8765)
" 2>/dev/null || echo "8765")

echo "Starting Whisper server on port $PORT ..."
echo "The model will be downloaded and loaded into GPU on first start."
echo "This takes 30-60 seconds. Subsequent starts are faster (model is cached)."
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# uvicorn flags:
#   engine.server:app   → Python module path to the FastAPI app object
#   --host 0.0.0.0      → listen on all interfaces (required for WSL↔Windows)
#   --port $PORT        → from config.json
#   --log-level info    → show request logs in terminal
$PYTHON -m uvicorn engine.server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info
