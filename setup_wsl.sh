#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_wsl.sh — one-time setup for the WSL side of Whisper Audio Recognition
# Run from the project root: bash setup_wsl.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e   # exit immediately if any command fails

echo "=== Whisper Audio Recognition — WSL Setup ==="
echo ""

# ── 1. Check Python ───────────────────────────────────────────────────────────
echo "Checking Python..."
python3 --version || { echo "ERROR: python3 not found."; exit 1; }

# ── 2. Check GPU access ───────────────────────────────────────────────────────
echo ""
echo "Checking GPU access from WSL..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "GPU is accessible from WSL."
else
    echo "ERROR: nvidia-smi not found."
    echo "Your GPU driver on Windows should expose CUDA to WSL2 automatically."
    echo "Make sure your NVIDIA Windows driver is up to date (>=522.06)."
    exit 1
fi

# ── 3. Create virtual environment ────────────────────────────────────────────
echo ""
echo "Creating virtual environment at .venv/ ..."

# python3-venv may not be installed on minimal WSL images
if ! python3 -m venv --help &>/dev/null; then
    echo "python3-venv not found — installing it via apt..."
    sudo apt install -y python3-venv python3-full
fi

python3 -m venv .venv
echo "Virtual environment created."

# ── 4. Install packages into the venv ────────────────────────────────────────
echo ""
echo "Installing WSL Python packages into .venv ..."
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt

echo ""
echo "=== WSL setup complete ==="
echo ""
echo "The model (~3 GB) will be downloaded automatically on first server start."
echo ""
echo "Next steps:"
echo "  1. Start the server (WSL terminal):     ./start_server.sh"
echo "  2. Transcribe a file (WSL terminal):    .venv/bin/python -m transcription.cli audio.mp3 -o out.txt"
echo "  3. Install Windows tray app (Windows):  pip install -r dictation/requirements_windows.txt"
echo "     Then run (Windows):                  python dictation/tray_app.py"
echo ""
echo "TIP: To avoid typing .venv/bin/python every time, activate the venv:"
echo "       source .venv/bin/activate"
echo "     Then just use: python -m transcription.cli ..."
echo "     Deactivate when done: deactivate"
