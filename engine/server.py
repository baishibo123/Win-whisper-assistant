"""
Whisper FastAPI Server — the HTTP interface around WhisperEngine.

How it fits in the architecture:
  - This file is the "waiter" between HTTP clients and the WhisperEngine "cook".
  - It starts once, loads the model (via WhisperEngine.__init__), then waits.
  - Every POST to /transcribe calls engine.transcribe() and returns JSON.
  - GET /health lets clients check if the server is alive before sending audio.

Start with:
  python -m uvicorn engine.server:app --host 0.0.0.0 --port 8765
  (or use ./start_server.sh)
"""

import asyncio
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

# Allow running as: python -m uvicorn engine.server:app from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.whisper_engine import WhisperEngine

# ── Lifespan: runs setup/teardown around the server's life ────────────────────
#
# @asynccontextmanager + lifespan is FastAPI's way of saying:
#   "run this code when the server starts, and this other code when it stops"
# It replaces the older @app.on_event("startup") pattern.
#
engine: Optional[WhisperEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = WhisperEngine()      # loads model into GPU — happens ONCE at startup
    yield                         # server runs here, handling requests
    engine = None                 # cleanup when server shuts down (Ctrl+C)


app = FastAPI(
    title="Whisper Audio Recognition",
    description="Local speech-to-text server powered by faster-whisper.",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Lightweight ping endpoint.
    Clients call this at startup to confirm the server is running.
    Returns 200 OK immediately — no model inference involved.
    """
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    timestamps: str   = Form("false"),
    language: str     = Form(""),
):
    """
    Transcribe an uploaded audio file.

    Parameters (sent as multipart/form-data):
      audio      : the audio file (wav, mp3, m4a, flac, etc.)
      timestamps : "true" or "false" — include segment timestamps in output
      language   : force a language code ("zh", "en") or "" for auto-detect

    Returns JSON:
      {
        "text":                 string or list of {start, end, text}
        "language":             detected language code
        "language_probability": 0.0–1.0
      }

    Why Form() instead of Query()?
    Audio files must be sent as multipart/form-data (not URL parameters),
    so all companion fields must also be Form fields in the same request.
    """
    use_timestamps = timestamps.lower() == "true"
    lang           = language.strip() or None      # "" → None (auto-detect)

    # Determine file extension so the temp file has the right suffix.
    # faster-whisper uses the extension to pick the right audio decoder.
    original_name = audio.filename or "recording.wav"
    ext = "." + original_name.rsplit(".", 1)[-1] if "." in original_name else ".wav"

    audio_bytes = await audio.read()   # read from HTTP request into memory

    # Write to a temp file — faster-whisper needs a file path, not raw bytes.
    # delete=False because we manually delete after inference (generator safety).
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Run inference in a thread so we don't block FastAPI's async event loop.
        # faster-whisper is synchronous (CPU/GPU bound), so we offload it to a
        # thread via run_in_executor. This lets the server stay responsive to
        # other requests (e.g. /health checks) while inference runs.
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,                                     # use default thread pool
            lambda: engine.transcribe(tmp_path, use_timestamps, lang),
        )
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        # Always clean up the temp file, even if transcription raised an error.
        # "finally" runs whether or not an exception occurred.
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
