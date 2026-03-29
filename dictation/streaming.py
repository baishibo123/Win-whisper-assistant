"""
Streaming dictation — real-time VAD-chunked transcription.

When the user presses Ctrl+Win+X:
  1. Opens an audio stream (WASAPI → MME → default fallback)
  2. Feeds audio frames to VADChunker
  3. Each detected speech chunk is POSTed to /transcribe-chunk
  4. Server returns text, which is pasted at the cursor immediately
  5. Context carry-forward: each request includes the last ~200 chars
     of accumulated text so Whisper keeps vocabulary/language consistent.

Press Ctrl+Win+X again to stop.
"""

import io
import json
import os
import sys
import threading
from typing import TYPE_CHECKING, Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

from dictation.clipboard import paste_with_restore
from dictation.vad_chunker import VADChunker

if TYPE_CHECKING:
    from dictation.app import Application

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
)

# VAD-compatible sample rates. We prefer 16000 Hz.
_VAD_RATE = 16000


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"server_port": 8765}


def _get_devices_for_hostapi(api_substring: str) -> list[tuple[str, int]]:
    try:
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()
        api_idx = next(
            (i for i, h in enumerate(hostapis) if api_substring in h["name"]), None
        )
        if api_idx is None:
            return []
        return [
            (d["name"], i)
            for i, d in enumerate(devices)
            if d["hostapi"] == api_idx and d["max_input_channels"] > 0
        ]
    except Exception:
        return []


def _find_device_index(name: Optional[str], api: str = "WASAPI") -> Optional[int]:
    if not name:
        return None
    for dev_name, idx in _get_devices_for_hostapi(api):
        if name.lower() in dev_name.lower() or dev_name.lower() in name.lower():
            return idx
    return None


class StreamingDictation:
    """
    Manages a streaming dictation session.

    Lifecycle: start() → [user speaks, chunks transcribed & pasted] → stop()
    """

    def __init__(self, app: "Application"):
        self.app = app
        self._stream: Optional[sd.InputStream] = None
        self._chunker: Optional[VADChunker] = None
        self._sender_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._accumulated_text = ""

    def start(self) -> None:
        """Open audio stream and begin VAD chunking."""
        cfg = _load_config()
        dev_name = cfg.get("audio_device_name")
        aggressiveness = cfg.get("vad_aggressiveness", 2)
        silence_ms = cfg.get("silence_threshold_ms", 600)

        self._stop_event.clear()
        self._accumulated_text = ""

        # Build fallback chain: WASAPI → MME → default
        # For streaming, we force 16kHz via MME (webrtcvad compatible).
        # WASAPI may record at 48kHz which also works with webrtcvad.
        attempts = []
        if dev_name:
            wasapi_idx = _find_device_index(dev_name, "WASAPI")
            mme_idx = _find_device_index(dev_name, "MME")
            if wasapi_idx is not None:
                dev_info = sd.query_devices(wasapi_idx)
                native_rate = int(dev_info["default_samplerate"])
                # webrtcvad needs 8/16/32/48 kHz
                if native_rate in (8000, 16000, 32000, 48000):
                    attempts.append(("WASAPI", wasapi_idx, native_rate))
            if mme_idx is not None:
                attempts.append(("MME", mme_idx, _VAD_RATE))
        attempts.append(("default", None, _VAD_RATE))

        last_error = None
        for api_label, device_idx, rate in attempts:
            try:
                self._chunker = VADChunker(
                    sample_rate=rate,
                    aggressiveness=aggressiveness,
                    silence_threshold_ms=silence_ms,
                )

                self._stream = sd.InputStream(
                    samplerate=rate,
                    channels=1,
                    dtype="float32",
                    device=device_idx,
                    callback=self._audio_callback,
                )
                self._stream.start()

                # Start the chunk sender thread
                self._sender_thread = threading.Thread(
                    target=self._send_chunks,
                    args=(rate,),
                    daemon=True,
                )
                self._sender_thread.start()

                print(f"[streaming] Started via {api_label} @ {rate} Hz")
                return

            except sd.PortAudioError as e:
                last_error = e
                print(f"[streaming] {api_label} failed ({e}), trying next...", file=sys.stderr)
                if self._stream:
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None

        print(f"[streaming] ERROR: could not open audio: {last_error}", file=sys.stderr)
        self.app.release_mode()

    def stop(self) -> None:
        """Stop recording, flush remaining audio, clean up."""
        self._stop_event.set()

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Flush any remaining speech in the VAD buffer
        if self._chunker:
            final_chunk = self._chunker.flush()
            if final_chunk is not None:
                self._transcribe_and_paste(final_chunk, self._chunker.sample_rate)
            self._chunker = None

        if self._sender_thread:
            self._sender_thread.join(timeout=5)
            self._sender_thread = None

        print(f"[streaming] Stopped. Total text: {len(self._accumulated_text)} chars")

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """sounddevice callback — feed raw audio to VAD chunker."""
        if self._chunker and not self._stop_event.is_set():
            self._chunker.feed(indata)

    def _send_chunks(self, sample_rate: int) -> None:
        """Background thread: read chunks from VAD queue, transcribe, paste."""
        while not self._stop_event.is_set():
            try:
                chunk = self._chunker.chunk_queue.get(timeout=0.2)
            except Exception:
                continue

            self._transcribe_and_paste(chunk, sample_rate)

    def _transcribe_and_paste(self, chunk: np.ndarray, sample_rate: int) -> None:
        """Send one audio chunk to the server and paste the result."""
        cfg = _load_config()
        server_url = f"http://localhost:{cfg.get('server_port', 8765)}"
        prior_context = self._accumulated_text[-200:]

        # Encode chunk as WAV in memory
        buf = io.BytesIO()
        sf.write(buf, chunk, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)

        try:
            resp = requests.post(
                f"{server_url}/transcribe-chunk",
                files={"audio": ("chunk.wav", buf, "audio/wav")},
                data={"prior_context": prior_context},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            text = result.get("text", "").strip()

            if text:
                lang = result.get("language", "?")
                print(f"[streaming] [{lang}] {text[:80]}")
                self._accumulated_text += text + " "
                paste_with_restore(text)

        except requests.ConnectionError:
            print("[streaming] ERROR: server unreachable", file=sys.stderr)
        except Exception as e:
            print(f"[streaming] ERROR: {e}", file=sys.stderr)
