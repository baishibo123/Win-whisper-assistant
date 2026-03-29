# Whisper Audio Recognition — Project Notes for Claude

## What This Project Does

A personal speech-to-text system running locally on an RTX 4090. Three interfaces share one Whisper engine:

1. **Streaming Dictation** — press `Ctrl+Win+X`, speak, text appears at cursor in real-time (VAD-chunked), press again to stop
2. **Transcription GUI** — double-click tray icon → browse audio file → progress bar → save transcript
3. **Transcription CLI** — command-line tool that sends an audio file to the server and writes a transcript

## Hardware / Environment

- **GPU**: NVIDIA RTX 4090, 24 GB VRAM — more than sufficient for large-v3
- **OS split**: WSL2 (Ubuntu) for the GPU/model side, Windows for audio capture and input injection
- **Python**: 3.12.3 in WSL (venv at `.venv/`), 3.13 natively on Windows
- **CUDA**: works in WSL2 via Windows driver passthrough — no separate Linux driver needed

## Architecture

```
WSL2 (Ubuntu)
├── engine/whisper_engine.py   model wrapper, loads large-v3 into GPU once
├── engine/server.py           FastAPI HTTP server on localhost:8765
│                              endpoints: /health, /transcribe, /transcribe-chunk, /transcribe/stream
└── transcription/cli.py       CLI client → POSTs audio → gets text

Windows (Python 3.13, native)
├── dictation/tray_app.py      entry point — launches everything
├── dictation/app.py           Application class — central coordinator, mode management
├── dictation/tray.py          TrayManager — pystray icon, menu, status colors
├── dictation/streaming.py     StreamingDictation — VAD loop + chunk sender
├── dictation/vad_chunker.py   VADChunker — webrtcvad speech/silence detection
├── dictation/server_manager.py ServerManager — auto-launches WSL server, health polls, logs
├── dictation/gui.py           TranscriptionWindow — tkinter file transcription UI
└── dictation/clipboard.py     Clipboard save/restore utility
```

WSL2 and Windows communicate via `localhost:8765` — WSL2 port forwarding handles this automatically.

**Why the split**: CUDA lives in WSL. System tray icons, global hotkeys (`Ctrl+Win+X`), and clipboard paste into Windows apps require native Windows APIs — they cannot be done from WSL.

**Threading model** (Windows side):
- Main thread: tkinter mainloop (hidden root, GUI window shown on demand)
- Thread 1: pystray Icon.run_detached() — Win32 message pump
- Thread 2: pynput GlobalHotKeys (Ctrl+Win+X)
- Thread 3: sounddevice audio callback (PortAudio internal)
- Thread 4: VAD chunk sender (reads queue, POSTs to server)
- Thread 5: Server subprocess stdout reader
- Thread 6: Health check poller (every 3s)
- Cross-thread communication: queue.Queue polled by tkinter root.after(100ms)

## How to Run

```bash
# WSL — one-time setup
bash setup_wsl.sh

# Windows PowerShell — one-time setup
pip install -r dictation\requirements_windows.txt

# Windows — run everything (server auto-launches, no terminal window needed)
pythonw dictation\tray_app.py

# WSL — transcribe an audio file (CLI, alternative to GUI)
.venv/bin/python -m transcription.cli meeting.mp3 -o transcript.txt
.venv/bin/python -m transcription.cli meeting.mp3 --timestamps
```

The tray app auto-launches the WSL server as a hidden subprocess. Use `pythonw` (not `python`) to avoid keeping a PowerShell window open. Server logs are visible in the GUI transcription window.

## Config (`config.json`)

All runtime settings live here. Both WSL server and Windows tray app read this file.

| Key | Default | Notes |
|-----|---------|-------|
| `model` | `"large-v3"` | Whisper model variant |
| `device` | `"cuda"` | `"cpu"` as fallback |
| `compute_type` | `"float16"` | RTX 4090 supports float16 natively |
| `language` | `null` | null = auto-detect per segment (best for CN/EN mixing) |
| `beam_size` | `5` | Higher = more accurate, slower |
| `initial_prompt` | see file | Seeds Whisper with custom vocabulary for proper nouns |
| `server_port` | `8765` | FastAPI server port |
| `audio_device_name` | `"麦克风 (2K USB Camera-Audio)"` | Matched by name, not index |
| `vad_filter` | `true` | Filters silence before inference |
| `vad_aggressiveness` | `2` | webrtcvad aggressiveness 0-3 for streaming dictation |
| `silence_threshold_ms` | `600` | Silence duration (ms) that triggers chunk emit in streaming |
| `chunk_beam_size` | `1` | Beam size for streaming chunks (lower = faster) |

**Important**: `audio_device_name` stores the device name as a string, not a number. Device indices in Windows shift when USB devices are plugged/unplugged. The tray app resolves the name to an index at runtime via `sounddevice.query_devices()`.

## Bilingual Support

User speaks primarily Chinese (Mandarin) with incidental English words, or pure Chinese/pure English. `language: null` lets Whisper auto-detect per 30-second segment — this handles all three patterns without configuration. The `initial_prompt` field improves recognition of proper nouns like "Claude", "Anthropic", "WSL".

## Bugs Fixed During Initial Development

### 1. `externally-managed-environment` pip error (WSL)
Modern Debian/Ubuntu blocks system-wide pip installs. Fixed by creating a venv: `setup_wsl.sh` now runs `python3 -m venv .venv` and installs into `.venv/bin/pip`.

### 2. Tray app `PortAudio library not found` in WSL
User ran `tray_app.py` from WSL — it must run on Windows. PortAudio (microphone access) and Windows APIs for tray/hotkey/clipboard are not available from WSL.

### 3. Empty transcription despite chunks being captured
Default Windows microphone (Sound Mapper) was pointing to an inactive device. VAD filter silently discarded the near-silence. Fixed by selecting the correct microphone via the tray menu.

### 4. `Invalid sample rate [PaErrorCode -9997]` on WASAPI device
WASAPI devices only accept their hardware native sample rate (typically 44100 or 48000 Hz), not the 16000 Hz we were requesting. Fixed by querying `device["default_samplerate"]` at recording start and recording at native rate. `faster-whisper` uses `ffmpeg` internally which resamples to 16 kHz before inference — so the WAV can be sent at any rate.

### 5. `WdmSyncIoctl` / `Unanticipated host error` on WASAPI stream start
The thinkplus meeting dock's WASAPI stream was rejected at the driver level (likely held in exclusive mode by another app). Fixed by adding a fallback chain: WASAPI → MME → system default. MME is older but Windows mediates access and handles sample rate conversion, making it far more tolerant. The tray app logs which API it ultimately used.

### 6. `SyntaxError: name '_is_recording' is used prior to global declaration`
A `global _is_recording` statement was placed inside a `with _lock:` block after the variable was already used in the function. Python's `global` declarations must appear at the function's top level before any use of the name. Removed the duplicate declaration.

### 7. Orphaned WSL server after crash or Task Manager kill
If the tray app was killed via Task Manager (or crashed), the WSL uvicorn server kept running on port 8765. On next launch, the new server failed with "address already in use". Fixed with a multi-layer cleanup strategy: (1) on startup, `ServerManager._check_port()` probes localhost — if it finds a stale Whisper server (verified via `/health` response shape), it kills it before launching a new one; if the port is occupied by another program, it refuses to start and notifies the user. (2) `atexit.register(self.stop)` catches semi-graceful exits (unhandled exceptions, sys.exit). (3) SIGINT/SIGTERM signal handlers in `tray_app.py` catch Ctrl+C.

## Known Issues

### Background noise triggers false transcription in streaming mode
When streaming dictation is active and the user is silent, background noise (keyboard, fan, ambient) can be picked up by webrtcvad as speech. The noise chunk gets sent to Whisper which hallucinates text from it. The real speech transcription works correctly — this only affects idle periods. Potential fixes: raise `vad_aggressiveness` to 3, add a minimum RMS energy threshold before accepting a VAD-detected chunk, or add a minimum chunk duration filter (reject chunks shorter than ~300ms).

## Future Work

### Near-term
- **Background noise filtering**: see Known Issues above. Needs an energy gate or stricter VAD tuning.
- **Text injection via SendInput**: currently uses clipboard save/restore. `win32api.SendInput` would inject characters directly without touching the clipboard at all. Requires `pywin32`. Challenge: Chinese characters need IME simulation which can be unreliable.
- **Tray icon hidden by default**: Windows 11 hides new tray icons in the overflow (`^`) area. User must manually pin it. No code fix — Windows setting.
- **Dynamic microphone submenu**: currently built once at startup. Plugging in a new device after startup requires restart. Fix by making the submenu a dynamic callable.

### Medium-term
- **Custom text replacements**: a `replacements` dict in `config.json` (e.g. `"CloakCode": "Claude Code"`). Applied post-transcription, before paste. Complements `initial_prompt` for recurring STT errors.
- **Transcription history**: in-memory list of last N transcriptions accessible from the tray menu. Recovery option if paste goes to the wrong window.
- **PyInstaller packaging**: `pyinstaller --onefile --noconsole dictation/tray_app.py` to produce `WhisperDictation.exe` — no Python required on the Windows machine.
- **Custom vocabulary fine-tuning**: fine-tune `large-v3` on ~1–2 hours of the user's own recordings. More effective than `initial_prompt` for personal speaking patterns.

### Long-term
- **Voice commands**: trigger actions by spoken keyword ("open file", "new paragraph"). Requires a keyword detection layer before Whisper.
- **RAG-enhanced correction**: post-process transcripts against a personal dictionary of common phrases.
- **Auto-start on login**: register as a Windows startup entry so it launches automatically.

## Streaming Dictation Mode — Implemented

### How it works
Press Ctrl+Win+X to start streaming. Speak naturally — each phrase is detected by client-side VAD (webrtcvad), sent to `/transcribe-chunk`, and pasted at the cursor immediately. Press Ctrl+Win+X again to stop.

### VAD chunking
webrtcvad processes 20ms audio frames and detects speech/silence boundaries. A silence gap of ≥600ms (configurable via `silence_threshold_ms`) triggers a chunk emit. Each chunk is typically 2–8 seconds of continuous speech. This avoids the word-boundary-cutting problem that fixed-time chunking causes.

### Context carry-forward
Each chunk is transcribed with `initial_prompt` = the static vocabulary prompt + the last ~200 characters of accumulated text. This maintains vocabulary consistency ("Claude" stays "Claude"), language consistency (Chinese/English carries forward), and topic continuity.

### Server endpoint
`POST /transcribe-chunk` uses `beam_size=1` (configurable via `chunk_beam_size`) and `vad_filter=False` (client already did VAD). Per-chunk latency on RTX 4090: ~0.5-1.5 seconds.

### Mutual exclusion
File transcription (GUI) and streaming dictation cannot run simultaneously — they share the same Whisper model. The Application class enforces this via mode locking. The tray icon turns orange during file transcription to signal dictation is unavailable.

---

## Key Design Decisions (with rationale)

- **FastAPI over direct function calls**: model loads once into GPU VRAM at server startup (~30s). All subsequent requests cost only inference time (~1s). Without a persistent server, every dictation event would require a 30s model reload.
- **HTTP over other IPC**: WSL2↔Windows boundary makes Unix pipes and domain sockets awkward. WSL2 automatically forwards TCP ports, making `localhost:8765` accessible from Windows without configuration.
- **Name-based device selection over index**: sounddevice indices shift when USB devices are plugged/unplugged or virtual audio drivers (VB-Audio) are installed/uninstalled. Storing the device name and resolving to an index at runtime is stable.
- **WASAPI→MME fallback**: WASAPI gives lower latency and bypasses Windows audio mixing, but is susceptible to exclusive-mode conflicts. MME is universally compatible at the cost of slightly more latency (~20ms) — imperceptible for dictation.
- **`language: null` for bilingual**: forcing `language: "zh"` would transliterate English words into Chinese characters. Auto-detect correctly keeps English words in English even in predominantly Chinese audio.
- **webrtcvad over silero-vad**: webrtcvad is a ~100KB C extension. Silero-VAD would pull in PyTorch (~2GB) onto the Windows side just for silence detection — completely disproportionate for the task.
- **Clipboard save/restore over SendInput**: SendInput works for English but Chinese characters require IME simulation which is unreliable. Clipboard save/restore is simple, bilingual-safe, and only adds ~100ms overhead.
- **pystray run_detached + tkinter mainloop**: pystray's Win32 message pump runs in a daemon thread, while tkinter owns the main thread. Cross-thread communication goes through a queue.Queue polled by tkinter's after(100ms). This is the standard pattern for combining these two frameworks.
- **SSE for progress reporting**: The /transcribe/stream endpoint uses Server-Sent Events to yield progress as each Whisper segment completes. This leverages faster-whisper's lazy generator — no need for polling or estimation.
