# Whisper Audio Recognition — Project Notes for Claude

## What This Project Does

A personal speech-to-text system running locally on an RTX 4090. Two independent tools share one Whisper engine:

1. **Dictation** — press `Ctrl+Win+H`, speak, press again → text pasted at Windows cursor
2. **Transcription** — CLI tool that sends an audio file to the server and writes a transcript

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
└── transcription/cli.py       CLI client → POSTs audio → gets text

Windows (Python 3.13, native)
└── dictation/tray_app.py      system tray + Ctrl+Win+H hotkey + clipboard paste
```

WSL2 and Windows communicate via `localhost:8765` — WSL2 port forwarding handles this automatically.

**Why the split**: CUDA lives in WSL. System tray icons, global hotkeys (`Ctrl+Win+H`), and clipboard paste into Windows apps require native Windows APIs — they cannot be done from WSL.

## How to Run

```bash
# WSL — one-time setup
bash setup_wsl.sh

# WSL — every session (start the model server)
./start_server.sh

# WSL — transcribe an audio file
.venv/bin/python -m transcription.cli meeting.mp3 -o transcript.txt
.venv/bin/python -m transcription.cli meeting.mp3 --timestamps

# Windows PowerShell — one-time setup
pip install -r dictation\requirements_windows.txt

# Windows PowerShell — run dictation tray app
python dictation\tray_app.py
```

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
| `audio_device_name` | `"麦克风 (thinkplus meeting dock)"` | Matched by name, not index |
| `vad_filter` | `true` | Filters silence before inference |

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

## Known Limitations / Future Work

### Near-term improvements
- **Text injection method**: currently uses clipboard (Option A) — copies text to clipboard and simulates `Ctrl+V`. This overwrites the user's clipboard. A better approach is Option B: use `win32api.SendInput` to inject characters directly without touching the clipboard. Requires `pywin32` package.
- **Tray icon hidden by default**: Windows 11 hides new tray icons in the overflow (`^`) area. The user must manually drag the icon to the visible taskbar area to pin it. No code fix — Windows setting.
- **Microphone submenu rebuilds statically**: The microphone submenu in the tray is built once at startup. If the user plugs in a new device after startup, it won't appear until restart. Could be fixed by making the submenu a dynamic callable.

### Medium-term
- **Streaming transcription**: current flow waits for the full recording before transcribing. FastAPI's `StreamingResponse` + generator iteration (instead of `list()`) would allow words to appear live as you speak. Useful for long dictation sessions.
- **PyInstaller packaging**: `pyinstaller --onefile --noconsole dictation/tray_app.py` to produce `WhisperDictation.exe` — no Python required on the Windows machine.
- **Custom vocabulary fine-tuning**: fine-tune `large-v3` on ~1–2 hours of the user's own recordings to improve recognition of personal speaking patterns. More effective than `initial_prompt` for recurring vocabulary.

### Long-term (discussed but not designed)
- **Voice commands**: trigger actions by spoken keyword ("open file", "new paragraph") rather than just transcription. Requires a separate keyword detection layer before Whisper.
- **Personalization / RAG-enhanced correction**: post-process transcripts against a personal dictionary of common phrases. More practical than reinforcement learning for this use case.
- **Auto-start on login**: register `tray_app.py` as a Windows startup entry so it launches automatically.

## Key Design Decisions (with rationale)

- **FastAPI over direct function calls**: model loads once into GPU VRAM at server startup (~30s). All subsequent requests cost only inference time (~1s). Without a persistent server, every dictation event would require a 30s model reload.
- **HTTP over other IPC**: WSL2↔Windows boundary makes Unix pipes and domain sockets awkward. WSL2 automatically forwards TCP ports, making `localhost:8765` accessible from Windows without configuration.
- **Name-based device selection over index**: sounddevice indices shift when USB devices are plugged/unplugged or virtual audio drivers (VB-Audio) are installed/uninstalled. Storing the device name and resolving to an index at runtime is stable.
- **WASAPI→MME fallback**: WASAPI gives lower latency and bypasses Windows audio mixing, but is susceptible to exclusive-mode conflicts. MME is universally compatible at the cost of slightly more latency (~20ms) — imperceptible for dictation.
- **`language: null` for bilingual**: forcing `language: "zh"` would transliterate English words into Chinese characters. Auto-detect correctly keeps English words in English even in predominantly Chinese audio.
