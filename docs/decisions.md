# Architecture Decision Records

Non-obvious design decisions made during development, with context on what was considered and why each choice was made.

---

## ADR-001: FastAPI persistent server over per-call model loading

The Whisper large-v3 model occupies ~3 GB of GPU VRAM and takes ~30 seconds to load. We run a persistent FastAPI server that loads the model once at startup and keeps it warm. All clients (dictation, CLI, GUI) send HTTP requests to this server. The alternative — loading the model on every dictation hotkey press — would add 30 seconds of latency to every interaction, making real-time dictation impossible.

## ADR-002: HTTP over other IPC mechanisms (WSL↔Windows)

The system is split across WSL2 (GPU/model) and Windows (audio/UI). We use HTTP on localhost:8765 for communication. Alternatives considered: Unix domain sockets (not available across the WSL2 boundary), named pipes (complex cross-OS setup), shared memory (same issue). WSL2 automatically forwards TCP ports from the Linux guest to the Windows host, making localhost HTTP work transparently with zero configuration.

## ADR-003: Device name-based selection over index-based

Audio device indices from sounddevice (`sd.query_devices()`) are assigned by enumeration order and shift whenever USB devices are plugged/unplugged or virtual audio drivers change. We store the device name string in `config.json` and resolve it to an index at runtime via substring matching. This makes the saved preference stable across reboots and hardware changes. The tradeoff is fuzzy matching, which could theoretically match the wrong device if two devices have very similar names.

## ADR-004: WASAPI→MME→default fallback chain

WASAPI (Windows Audio Session API) provides the lowest latency and bypasses the Windows audio mixer, but it has two failure modes: (1) exclusive-mode conflicts when another app holds the device, and (2) strict sample rate requirements (only the hardware native rate). When WASAPI fails, we fall back to MME, which is an older API where Windows mediates all access and handles sample rate conversion. The ~20ms additional latency is imperceptible for dictation. We tried DirectSound as a middle ground but it offered no advantages over MME for our use case. System default is the final fallback.

## ADR-005: language: null for bilingual auto-detection

The user speaks Chinese with incidental English words, or pure Chinese, or pure English. Setting `language: "zh"` causes Whisper to transliterate English words into Chinese characters ("Claude" becomes "克劳德"). Setting `language: "en"` drops Chinese entirely. With `language: null`, Whisper auto-detects per 30-second segment, correctly preserving English words within Chinese speech. The tradeoff is slightly more compute per segment for the detection step, but this is negligible on the RTX 4090.

## ADR-006: webrtcvad over silero-vad for client-side VAD

Streaming dictation needs client-side Voice Activity Detection to split audio into chunks at silence boundaries. We chose webrtcvad (Google's WebRTC VAD, ~100KB C extension) over silero-vad (a neural network approach). Silero is more accurate but requires PyTorch (~2 GB install) on the Windows side — purely for detecting speech vs. silence, which webrtcvad handles adequately. Energy-based detection (RMS threshold) was also considered but is too sensitive to background noise levels and requires manual calibration.

## ADR-007: VAD chunking over fixed-time chunking

Whisper was trained on ~30-second segments. Fixed-time chunking (e.g., every 2 seconds) frequently cuts mid-word, destroying context and producing garbled output. VAD chunking waits for natural silence gaps (≥600ms configurable) before emitting a chunk, so each chunk is a complete phrase. Chunks are typically 2–8 seconds. Whisper internally pads short chunks to 30 seconds, so there's no accuracy penalty from the shorter duration. The tradeoff is variable latency — if the user doesn't pause, audio accumulates until they do.

## ADR-008: Clipboard save/restore over SendInput for text injection

Transcribed text needs to be injected at the user's cursor position. We use clipboard save/restore: save the current clipboard, copy the transcribed text, simulate Ctrl+V, then restore the original clipboard. The alternative — `win32api.SendInput` — sends individual keystrokes and avoids the clipboard entirely. However, SendInput works well for ASCII/English but Chinese characters require simulating IME (Input Method Editor) input, which is unreliable and varies across IME implementations. Since the user is bilingual, clipboard is the safer choice. The ~100ms overhead for save/restore is imperceptible.

## ADR-009: pystray run_detached + tkinter mainloop

The app needs both a system tray icon (pystray) and a GUI window (tkinter). Both frameworks have blocking event loops. We run pystray in a background daemon thread via `Icon.run_detached()` while tkinter owns the main thread with `root.mainloop()`. Cross-thread communication uses `queue.Queue` polled by tkinter's `root.after(100ms)`. The alternative — running tkinter in a thread — is unreliable because tkinter is not thread-safe on Windows and crashes unpredictably. Another alternative — replacing pystray with tkinter's own tray support — was rejected because tkinter's system tray integration on Windows is poor and undocumented.

## ADR-010: SSE streaming for transcription progress

The file transcription GUI needs a real-time progress bar. We use Server-Sent Events (SSE) via FastAPI's `StreamingResponse`. The server iterates faster-whisper's lazy segment generator and yields a JSON event after each segment completes. Progress is estimated as `segment.end / audio_duration`. Alternatives: (1) polling a status endpoint — adds complexity and latency; (2) WebSockets — overkill for unidirectional progress; (3) client-side estimation based on file size — inaccurate because Whisper processing time varies with speech density. SSE naturally fits the one-way server→client data flow and integrates cleanly with `requests.post(stream=True)`.

## ADR-011: webrtcvad-wheels over webrtcvad

The original `webrtcvad` PyPI package requires a C compiler to build from source. The user's Windows machine doesn't have Microsoft Visual C++ Build Tools installed (~4 GB). `webrtcvad-wheels` is a drop-in replacement that ships pre-compiled binary wheels for Windows, macOS, and Linux. Same API, same `import webrtcvad`, no build step. The tradeoff is depending on a third-party repackaging, but the package is widely used and the underlying C code is identical.

## ADR-012: Hidden WSL server subprocess via CREATE_NO_WINDOW

The tray app auto-launches the WSL Whisper server via `subprocess.Popen` with `creationflags=CREATE_NO_WINDOW`. Without this flag, a cmd.exe window flashes on screen every time the app starts. The server's stdout/stderr is captured via `subprocess.PIPE` and piped to the GUI log panel and an in-memory ring buffer. The user runs the tray app itself with `pythonw.exe` (the windowless Python interpreter) to avoid even the parent PowerShell window staying open. Together, the entire system runs with no visible terminal — only the tray icon and the GUI window on demand.

## ADR-013: Safe stale server cleanup via /health fingerprinting

When the tray app is killed via Task Manager, `atexit` handlers don't fire and the WSL uvicorn server is orphaned on its port. On next launch, we need to kill it — but we can't blindly kill whatever is on that port, because the user might have another service on the same port. We probe `GET /health` and check for our specific response shape (`{"status": "ok", "model_loaded": bool}`). If it matches, it's our stale server → safe to `pkill`. If the port responds with something else, it's another program → refuse to start, notify the user to free the port or change `server_port` in config.json. If connection refused, the port is free → proceed normally. This avoids both the "address already in use" crash and the risk of killing an unrelated service. Alternatives considered: PID file in WSL (fragile if WSL restarts, stale PID could match a different process), always killing by port number (dangerous), requiring manual cleanup (poor UX).
