"""
Application — central coordinator for all dictation components.

Owns shared state, enforces mutual exclusion between streaming dictation
and file transcription, and wires together the tray icon, GUI, server
manager, and streaming engine.
"""

import json
import os
import queue
import threading
import tkinter as tk
from typing import Optional

from dictation.clipboard import paste_with_restore
from dictation.server_manager import ServerManager
from dictation.streaming import StreamingDictation
from dictation.tray import TrayManager

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
)


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"server_port": 8765}


class Application:
    """
    Single entry point that coordinates all subsystems.

    Threading model:
      Main thread  : tkinter mainloop
      Thread 1     : pystray (run_detached)
      Thread 2     : pynput GlobalHotKeys
      Thread 3     : sounddevice audio callback
      Thread 4     : VAD chunk sender
      Thread 5     : Server subprocess log reader
      Thread 6     : Health check poller

    Cross-thread communication goes through self.ui_queue, polled by
    tkinter's after() mechanism every 100ms.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.cfg = _load_config()
        self.port = self.cfg.get("server_port", 8765)

        # Queue for cross-thread → tkinter communication
        self.ui_queue: queue.Queue = queue.Queue()

        # Mode management: "idle", "streaming", "transcribing_file"
        self._mode = "idle"
        self._mode_lock = threading.Lock()

        # Server manager — auto-launches WSL server
        self.server = ServerManager(
            port=self.port,
            on_state_change=self._on_server_state_change,
            on_log_line=self._on_server_log,
        )

        # Tray manager — pystray icon + menu
        self.tray = TrayManager(self)

        # Streaming dictation engine
        self.streamer = StreamingDictation(self)

        # GUI window — created lazily on first double-click
        self.gui = None  # type: Optional[object]  # avoid circular import

    # ── Mode management ───────────────────────────────────────────────────────

    def acquire_mode(self, new_mode: str) -> bool:
        """Atomically switch from idle to new_mode. Returns False if busy."""
        with self._mode_lock:
            if self._mode != "idle":
                return False
            self._mode = new_mode
            return True

    def release_mode(self) -> None:
        """Return to idle mode."""
        with self._mode_lock:
            self._mode = "idle"
        self.tray.update_icon()

    @property
    def mode(self) -> str:
        return self._mode

    # ── Hotkey handler ────────────────────────────────────────────────────────

    def on_hotkey(self) -> None:
        """Called when Ctrl+Win+X is pressed. Toggles streaming dictation."""
        if self._mode == "streaming":
            self.streamer.stop()
            self.release_mode()
        elif self.acquire_mode("streaming"):
            self.tray.update_icon()
            self.streamer.start()
        else:
            # File transcription in progress — ignore hotkey
            self.tray.notify("File transcription in progress — dictation unavailable.")

    # ── Server callbacks (called from background threads) ─────────────────────

    def _on_server_state_change(self, new_state: str) -> None:
        self.ui_queue.put(("server_state", new_state))

    def _on_server_log(self, line: str) -> None:
        self.ui_queue.put(("log", line))

    # ── UI queue polling ──────────────────────────────────────────────────────

    def start_queue_polling(self) -> None:
        """Start the 100ms tkinter polling loop for cross-thread messages."""
        self._poll_queue()

    def _poll_queue(self) -> None:
        while not self.ui_queue.empty():
            try:
                msg = self.ui_queue.get_nowait()
            except queue.Empty:
                break

            kind = msg[0]
            if kind == "server_state":
                self.tray.update_icon()
                if self.gui:
                    self.gui.on_server_state(msg[1])
            elif kind == "log":
                if self.gui:
                    self.gui.append_log(msg[1])
            elif kind == "progress":
                if self.gui:
                    self.gui.update_progress(msg[1])
            elif kind == "transcribe_done":
                if self.gui:
                    self.gui.on_transcription_done(msg[1])
            elif kind == "show_window":
                self._ensure_gui()
                self.gui.show()

        self.root.after(100, self._poll_queue)

    # ── GUI lifecycle ─────────────────────────────────────────────────────────

    def _ensure_gui(self) -> None:
        """Create the GUI window on first use (avoids circular import)."""
        if self.gui is None:
            from dictation.gui import TranscriptionWindow
            self.gui = TranscriptionWindow(self)
            # Dump existing server logs into the GUI
            for line in self.server.logs:
                self.gui.append_log(line)

    def show_gui(self) -> None:
        """Called from tray double-click (via ui_queue)."""
        self.ui_queue.put(("show_window",))

    # ── Startup / shutdown ────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch server, start tray, begin polling."""
        self.server.start()
        self.tray.start()
        self.start_queue_polling()

    def shutdown(self) -> None:
        """Clean shutdown of all subsystems."""
        if self._mode == "streaming":
            self.streamer.stop()
        self.tray.stop()
        self.server.stop()
