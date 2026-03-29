"""
Tray icon manager — pystray integration for the Whisper dictation app.

Handles:
  - System tray icon with color-coded status (gray/green/red/orange)
  - Right-click context menu with microphone selection
  - Double-click to open the transcription GUI window
  - Hotkey registration (Ctrl+Win+X) via pynput
"""

import json
import os
import sys
import threading
from typing import TYPE_CHECKING

import pystray
import sounddevice as sd
from PIL import Image, ImageDraw
from pynput import keyboard as pykbd

if TYPE_CHECKING:
    from dictation.app import Application

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
)


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"server_port": 8765}


def _save_config(cfg: dict) -> None:
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


# ── Icon images ───────────────────────────────────────────────────────────────

def _make_icon(fill: str) -> Image.Image:
    img = Image.new("RGB", (64, 64), (32, 32, 32))
    draw = ImageDraw.Draw(img)
    draw.ellipse([6, 6, 58, 58], fill=fill, outline="white", width=2)
    return img


ICON_GRAY      = _make_icon("#808080")   # server down / starting
ICON_IDLE      = _make_icon("#4CAF50")   # green — idle, ready
ICON_RECORDING = _make_icon("#F44336")   # red — streaming dictation
ICON_WORKING   = _make_icon("#FF9800")   # orange — file transcription


# ── Device helpers (carried over from original tray_app.py) ───────────────────

def get_wasapi_input_devices() -> list[tuple[str, int]]:
    try:
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()
        wasapi_idx = next(
            (i for i, h in enumerate(hostapis) if "WASAPI" in h["name"]), None
        )
        if wasapi_idx is not None:
            return [
                (d["name"], i)
                for i, d in enumerate(devices)
                if d["hostapi"] == wasapi_idx and d["max_input_channels"] > 0
            ]
        return [
            (d["name"], i)
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
    except Exception as e:
        print(f"[tray] Could not query audio devices: {e}", file=sys.stderr)
        return []


class TrayManager:
    """Wraps pystray icon lifecycle and menu building."""

    def __init__(self, app: "Application"):
        self.app = app
        self._icon: pystray.Icon | None = None
        self._hotkey: pykbd.GlobalHotKeys | None = None

        cfg = _load_config()
        self._active_device_name = cfg.get("audio_device_name")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start pystray in a background thread and register the hotkey."""
        self._icon = pystray.Icon(
            name="whisper_dictation",
            icon=ICON_GRAY,  # start gray until server is confirmed running
            title=self._make_title(),
            menu=self._build_menu(),
        )
        # run_detached() starts the Win32 message pump in a daemon thread
        self._icon.run_detached()

        self._hotkey = pykbd.GlobalHotKeys({
            "<ctrl>+<cmd>+x": self.app.on_hotkey,
        })
        self._hotkey.start()
        print("[tray] Hotkey registered: Ctrl+Win+X")

    def stop(self) -> None:
        """Stop the tray icon and hotkey listener."""
        if self._icon:
            self._icon.stop()
        if self._hotkey:
            self._hotkey.stop()

    def update_icon(self) -> None:
        """Refresh icon color and tooltip based on current app + server state."""
        if not self._icon:
            return

        server_state = self.app.server.state
        app_mode = self.app.mode

        if server_state != "running":
            self._icon.icon = ICON_GRAY
        elif app_mode == "streaming":
            self._icon.icon = ICON_RECORDING
        elif app_mode == "transcribing_file":
            self._icon.icon = ICON_WORKING
        else:
            self._icon.icon = ICON_IDLE

        self._icon.title = self._make_title()

    def notify(self, message: str, title: str = "Whisper Dictation") -> None:
        """Show a system notification via pystray."""
        if self._icon:
            self._icon.notify(message, title)

    # ── Menu ──────────────────────────────────────────────────────────────────

    def _build_menu(self) -> pystray.Menu:
        return pystray.Menu(
            pystray.MenuItem("Open Transcription Window", self._on_open_window, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Start / Stop Dictation  (Ctrl+Win+X)", self._on_toggle),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Microphone", self._make_mic_menu()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._on_quit),
        )

    def _make_mic_menu(self) -> pystray.Menu:
        cfg = _load_config()
        current_name = cfg.get("audio_device_name")
        devices = get_wasapi_input_devices()
        items = []

        is_default = current_name is None
        items.append(
            pystray.MenuItem(
                "(Use Windows Default)",
                self._make_device_selector(None),
                checked=lambda item, d=is_default: d,
                radio=True,
            )
        )

        for dev_name, _idx in devices:
            is_active = current_name is not None and (
                current_name.lower() in dev_name.lower()
                or dev_name.lower() in current_name.lower()
            )
            items.append(
                pystray.MenuItem(
                    dev_name,
                    self._make_device_selector(dev_name),
                    checked=lambda item, active=is_active: active,
                    radio=True,
                )
            )

        return pystray.Menu(*items)

    def _make_device_selector(self, name: str | None):
        def _select(icon, item):
            cfg = _load_config()
            cfg["audio_device_name"] = name
            _save_config(cfg)
            self._active_device_name = name
            print(f"[tray] Microphone set to: {name or 'Windows Default'}")
            self.update_icon()
        return _select

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_open_window(self, icon, item) -> None:
        """Fired on tray icon double-click. Posts to ui_queue for main thread."""
        self.app.show_gui()

    def _on_toggle(self, icon, item) -> None:
        self.app.on_hotkey()

    def _on_quit(self, icon, item) -> None:
        self.app.shutdown()
        # Schedule tkinter destroy on main thread
        self.app.root.after(0, self.app.root.destroy)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_title(self) -> str:
        server = self.app.server.state
        mode = self.app.mode
        mic = self._active_device_name or "default mic"

        if server != "running":
            status = f"Server {server}"
        elif mode == "streaming":
            status = "Recording... (Ctrl+Win+X to stop)"
        elif mode == "transcribing_file":
            status = "Transcribing file..."
        else:
            status = "Idle"

        return f"Whisper Dictation — {status}  [{mic}]"
