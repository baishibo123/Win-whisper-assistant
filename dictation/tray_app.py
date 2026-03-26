"""
Whisper Dictation — Windows System Tray App
────────────────────────────────────────────
Run this on Windows (not in WSL).

Hotkey : Ctrl+Win+H  →  start recording
         Ctrl+Win+H  →  stop + transcribe + paste at cursor

Tray menu:
  Start / Stop recording
  Microphone ► (submenu — lists all WASAPI input devices by name)
  Quit

Device selection is saved to config.json by name (not by number).
Names are stable across reboots and USB reconnects.
Numbers are NOT used — they shift whenever devices are added/removed.
"""

import io
import json
import os
import sys
import threading
import time

import numpy as np
import pyperclip
import requests
import sounddevice as sd
import soundfile as sf
import pystray
from PIL import Image, ImageDraw
from pynput import keyboard as pykbd

# ── Config path ───────────────────────────────────────────────────────────────

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
)

SAMPLE_RATE = 16000
CHANNELS    = 1


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"server_port": 8765}


def _save_config(cfg: dict) -> None:
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


# ── Device helpers ─────────────────────────────────────────────────────────────
#
# Why name-based and not index-based?
# sounddevice device indices are assigned in order of device enumeration.
# When you plug/unplug a USB device or install/uninstall a virtual audio driver,
# Windows re-enumerates and indices shift. A device that was #27 becomes #24.
# Device *names* come from the Windows audio registry and are stable.
#
# Why WASAPI only?
# Each physical microphone appears 3–4 times in Windows (MME, DirectSound,
# WASAPI, WDM-KS). WASAPI is the modern low-latency API. Filtering to WASAPI
# gives one entry per real microphone.

def get_wasapi_input_devices() -> list[tuple[str, int]]:
    """
    Return [(display_name, device_index), ...] for all WASAPI input devices.
    Falls back to all input devices if WASAPI is not available.
    """
    try:
        hostapis = sd.query_hostapis()
        devices  = sd.query_devices()

        wasapi_idx = next(
            (i for i, h in enumerate(hostapis) if "WASAPI" in h["name"]),
            None,
        )

        if wasapi_idx is not None:
            return [
                (d["name"], i)
                for i, d in enumerate(devices)
                if d["hostapi"] == wasapi_idx and d["max_input_channels"] > 0
            ]

        # WASAPI not found — fall back to any input device
        return [
            (d["name"], i)
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
    except Exception as e:
        print(f"[dictation] Could not query audio devices: {e}", file=sys.stderr)
        return []


def _get_devices_for_hostapi(api_substring: str) -> list[tuple[str, int]]:
    """Return input devices belonging to a specific host API (matched by substring)."""
    try:
        hostapis = sd.query_hostapis()
        devices  = sd.query_devices()
        api_idx  = next(
            (i for i, h in enumerate(hostapis) if api_substring in h["name"]),
            None,
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


def find_device_index(name: str | None, api: str = "WASAPI") -> int | None:
    """
    Resolve a device name to its current index within the given host API.
    Returns None (= system default) if name is None or not found.
    """
    if not name:
        return None

    for dev_name, idx in _get_devices_for_hostapi(api):
        if name.lower() in dev_name.lower() or dev_name.lower() in name.lower():
            return idx

    print(
        f"[dictation] WARNING: saved device '{name}' not found in {api} — "
        "using Windows default. Is the device connected?",
        file=sys.stderr,
    )
    return None


# ── Shared state ──────────────────────────────────────────────────────────────

_lock                = threading.Lock()
_is_recording        = False
_audio_frames:       list[np.ndarray] = []
_stream              = None
_tray_icon           = None
_active_device_name: str | None = None   # human-readable, kept in sync with config
_recording_samplerate: int = 16000        # updated at recording start to match device


# ── Icon images ───────────────────────────────────────────────────────────────

def _make_icon(fill: str) -> Image.Image:
    img  = Image.new("RGB", (64, 64), (32, 32, 32))   # RGB avoids RGBA alpha issues on Windows
    draw = ImageDraw.Draw(img)
    draw.ellipse([6, 6, 58, 58], fill=fill, outline="white", width=2)
    return img


ICON_IDLE      = _make_icon("#4CAF50")   # green
ICON_RECORDING = _make_icon("#F44336")   # red
ICON_WORKING   = _make_icon("#FF9800")   # orange


def _set_icon(state: str) -> None:
    if _tray_icon is None:
        return
    icons  = {"idle": ICON_IDLE, "recording": ICON_RECORDING, "working": ICON_WORKING}
    titles = {
        "idle":      f"Whisper Dictation — Idle  [{_active_device_name or 'default mic'}]  (Ctrl+Win+H)",
        "recording": "Whisper Dictation — Recording...  (Ctrl+Win+H to stop)",
        "working":   "Whisper Dictation — Transcribing...",
    }
    _tray_icon.icon  = icons.get(state, ICON_IDLE)
    _tray_icon.title = titles.get(state, "Whisper Dictation")


# ── Audio recording ───────────────────────────────────────────────────────────

def _audio_callback(indata, frames, time_info, status) -> None:
    with _lock:
        if _is_recording:
            _audio_frames.append(indata.copy())


def _start_recording() -> None:
    global _is_recording, _audio_frames, _stream

    with _lock:
        if _is_recording:
            return
        _audio_frames = []
        _is_recording = True

    _set_icon("recording")

    cfg       = _load_config()
    dev_name  = cfg.get("audio_device_name")
    dev_label = dev_name or "Windows default"

    global _recording_samplerate

    # Fallback chain: WASAPI → MME → system default
    # WASAPI gives the best quality but can fail if the device is held in
    # exclusive mode by another app, or if the driver rejects the stream.
    # MME is older but Windows mediates access and handles sample rate
    # conversion, making it much more tolerant. We try each in order.
    attempts = []
    if dev_name:
        wasapi_idx = find_device_index(dev_name, api="WASAPI")
        mme_idx    = find_device_index(dev_name, api="MME")
        if wasapi_idx is not None:
            attempts.append(("WASAPI", wasapi_idx))
        if mme_idx is not None:
            attempts.append(("MME",    mme_idx))
    attempts.append(("default", None))   # final fallback: let Windows decide

    last_error = None
    for api_label, device_idx in attempts:
        try:
            if device_idx is not None:
                native_rate = int(sd.query_devices(device_idx)["default_samplerate"])
            else:
                native_rate = SAMPLE_RATE   # 16000 Hz works for system default

            # MME lets Windows resample, so 16000 Hz works directly — no need
            # to record at the hardware rate and rely on ffmpeg downsampling.
            if api_label == "MME":
                native_rate = SAMPLE_RATE

            _recording_samplerate = native_rate
            _stream = sd.InputStream(
                samplerate = native_rate,
                channels   = CHANNELS,
                dtype      = "float32",
                device     = device_idx,
                callback   = _audio_callback,
            )
            _stream.start()
            print(f"[dictation] Recording started  [{dev_label}]  via {api_label}  @ {native_rate} Hz")
            return   # success — exit the fallback loop

        except sd.PortAudioError as e:
            last_error = e
            print(f"[dictation] {api_label} failed ({e}), trying next...", file=sys.stderr)
            if _stream:
                try:
                    _stream.close()
                except Exception:
                    pass
                _stream = None

    # All attempts exhausted
    print(f"[dictation] ERROR: could not open any audio input: {last_error}", file=sys.stderr)
    with _lock:
        _is_recording = False
    _set_icon("idle")


def _stop_and_transcribe() -> None:
    global _is_recording, _stream

    with _lock:
        if not _is_recording:
            return
        _is_recording = False
        frames = list(_audio_frames)

    if _stream:
        _stream.stop()
        _stream.close()
        _stream = None

    print(f"[dictation] Stopped — {len(frames)} chunks captured.")

    if not frames:
        _set_icon("idle")
        return

    _set_icon("working")

    cfg = _load_config()
    server_url = f"http://localhost:{cfg.get('server_port', 8765)}"

    try:
        audio = np.concatenate(frames, axis=0)
        buf   = io.BytesIO()
        sf.write(buf, audio, _recording_samplerate, format="WAV", subtype="PCM_16")
        buf.seek(0)

        resp = requests.post(
            f"{server_url}/transcribe",
            files = {"audio": ("recording.wav", buf, "audio/wav")},
            data  = {"timestamps": "false"},
            timeout = 60,
        )
        resp.raise_for_status()

        result = resp.json()
        text   = result.get("text", "").strip()
        lang   = result.get("language", "?")
        print(f"[dictation] [{lang}] {text[:100]}{'...' if len(text) > 100 else ''}")

        if text:
            _paste_text(text)
        else:
            print("[dictation] Empty result — check that the correct mic is selected.")

    except requests.ConnectionError:
        msg = f"Cannot connect to Whisper server at {server_url}\nIs ./start_server.sh running in WSL?"
        print(f"[dictation] ERROR: {msg}", file=sys.stderr)
        if _tray_icon:
            _tray_icon.notify(msg, "Connection Error")
    except Exception as e:
        print(f"[dictation] ERROR: {e}", file=sys.stderr)
        if _tray_icon:
            _tray_icon.notify(str(e), "Whisper Error")
    finally:
        _set_icon("idle")


def _paste_text(text: str) -> None:
    """Copy text to clipboard then simulate Ctrl+V at the active cursor."""
    pyperclip.copy(text)
    time.sleep(0.05)
    kb = pykbd.Controller()
    with kb.pressed(pykbd.Key.ctrl):
        kb.press('v')
        kb.release('v')


# ── Hotkey ────────────────────────────────────────────────────────────────────

def _on_hotkey() -> None:
    with _lock:
        recording = _is_recording
    if recording:
        threading.Thread(target=_stop_and_transcribe, daemon=True).start()
    else:
        threading.Thread(target=_start_recording,     daemon=True).start()


# ── Microphone submenu ────────────────────────────────────────────────────────

def _make_mic_menu() -> pystray.Menu:
    """
    Build the Microphone submenu dynamically each time the tray menu is opened.
    Each item shows the real device name. A checkmark appears next to the
    currently selected device.
    """
    cfg          = _load_config()
    current_name = cfg.get("audio_device_name")
    devices      = get_wasapi_input_devices()

    items = []

    # "Use Windows Default" option at the top
    is_default = current_name is None
    items.append(
        pystray.MenuItem(
            "(Use Windows Default)",
            _make_device_selector(None),
            checked = lambda item, d=is_default: d,
            radio   = True,
        )
    )

    # One entry per WASAPI input device
    for dev_name, _idx in devices:
        is_active = current_name is not None and (
            current_name.lower() in dev_name.lower()
            or dev_name.lower() in current_name.lower()
        )
        items.append(
            pystray.MenuItem(
                dev_name,
                _make_device_selector(dev_name),
                checked = lambda item, active=is_active: active,
                radio   = True,
            )
        )

    return pystray.Menu(*items)


def _make_device_selector(name: str | None):
    """
    Returns a callback that saves the chosen device name to config.json
    and updates the tray icon title.
    """
    def _select(icon, item):
        global _active_device_name
        cfg = _load_config()
        cfg["audio_device_name"] = name
        _save_config(cfg)
        _active_device_name = name
        label = name or "Windows Default"
        print(f"[dictation] Microphone set to: {label}")
        _set_icon("idle")   # refresh title to show new device name
    return _select


# ── Tray menu ─────────────────────────────────────────────────────────────────

def _menu_toggle(icon, item) -> None:
    _on_hotkey()


def _menu_quit(icon, item) -> None:
    global _is_recording
    with _lock:
        _is_recording = False
    if _stream:
        try:
            _stream.stop()
            _stream.close()
        except Exception:
            pass
    icon.stop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global _tray_icon, _active_device_name

    cfg = _load_config()
    _active_device_name = cfg.get("audio_device_name")
    server_url = f"http://localhost:{cfg.get('server_port', 8765)}"

    # Confirm server is reachable
    try:
        r = requests.get(f"{server_url}/health", timeout=2)
        print(f"[dictation] Server reachable: {r.json()}")
    except Exception:
        print(
            f"[dictation] WARNING: cannot reach server at {server_url}\n"
             "            Start ./start_server.sh in WSL first.",
            file=sys.stderr,
        )

    # Confirm active microphone
    dev_label = _active_device_name or "Windows Default"
    dev_idx   = find_device_index(_active_device_name)
    print(f"[dictation] Microphone : {dev_label}  (index={dev_idx})")

    # Register global hotkey  (<cmd> = Windows key in pynput)
    hotkey = pykbd.GlobalHotKeys({"<ctrl>+<cmd>+h": _on_hotkey})
    hotkey.start()
    print("[dictation] Hotkey registered: Ctrl+Win+H")

    # Build tray icon
    # NOTE: If the icon does not appear, look for the  ^  arrow in the
    # bottom-right corner of the taskbar → hidden icons overflow area.
    _tray_icon = pystray.Icon(
        name  = "whisper_dictation",
        icon  = ICON_IDLE,
        title = f"Whisper Dictation — Idle  [{dev_label}]  (Ctrl+Win+H)",
        menu  = pystray.Menu(
            pystray.MenuItem("Start / Stop recording  (Ctrl+Win+H)", _menu_toggle),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Microphone", _make_mic_menu()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", _menu_quit),
        ),
    )

    print("[dictation] Running. Right-click the tray icon to select microphone.")
    print("[dictation] If icon is not visible, click the  ^  arrow in the taskbar.")
    _tray_icon.run()
    hotkey.stop()


if __name__ == "__main__":
    main()
