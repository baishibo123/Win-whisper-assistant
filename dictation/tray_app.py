"""
Whisper Dictation — Windows System Tray App (entry point)
─────────────────────────────────────────────────────────
Run this on Windows (not in WSL):
    python dictation/tray_app.py

What happens:
  1. The WSL Whisper server is launched automatically (no visible terminal)
  2. A system tray icon appears (gray while loading, green when ready)
  3. Ctrl+Win+X toggles streaming dictation — speak and text appears at cursor
  4. Double-click the tray icon to open the file transcription GUI
  5. Right-click for microphone selection and other options

Architecture:
  Main thread   : tkinter mainloop (hidden root, GUI shown on demand)
  Background    : pystray icon, pynput hotkey, sounddevice, server subprocess
  Communication : queue.Queue polled by tkinter root.after(100ms)
"""

import signal
import sys
import tkinter as tk

# Ensure the project root is on sys.path so "dictation.*" imports work
# when running as: python dictation/tray_app.py from /mnt/e/whisper
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dictation.app import Application


def main() -> None:
    # tkinter root must be created on the main thread and before pystray
    root = tk.Tk()
    root.withdraw()  # start hidden — no window until user double-clicks tray

    app = Application(root)
    app.start()

    # Signal handlers for Ctrl+C and termination (won't catch Task Manager
    # kills, but atexit + startup stale-server check cover that case)
    def _signal_handler(signum, frame):
        app.shutdown()
        root.destroy()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("[main] Whisper Dictation running. Ctrl+Win+X to dictate.")
    print("[main] Double-click the tray icon to open the transcription window.")
    print("[main] Right-click the tray icon for options. Select Quit to exit.")

    # tkinter mainloop blocks the main thread, processing UI events
    # and our 100ms queue polling (started by app.start())
    root.mainloop()

    # After mainloop exits (user clicked Quit), clean up
    app.shutdown()


if __name__ == "__main__":
    main()
