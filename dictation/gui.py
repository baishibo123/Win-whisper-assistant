"""
Transcription GUI window — tkinter interface for file transcription.

Opened by double-clicking the system tray icon. Provides:
  - Audio file browser with output path selection
  - Parameter controls (language, timestamps)
  - Real-time progress bar fed by SSE from /transcribe/stream
  - Server log text area (continuously updated from server subprocess)
  - Completion popup with "Open in Explorer" option
  - X button minimizes to tray instead of closing

Layout:
  +----------------------------------------------------------+
  | Whisper Transcription                              [_][X] |
  +----------------------------------------------------------+
  | Audio File:  [____________________________] [Browse...]   |
  | Output Path: [____________________________] [Browse...]   |
  |                                                           |
  | Language: [Auto  v]    [x] Include Timestamps             |
  |                                                           |
  | [          Transcribe          ]                          |
  |                                                           |
  | Progress: [=========>                        ] 35%        |
  +----------------------------------------------------------+
  | Server Log:                                               |
  | +-------------------------------------------------------+ |
  | | [engine] Model loaded and ready.                      | |
  | | Segment 3: [1:00-1:30] 你好世界...                     | |
  | +-------------------------------------------------------+ |
  +----------------------------------------------------------+
"""

import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

import requests

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


_LANG_MAP = {"Auto": "", "Chinese": "zh", "English": "en"}

# Audio file extensions for the browse dialog
_AUDIO_FILETYPES = [
    ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg *.wma *.aac *.opus"),
    ("All files", "*.*"),
]


class TranscriptionWindow:
    """
    tkinter GUI for file transcription, shown as a Toplevel window.

    The window is never destroyed — it's withdrawn (hidden) when the user
    clicks X, and deiconified (shown) on tray double-click.
    """

    def __init__(self, app: "Application"):
        self.app = app
        self._transcribing = False

        # Build the window as a Toplevel (child of the hidden root)
        self.win = tk.Toplevel(app.root)
        self.win.title("Whisper Transcription")
        self.win.geometry("660x520")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_widgets()

    def _build_widgets(self) -> None:
        w = self.win

        # ── Row 0: Audio file ─────────────────────────────────────────────────
        tk.Label(w, text="Audio File:").grid(row=0, column=0, sticky="e", padx=8, pady=6)
        self.audio_var = tk.StringVar()
        tk.Entry(w, textvariable=self.audio_var, width=55).grid(
            row=0, column=1, sticky="ew", padx=4, pady=6
        )
        tk.Button(w, text="Browse...", command=self._browse_audio).grid(
            row=0, column=2, padx=8, pady=6
        )

        # ── Row 1: Output path ────────────────────────────────────────────────
        tk.Label(w, text="Output Path:").grid(row=1, column=0, sticky="e", padx=8, pady=6)
        self.output_var = tk.StringVar()
        tk.Entry(w, textvariable=self.output_var, width=55).grid(
            row=1, column=1, sticky="ew", padx=4, pady=6
        )
        tk.Button(w, text="Browse...", command=self._browse_output).grid(
            row=1, column=2, padx=8, pady=6
        )

        # ── Row 2: Options ────────────────────────────────────────────────────
        opts = tk.Frame(w)
        opts.grid(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=6)

        tk.Label(opts, text="Language:").pack(side="left")
        self.lang_var = tk.StringVar(value="Auto")
        ttk.Combobox(
            opts, textvariable=self.lang_var,
            values=list(_LANG_MAP.keys()),
            state="readonly", width=10,
        ).pack(side="left", padx=6)

        self.ts_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opts, text="Include Timestamps", variable=self.ts_var).pack(
            side="left", padx=20
        )

        # ── Row 3: Transcribe button ──────────────────────────────────────────
        self.btn = tk.Button(
            w, text="Transcribe", command=self._on_transcribe,
            height=2, bg="#4CAF50", fg="white", font=("Segoe UI", 10, "bold"),
        )
        self.btn.grid(row=3, column=0, columnspan=3, sticky="ew", padx=12, pady=6)

        # ── Row 4-5: Progress bar + label ─────────────────────────────────────
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(w, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky="ew", padx=12, pady=(6, 0))

        self.progress_label = tk.Label(w, text="", font=("Segoe UI", 9))
        self.progress_label.grid(row=5, column=0, columnspan=3, sticky="w", padx=14)

        # ── Row 6-7: Server log ───────────────────────────────────────────────
        tk.Label(w, text="Server Log:", font=("Segoe UI", 9, "bold")).grid(
            row=6, column=0, sticky="nw", padx=8, pady=(8, 0)
        )

        log_frame = tk.Frame(w)
        log_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=12, pady=(4, 10))

        self.log_text = tk.Text(
            log_frame, height=10, state="disabled", wrap="word",
            font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4",
        )
        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Make log area expand with window resize
        w.grid_columnconfigure(1, weight=1)
        w.grid_rowconfigure(7, weight=1)

        # Start hidden
        self.win.withdraw()

    # ── Show / hide ───────────────────────────────────────────────────────────

    def show(self) -> None:
        self.win.deiconify()
        self.win.lift()
        self.win.focus_force()

    def _on_close(self) -> None:
        """X button hides the window instead of destroying it."""
        self.win.withdraw()

    # ── File browsing ─────────────────────────────────────────────────────────

    def _browse_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=_AUDIO_FILETYPES,
        )
        if path:
            self.audio_var.set(path)
            # Auto-fill output path: same directory, .txt extension
            base = os.path.splitext(path)[0]
            self.output_var.set(base + ".txt")

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save Transcript As",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)

    # ── Transcription ─────────────────────────────────────────────────────────

    def _on_transcribe(self) -> None:
        audio_path = self.audio_var.get().strip()
        output_path = self.output_var.get().strip()

        if not audio_path:
            messagebox.showwarning("Missing Input", "Please select an audio file.")
            return
        if not os.path.exists(audio_path):
            messagebox.showerror("File Not Found", f"Cannot find:\n{audio_path}")
            return
        if not output_path:
            messagebox.showwarning("Missing Output", "Please specify an output path.")
            return

        # Try to acquire exclusive mode
        if not self.app.acquire_mode("transcribing_file"):
            messagebox.showinfo(
                "Busy",
                "Dictation is active. Stop dictation first (Ctrl+Win+X)."
            )
            return

        self._transcribing = True
        self.btn.config(state="disabled", text="Transcribing...")
        self.progress_var.set(0)
        self.progress_label.config(text="Starting...")
        self.app.tray.update_icon()

        threading.Thread(
            target=self._do_transcribe,
            args=(audio_path, output_path, self.ts_var.get(), self.lang_var.get()),
            daemon=True,
        ).start()

    def _do_transcribe(self, audio_path: str, output_path: str,
                       timestamps: bool, language: str) -> None:
        """Background thread: streams transcription via SSE, posts updates to ui_queue."""
        cfg = _load_config()
        server_url = f"http://localhost:{cfg.get('server_port', 8765)}"
        lang_code = _LANG_MAP.get(language, "")
        uiq = self.app.ui_queue

        try:
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            uiq.put(("log", f"[gui] Sending {os.path.basename(audio_path)} ({file_size_mb:.1f} MB) ..."))

            with open(audio_path, "rb") as f:
                resp = requests.post(
                    f"{server_url}/transcribe/stream",
                    files={"audio": (os.path.basename(audio_path), f)},
                    data={
                        "timestamps": "true" if timestamps else "false",
                        "language": lang_code,
                    },
                    timeout=600,
                    stream=True,
                )
                resp.raise_for_status()

                full_result = None
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    if event["type"] == "progress":
                        uiq.put(("progress", event["progress_pct"]))
                        seg_text = event["text"][:60]
                        uiq.put(("log",
                            f"[gui] Segment {event['segment_index']}: "
                            f"[{event['start']:.1f}s - {event['end']:.1f}s] {seg_text}"
                        ))
                    elif event["type"] == "complete":
                        full_result = event
                    elif event["type"] == "error":
                        uiq.put(("log", f"[gui] ERROR: {event['error']}"))

            if full_result:
                # Format and write output
                text_data = full_result["text"]
                if timestamps and isinstance(text_data, list):
                    lines = []
                    for seg in text_data:
                        start = seg["start"]
                        h, m, s = int(start // 3600), int((start % 3600) // 60), start % 60
                        ts = f"{h:02d}:{m:02d}:{s:06.3f}"
                        lines.append(f"[{ts}]  {seg['text']}")
                    formatted = "\n".join(lines)
                elif isinstance(text_data, list):
                    formatted = " ".join(seg["text"] for seg in text_data)
                else:
                    formatted = str(text_data)

                with open(output_path, "w", encoding="utf-8") as out:
                    out.write(formatted)
                    out.write("\n")

                lang = full_result.get("language", "?")
                prob = full_result.get("language_probability", 0)
                uiq.put(("log", f"[gui] Done! Language: {lang} ({prob:.0%}). Saved to: {output_path}"))
                uiq.put(("progress", 100))
                uiq.put(("transcribe_done", output_path))
            else:
                uiq.put(("log", "[gui] No result received from server."))

        except requests.ConnectionError:
            uiq.put(("log", f"[gui] ERROR: cannot connect to {server_url}. Is the server running?"))
        except Exception as e:
            uiq.put(("log", f"[gui] ERROR: {e}"))
        finally:
            self.app.release_mode()
            # Re-enable button on the main thread
            self.app.root.after(0, self._reset_ui)

    # ── UI updates (called on main thread via ui_queue or root.after) ─────────

    def update_progress(self, pct: float) -> None:
        self.progress_var.set(pct)
        self.progress_label.config(text=f"{pct:.0f}%")

    def append_log(self, line: str) -> None:
        self.log_text.config(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")  # auto-scroll
        self.log_text.config(state="disabled")

    def on_server_state(self, state: str) -> None:
        self.append_log(f"[server] State: {state}")

    def on_transcription_done(self, output_path: str) -> None:
        result = messagebox.askyesno(
            "Transcription Complete",
            f"Transcript saved to:\n{output_path}\n\nOpen in Explorer?",
            parent=self.win,
        )
        if result:
            # Open Explorer with the file selected
            subprocess.Popen(f'explorer /select,"{output_path}"')

    def _reset_ui(self) -> None:
        self._transcribing = False
        self.btn.config(state="normal", text="Transcribe")
        self.progress_label.config(text="")
