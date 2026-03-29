"""
Server lifecycle manager — launches the WSL Whisper server as a hidden subprocess.

Handles:
  - Starting the server with no visible terminal window (CREATE_NO_WINDOW)
  - Capturing stdout/stderr for the GUI log panel
  - Health-check polling to track server state
  - Clean shutdown on app exit
"""

import atexit
import collections
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

import requests

# The project directory as seen from WSL
_WSL_PROJECT_DIR = "/mnt/e/whisper"


class ServerManager:
    """
    Manages the WSL Whisper server subprocess lifecycle.

    Attributes
    ----------
    state : str
        One of "stopped", "starting", "running", "error".
    logs : collections.deque
        Ring buffer of recent log lines (max 2000).
    """

    def __init__(
        self,
        port: int = 8765,
        on_state_change: Optional[Callable[[str], None]] = None,
        on_log_line: Optional[Callable[[str], None]] = None,
    ):
        self.port = port
        self.on_state_change = on_state_change
        self.on_log_line = on_log_line

        self.state = "stopped"
        self.logs: collections.deque = collections.deque(maxlen=2000)

        self._process: Optional[subprocess.Popen] = None
        self._health_thread: Optional[threading.Thread] = None
        self._log_threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Launch the WSL server as a hidden subprocess."""
        if self._process and self._process.poll() is None:
            return  # already running

        # Check for stale server from a previous crash
        port_status = self._check_port()
        if port_status == "whisper":
            # Our server is still running from a previous session — kill it
            self.logs.append(f"[server_manager] Stale Whisper server on port {self.port}, killing it...")
            if self.on_log_line:
                self.on_log_line(f"[server_manager] Stale Whisper server on port {self.port}, killing it...")
            self._kill_stale_server()
            time.sleep(1)  # brief wait for port to free up
        elif port_status == "other":
            # Something else is on this port — don't touch it
            msg = (
                f"Port {self.port} is in use by another program. "
                f"Free the port or change server_port in config.json."
            )
            self.logs.append(f"[server_manager] {msg}")
            if self.on_log_line:
                self.on_log_line(f"[server_manager] ERROR: {msg}")
            self._set_state("error")
            return
        # port_status == "free" → proceed normally

        self._stop_event.clear()
        self._set_state("starting")

        # Register atexit to clean up on semi-graceful exits (unhandled
        # exceptions, sys.exit). Does NOT fire on Task Manager kill, but
        # the stale server check above covers that on next launch.
        atexit.register(self.stop)

        cmd = [
            "wsl", "-d", "Ubuntu", "-e", "bash", "-c",
            f"cd {_WSL_PROJECT_DIR} && ./start_server.sh",
        ]

        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NO_WINDOW

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout
            creationflags=creation_flags,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Reader thread for stdout
        t = threading.Thread(target=self._read_output, daemon=True)
        t.start()
        self._log_threads = [t]

        # Health check poller
        self._health_thread = threading.Thread(target=self._poll_health, daemon=True)
        self._health_thread.start()

    def stop(self) -> None:
        """Shut down the server subprocess."""
        self._stop_event.set()

        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Fallback: kill uvicorn inside WSL directly
                try:
                    subprocess.run(
                        ["wsl", "-d", "Ubuntu", "-e", "bash", "-c",
                         "pkill -f 'uvicorn engine.server:app'"],
                        timeout=3,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                    )
                except Exception:
                    pass
                self._process.kill()

        self._process = None
        self._set_state("stopped")

    def _check_port(self) -> str:
        """
        Probe the port to see what's on it.
        Returns: "free", "whisper" (our stale server), or "other" (unknown program).
        """
        url = f"http://localhost:{self.port}/health"
        try:
            r = requests.get(url, timeout=2)
            data = r.json()
            # Our /health returns {"status": "ok", "model_loaded": bool}
            if r.status_code == 200 and "model_loaded" in data:
                return "whisper"
            return "other"
        except requests.ConnectionError:
            return "free"
        except Exception:
            # Port responds but not JSON / not our format
            return "other"

    def _kill_stale_server(self) -> None:
        """Kill a stale Whisper uvicorn process inside WSL."""
        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            subprocess.run(
                ["wsl", "-d", "Ubuntu", "-e", "bash", "-c",
                 "pkill -f 'uvicorn engine.server:app'"],
                timeout=5,
                creationflags=creation_flags,
            )
        except Exception as e:
            self.logs.append(f"[server_manager] Failed to kill stale server: {e}")

    def _read_output(self) -> None:
        """Read subprocess stdout line by line into the log buffer."""
        try:
            for line in self._process.stdout:
                line = line.rstrip("\n")
                self.logs.append(line)
                if self.on_log_line:
                    self.on_log_line(line)
                if self._stop_event.is_set():
                    break
        except Exception:
            pass

    def _poll_health(self) -> None:
        """Poll GET /health every 3 seconds to track server readiness."""
        url = f"http://localhost:{self.port}/health"
        was_running = False

        while not self._stop_event.is_set():
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200 and r.json().get("model_loaded"):
                    if self.state != "running":
                        self._set_state("running")
                    was_running = True
                else:
                    # Server responding but model not loaded yet
                    if self.state == "stopped":
                        self._set_state("starting")
            except requests.ConnectionError:
                if was_running:
                    self._set_state("stopped")
                # If never was running, stay in current state (starting)
            except Exception:
                pass

            # Check if process died unexpectedly
            if self._process and self._process.poll() is not None:
                exit_code = self._process.returncode
                if exit_code != 0 and self.state != "stopped":
                    self._set_state("error")
                elif self.state != "stopped":
                    self._set_state("stopped")
                break

            self._stop_event.wait(3)

    def _set_state(self, new_state: str) -> None:
        """Update state and fire callback if changed."""
        if self.state != new_state:
            old = self.state
            self.state = new_state
            self.logs.append(f"[server_manager] State: {old} → {new_state}")
            if self.on_state_change:
                self.on_state_change(new_state)
