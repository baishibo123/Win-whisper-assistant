"""
Whisper Transcription CLI
─────────────────────────
Sends an audio file to the running Whisper server and writes the transcript.

How the CLI works (end to end):
  1. argparse reads your terminal arguments into a Namespace object
  2. The audio file is opened and POSTed to the FastAPI server as multipart form-data
  3. The server returns JSON: {"text": ..., "language": ..., "language_probability": ...}
  4. We format the result and write to stdout or a file

Usage examples:
  python -m transcription.cli meeting.mp3
  python -m transcription.cli meeting.mp3 -o transcript.txt
  python -m transcription.cli meeting.mp3 --timestamps
  python -m transcription.cli meeting.mp3 --timestamps -o transcript.txt
  python -m transcription.cli meeting.mp3 --language zh
  python -m transcription.cli meeting.mp3 --language en -o out.txt --timestamps
"""

import argparse
import json
import os
import sys

import requests

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
)


def _get_server_url() -> str:
    """
    Read server port from config.json.
    Allows override via environment variable: WHISPER_SERVER=http://host:port
    """
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            port = json.load(f).get("server_port", 8765)
    except Exception:
        port = 8765
    return os.environ.get("WHISPER_SERVER", f"http://localhost:{port}")


# ── Formatting ────────────────────────────────────────────────────────────────

def _format_timestamp(seconds: float) -> str:
    """Convert float seconds → HH:MM:SS.mmm  e.g. 3723.5 → 01:02:03.500"""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _format_result(text_data, timestamps: bool) -> str:
    """
    Turn the server's "text" field into a printable string.

    Without timestamps: text_data is a plain string → return as-is.
    With timestamps:    text_data is a list of dicts → format each line.

    Why the isinstance check?
    If the user requests --timestamps but there are no detected speech segments
    (e.g. silent audio), the server still returns a list — just an empty one.
    """
    if timestamps and isinstance(text_data, list):
        lines = [
            f"[{_format_timestamp(seg['start'])}]  {seg['text']}"
            for seg in text_data
        ]
        return "\n".join(lines)

    # Plain string (no timestamps) or unexpected shape — return as string
    if isinstance(text_data, list):
        return " ".join(seg["text"] for seg in text_data)
    return str(text_data)


# ── Core logic ────────────────────────────────────────────────────────────────

def transcribe_file(
    audio_path: str,
    timestamps: bool = False,
    language:   str  = None,
    output:     str  = None,
) -> None:
    """
    Main function: send audio to server, format result, write output.

    Parameters
    ----------
    audio_path : str   Path to audio file on disk.
    timestamps : bool  Whether to request timestamped segments.
    language   : str   Language code to force ("zh", "en") or None for auto.
    output     : str   Path to write transcript, or None to print to stdout.
    """
    server_url = _get_server_url()

    # ── Validate input ────────────────────────────────────────────────────────
    if not os.path.exists(audio_path):
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"File   : {audio_path}  ({file_size_mb:.1f} MB)", file=sys.stderr)
    print(f"Server : {server_url}", file=sys.stderr)
    print(f"Options: timestamps={timestamps}  language={language or 'auto'}", file=sys.stderr)
    print("Sending to server ...", file=sys.stderr)

    # ── Send HTTP request ─────────────────────────────────────────────────────
    #
    # multipart/form-data format:
    #   "audio"      → the file bytes  (UploadFile on server side)
    #   "timestamps" → "true"/"false"  (Form field on server side)
    #   "language"   → language code   (Form field on server side)
    #
    # requests library handles multipart encoding automatically when you
    # pass both files= and data= arguments together.
    #
    with open(audio_path, "rb") as f:
        try:
            response = requests.post(
                f"{server_url}/transcribe",
                files={"audio": (os.path.basename(audio_path), f)},
                data={
                    "timestamps": "true" if timestamps else "false",
                    "language":   language or "",
                },
                timeout=600,   # 10-minute timeout for long audio files
            )
            response.raise_for_status()   # raises on 4xx / 5xx status codes

        except requests.ConnectionError:
            print(
                f"\nError: cannot connect to {server_url}\n"
                 "Start the server first:  ./start_server.sh",
                file=sys.stderr,
            )
            sys.exit(1)

        except requests.Timeout:
            print("Error: request timed out. The audio file may be very long.", file=sys.stderr)
            sys.exit(1)

        except requests.HTTPError as e:
            print(f"Server returned an error: {e}", file=sys.stderr)
            sys.exit(1)

    # ── Parse response ────────────────────────────────────────────────────────
    result = response.json()

    if "error" in result:
        print(f"Transcription error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    lang = result.get("language", "?")
    prob = result.get("language_probability", 0.0)
    print(f"Detected: {lang}  (confidence {prob:.1%})", file=sys.stderr)

    # ── Format and write output ───────────────────────────────────────────────
    formatted = _format_result(result["text"], timestamps)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(formatted)
            f.write("\n")
        print(f"Saved  : {output}", file=sys.stderr)
    else:
        print(formatted)   # stdout — can be piped: python -m transcription.cli a.mp3 | grep ...


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    argparse turns terminal arguments into a Python Namespace.

    The "python -m transcription.cli" invocation works because:
      1. transcription/ has __init__.py → Python treats it as a package
      2. "-m transcription.cli" tells Python to run cli.py as __main__
      3. The if __name__ == "__main__" block at the bottom calls main()
    """
    parser = argparse.ArgumentParser(
        prog="whisper-transcribe",
        description="Transcribe audio files via the local Whisper server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "audio",
        help="Audio file to transcribe (mp3, wav, m4a, flac, ogg …)",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Write transcript to FILE instead of printing to stdout",
    )
    parser.add_argument(
        "-t", "--timestamps",
        action="store_true",
        help="Prefix each segment with its timestamp  [HH:MM:SS.mmm]",
    )
    parser.add_argument(
        "-l", "--language",
        metavar="CODE",
        help=(
            "Force language (zh, en, ja …). "
            "Omit for automatic detection — best for mixed Chinese/English."
        ),
    )

    args = parser.parse_args()
    transcribe_file(args.audio, args.timestamps, args.language, args.output)


if __name__ == "__main__":
    main()
