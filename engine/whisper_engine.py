"""
WhisperEngine — wraps faster-whisper model with project config.

Responsibilities:
  - Load the model once at startup (keeps it in GPU VRAM)
  - Accept an audio file path, return transcribed text
  - Handle language auto-detection (supports Chinese/English mixing)
  - Apply initial_prompt from config to improve recognition of custom vocab
"""

import json
import os
from typing import Generator, Optional, Union

from faster_whisper import WhisperModel

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
)


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


class WhisperEngine:
    """
    Wraps the faster-whisper model.

    Why a class and not a plain function?
    The model object (3 GB in VRAM) must persist between calls.
    A class instance holds it alive as long as the server is running.
    """

    def __init__(self):
        cfg = load_config()
        model_name   = cfg.get("model",        "large-v3")
        device       = cfg.get("device",       "cuda")
        compute_type = cfg.get("compute_type", "float16")

        print(f"[engine] Loading {model_name} on {device} ({compute_type}) ...")
        self.model  = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.config = cfg
        print("[engine] Model loaded and ready.")

    # ──────────────────────────────────────────────────────────────────────────
    def transcribe(
        self,
        audio_path: str,
        timestamps: bool = False,
        language: Optional[str] = None,
    ) -> dict:
        """
        Transcribe an audio file and return a dict:
            {
                "text":                 str  | list[dict]   (list when timestamps=True)
                "language":             str                  detected language code
                "language_probability": float                0–1 confidence
            }

        Parameters
        ----------
        audio_path : str
            Path to any audio format faster-whisper can read (wav, mp3, m4a, flac …)
        timestamps : bool
            False  → "text" is a plain string
            True   → "text" is a list of {"start": float, "end": float, "text": str}
        language : str | None
            Force a language ("zh", "en", …).
            None = auto-detect per segment — best for Chinese/English mixed speech.
        """
        cfg            = self.config
        initial_prompt = cfg.get("initial_prompt") or None
        lang           = language or cfg.get("language") or None   # None = auto
        beam_size      = cfg.get("beam_size", 5)

        # faster-whisper returns a lazy GENERATOR for segments.
        # We must call list() to consume it before we do anything else
        # (e.g. delete the temp file), otherwise the generator silently yields nothing.
        segments_gen, info = self.model.transcribe(
            audio_path,
            language=lang,
            initial_prompt=initial_prompt,
            beam_size=beam_size,
            vad_filter=True,                          # skip silent gaps automatically
            vad_parameters={"min_silence_duration_ms": 500},
        )
        segments = list(segments_gen)                 # ← consume generator NOW

        if timestamps:
            text_result: Union[str, list] = [
                {
                    "start": round(s.start, 2),
                    "end":   round(s.end,   2),
                    "text":  s.text.strip(),
                }
                for s in segments
            ]
        else:
            text_result = " ".join(s.text.strip() for s in segments)

        return {
            "text":                 text_result,
            "language":             info.language,
            "language_probability": round(info.language_probability, 3),
        }

    # ──────────────────────────────────────────────────────────────────────────
    def transcribe_chunk(
        self,
        audio_path: str,
        prior_context: str = "",
    ) -> dict:
        """
        Fast transcription for streaming dictation — one VAD-detected speech chunk.

        Uses beam_size=1 and skips server-side VAD (client already split on silence).
        prior_context is the last ~200 chars of the accumulated transcript, passed as
        initial_prompt so Whisper maintains vocabulary/language consistency across chunks.
        """
        cfg  = self.config
        lang = cfg.get("language") or None
        chunk_beam = cfg.get("chunk_beam_size", 1)

        # Combine the static vocabulary prompt with the rolling context
        static_prompt = cfg.get("initial_prompt") or ""
        prompt = f"{static_prompt} {prior_context}".strip() or None

        segments_gen, info = self.model.transcribe(
            audio_path,
            language=lang,
            initial_prompt=prompt,
            beam_size=chunk_beam,
            vad_filter=False,       # client already did VAD — don't re-filter
        )
        segments = list(segments_gen)
        text = " ".join(s.text.strip() for s in segments)

        return {
            "text":                 text,
            "language":             info.language,
            "language_probability": round(info.language_probability, 3),
        }

    # ──────────────────────────────────────────────────────────────────────────
    def transcribe_streaming(
        self,
        audio_path: str,
        timestamps: bool = False,
        language: Optional[str] = None,
    ) -> Generator[dict, None, None]:
        """
        Generator that yields progress events as each segment completes.

        Used by the /transcribe/stream SSE endpoint so the GUI can show a
        real progress bar based on how far into the audio we've transcribed.

        Yields dicts of two types:
          {"type": "progress", "segment_index": int, "start": float, "end": float,
           "text": str, "progress_pct": float, "audio_duration": float}
          {"type": "complete", "text": str|list, "language": str,
           "language_probability": float}
        """
        cfg            = self.config
        initial_prompt = cfg.get("initial_prompt") or None
        lang           = language or cfg.get("language") or None
        beam_size      = cfg.get("beam_size", 5)

        segments_gen, info = self.model.transcribe(
            audio_path,
            language=lang,
            initial_prompt=initial_prompt,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        audio_duration = info.duration
        all_segments = []

        for seg in segments_gen:
            all_segments.append(seg)
            progress = min(99.0, (seg.end / audio_duration) * 100) if audio_duration > 0 else 0
            yield {
                "type":           "progress",
                "segment_index":  len(all_segments),
                "start":          round(seg.start, 2),
                "end":            round(seg.end, 2),
                "text":           seg.text.strip(),
                "progress_pct":   round(progress, 1),
                "audio_duration": round(audio_duration, 2),
            }

        # Final result
        if timestamps:
            text_result: Union[str, list] = [
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in all_segments
            ]
        else:
            text_result = " ".join(s.text.strip() for s in all_segments)

        yield {
            "type":                 "complete",
            "text":                 text_result,
            "language":             info.language,
            "language_probability": round(info.language_probability, 3),
        }
