"""
VAD-based audio chunker for streaming dictation.

Uses webrtcvad to detect speech/silence boundaries in real-time audio frames
from the sounddevice callback. When a complete speech chunk is detected
(speech followed by sufficient silence), the chunk is placed into a queue
for the sender thread to POST to the server.

webrtcvad was chosen over silero-vad because it's a ~100KB C extension,
while silero would pull in PyTorch (~2GB) just for silence detection.
"""

import collections
import queue
import struct
from typing import Optional

import numpy as np
import webrtcvad


class VADChunker:
    """
    State machine that receives raw PCM frames and emits complete speech chunks.

    States:
      IDLE       → waiting for speech
      SPEAKING   → speech detected, accumulating audio
      SILENCE    → speech stopped, waiting to see if silence is long enough to emit

    Parameters
    ----------
    sample_rate : int
        Must be 8000, 16000, 32000, or 48000 (webrtcvad requirement).
    aggressiveness : int
        webrtcvad aggressiveness 0-3. Higher = more aggressive filtering of
        non-speech. 2 is a good default for dictation.
    silence_threshold_ms : int
        How many ms of consecutive silence triggers a chunk emit (~600ms).
    frame_duration_ms : int
        Duration of each frame fed to webrtcvad (10, 20, or 30 ms).
    speech_pad_ms : int
        Extra audio to keep before speech onset for natural starts.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,
        silence_threshold_ms: int = 600,
        frame_duration_ms: int = 20,
        speech_pad_ms: int = 300,
    ):
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"webrtcvad requires 8/16/32/48 kHz, got {sample_rate}")

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.silence_threshold_ms = silence_threshold_ms

        # Number of samples per webrtcvad frame
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        # How many consecutive silent frames before we emit
        self._silence_frames_needed = silence_threshold_ms // frame_duration_ms

        # How many frames of pre-speech audio to keep
        self._pad_frames = speech_pad_ms // frame_duration_ms

        self._vad = webrtcvad.Vad(aggressiveness)
        self._chunk_queue: queue.Queue[np.ndarray] = queue.Queue()

        # Internal state
        self._state = "IDLE"    # IDLE, SPEAKING, SILENCE
        self._speech_frames: list[bytes] = []
        self._silence_count = 0
        # Ring buffer for pre-speech padding
        self._ring_buffer: collections.deque[bytes] = collections.deque(
            maxlen=self._pad_frames
        )
        # Buffer for incomplete frames from sounddevice callback
        self._leftover = b""

    @property
    def chunk_queue(self) -> queue.Queue:
        """Queue of complete speech chunks (numpy float32 arrays)."""
        return self._chunk_queue

    def feed(self, audio: np.ndarray) -> None:
        """
        Feed raw audio from sounddevice callback.

        audio : numpy float32 array, shape (N, 1) or (N,)
        """
        # Convert float32 [-1, 1] to int16 PCM bytes for webrtcvad
        mono = audio.flatten()
        pcm_int16 = (mono * 32767).astype(np.int16)
        raw = pcm_int16.tobytes()

        # Prepend any leftover bytes from previous call
        raw = self._leftover + raw
        frame_bytes = self.frame_size * 2  # 2 bytes per int16 sample

        offset = 0
        while offset + frame_bytes <= len(raw):
            frame = raw[offset : offset + frame_bytes]
            offset += frame_bytes
            self._process_frame(frame)

        # Save leftover for next call
        self._leftover = raw[offset:]

    def _process_frame(self, frame: bytes) -> None:
        """Process a single webrtcvad-sized frame."""
        is_speech = self._vad.is_speech(frame, self.sample_rate)

        if self._state == "IDLE":
            if is_speech:
                self._state = "SPEAKING"
                # Include pre-speech padding for natural onset
                self._speech_frames = list(self._ring_buffer)
                self._speech_frames.append(frame)
            else:
                self._ring_buffer.append(frame)

        elif self._state == "SPEAKING":
            self._speech_frames.append(frame)
            if not is_speech:
                self._state = "SILENCE"
                self._silence_count = 1
            # else: keep accumulating speech

        elif self._state == "SILENCE":
            self._speech_frames.append(frame)
            if is_speech:
                # Speech resumed before threshold — go back to SPEAKING
                self._state = "SPEAKING"
                self._silence_count = 0
            else:
                self._silence_count += 1
                if self._silence_count >= self._silence_frames_needed:
                    # Enough silence — emit the chunk
                    self._emit_chunk()

    def _emit_chunk(self) -> None:
        """Convert accumulated PCM frames to numpy and put on queue."""
        if not self._speech_frames:
            self._reset()
            return

        raw = b"".join(self._speech_frames)
        # Convert int16 PCM back to float32 for the WAV encoder
        pcm_int16 = np.frombuffer(raw, dtype=np.int16)
        audio_float = pcm_int16.astype(np.float32) / 32767.0
        self._chunk_queue.put(audio_float)
        self._reset()

    def _reset(self) -> None:
        """Reset state machine to IDLE."""
        self._state = "IDLE"
        self._speech_frames = []
        self._silence_count = 0
        self._ring_buffer.clear()

    def flush(self) -> Optional[np.ndarray]:
        """
        Flush any remaining speech when streaming stops.
        Returns the final chunk or None if nothing accumulated.
        """
        if self._speech_frames and self._state != "IDLE":
            raw = b"".join(self._speech_frames)
            pcm_int16 = np.frombuffer(raw, dtype=np.int16)
            audio_float = pcm_int16.astype(np.float32) / 32767.0
            self._reset()
            return audio_float
        self._reset()
        return None
