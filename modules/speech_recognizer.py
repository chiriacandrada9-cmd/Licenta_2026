"""
Offline speech-to-text with faster-whisper.

Loads a Whisper model and transcribes raw PCM audio buffers into text.
Supports Romanian and English.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

import config

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """
    Transcribe audio buffers offline using faster-whisper.

    The model is loaded lazily on first call to :meth:`transcribe` (or
    explicitly via :meth:`load_model`) because loading can take 2–10 s
    and downloads 150–500 MB on first run.
    """

    def __init__(
        self,
        model_size: str | None = None,
        language: str | None = None,
        device: str | None = None,
    ) -> None:
        voice_cfg = config.get("voice")
        self.model_size = model_size or voice_cfg["whisper_model"]
        self.language = language or voice_cfg["whisper_language"]
        self.device = device or voice_cfg["whisper_device"]
        self._model = None
        logger.info(
            "SpeechRecognizer configured - model=%s, lang=%s, device=%s",
            self.model_size, self.language, self.device,
        )

    def load_model(self) -> None:
        """Download (if needed) and load the Whisper model into memory."""
        if self._model is not None:
            return
        logger.info("Loading Whisper model '%s' (this may take a moment)...", self.model_size)
        t0 = time.monotonic()

        from faster_whisper import WhisperModel

        # Store models in a known location
        model_dir = str(config.MODEL_DIR)
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type="int8",  # Best for CPU
            download_root=model_dir,
            cpu_threads=min(4, os.cpu_count() or 4),  # Prevent thrashing
        )
        elapsed = time.monotonic() - t0
        logger.info("Whisper model loaded in %.1f s", elapsed)

    def transcribe(self, audio: bytes, sample_rate: int = 16000) -> str:
        """Transcrie audio PCM 16-bit in text. Returneaza string gol
        daca nu s-a recunoscut nimic."""
        if self._model is None:
            self.load_model()

        # Convert bytes -> float32 numpy array in [-1, 1]
        audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        t0 = time.monotonic()
        segments, info = self._model.transcribe(
            audio_np,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,  # Prevent hallucination loops
            temperature=0.0,                   # Deterministic decoding
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        text = " ".join(text_parts).strip()
        elapsed = time.monotonic() - t0
        logger.debug("Transcribed in %.2f s: '%s'", elapsed, text)
        return text
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)
    config.load()

    from modules.audio_capture import AudioCapture
    from modules.vad import VoiceActivityDetector

    print("Speak into your microphone.  Press Ctrl+C to stop.\n")

    audio = AudioCapture()
    vad = VoiceActivityDetector()
    asr = SpeechRecognizer()

    audio.start()
    try:
        while True:
            chunk = audio.get_chunk()
            if chunk is None:
                continue
            utterance = vad.feed(chunk)
            if utterance is not None:
                text = asr.transcribe(utterance)
                if text:
                    print(f">>> {text}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        audio.stop()
