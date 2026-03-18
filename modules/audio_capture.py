"""
Microphone audio streaming.

Continuously captures 16-bit PCM audio from the default microphone in a
background thread and exposes chunks via a queue for the VAD to consume.
"""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np
import sounddevice as sd

import config

logger = logging.getLogger(__name__)


class AudioCapture:
    """Stream audio de la microfonul implicit.

    Args:
        sample_rate: frecventa audio in Hz (implicit 16000)
        channels: numar de canale (1 = mono)
        block_duration_ms: durata unui bloc in ms (10, 20 sau 30 pt webrtcvad)
    """

    def __init__(
        self,
        sample_rate: int | None = None,
        channels: int = 1,
        block_duration_ms: int = 30,
    ) -> None:
        voice_cfg = config.get("voice")
        self.sample_rate = sample_rate or voice_cfg["sample_rate"]
        self.channels = channels
        self.block_size = int(self.sample_rate * block_duration_ms / 1000)

        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._stream: sd.InputStream | None = None
        self._running = False
        logger.info(
            "AudioCapture configured - %d Hz, %d ch, block=%d samples (%d ms)",
            self.sample_rate, self.channels, self.block_size, block_duration_ms,
        )

    def start(self) -> None:
        """Open the audio stream and begin capturing."""
        if self._running:
            logger.warning("AudioCapture already running")
            return
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.block_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("AudioCapture started")

    def stop(self) -> None:
        """Stop and close the audio stream."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Drain the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        logger.info("AudioCapture stopped")

    def get_chunk(self, timeout: float = 0.1) -> bytes | None:
        """
        Get the next audio chunk (blocking up to *timeout* seconds).

        Returns raw 16-bit PCM bytes or None on timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        return self._running

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        if status:
            logger.warning("Audio callback status: %s", status)
        if not self._running:
            return
        try:
            self._queue.put_nowait(indata.tobytes())
        except queue.Full:
            pass  # Drop the oldest frames silently
