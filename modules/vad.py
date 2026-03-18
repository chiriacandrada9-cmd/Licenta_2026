"""
Voice Activity Detection wrapper.

Uses webrtcvad to detect speech boundaries in a stream of PCM audio
chunks and emits complete utterances (with pre-padding for word onsets).
"""

from __future__ import annotations

import collections
import logging

import webrtcvad

import config

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Acumuleaza chunk-uri audio, detecteaza cand userul incepe si
    termina de vorbit, si returneaza pronuntarea completa.

    Args:
        aggressiveness: cat de agresiv filtreaza non-speech (0-3)
        sample_rate: frecventa audio (16000 recomandat)
        silence_duration_ms: cat timp de tacere = sfarsitul pronuntarii
        frame_duration_ms: durata fiecarui chunk (10, 20 sau 30ms)
    """

    def __init__(
        self,
        aggressiveness: int | None = None,
        sample_rate: int | None = None,
        silence_duration_ms: int | None = None,
        frame_duration_ms: int = 30,
    ) -> None:
        voice_cfg = config.get("voice")
        self.sample_rate = sample_rate or voice_cfg["sample_rate"]
        aggressiveness = aggressiveness if aggressiveness is not None else voice_cfg["vad_aggressiveness"]
        self.silence_duration_ms = silence_duration_ms or voice_cfg["silence_duration_ms"]
        self.frame_duration_ms = frame_duration_ms

        self._vad = webrtcvad.Vad(aggressiveness)

        # Number of silent frames that mark the end of an utterance
        self._silence_frames_needed = int(
            self.silence_duration_ms / self.frame_duration_ms
        )

        # Pre-roll: frames buffered *before* speech starts to avoid clipping
        pre_roll_ms = 300
        self._pre_roll_size = int(pre_roll_ms / self.frame_duration_ms)
        self._ring_buffer: collections.deque[bytes] = collections.deque(
            maxlen=self._pre_roll_size
        )

        # State
        self._active = False  # Currently collecting speech
        self._speech_buffer: list[bytes] = []
        self._silence_frames: int = 0

        # Minimum utterance length to emit (discard coughs / pops)
        self._min_utterance_ms = 500
        self._min_frames = int(self._min_utterance_ms / self.frame_duration_ms)

        logger.info(
            "VAD ready - aggressiveness=%d, silence=%d ms, pre-roll=%d ms",
            aggressiveness, self.silence_duration_ms, pre_roll_ms,
        )

    def feed(self, chunk: bytes) -> bytes | None:
        """
        Feed a single audio chunk and return either:
        - ``bytes`` - audio-ul complet al pronuntarii, sau
        - ``None``  - inca asculta / tacere.
        """
        is_speech = self._vad.is_speech(chunk, self.sample_rate)

        if not self._active:
            # -- IDLE state: waiting for speech --
            self._ring_buffer.append(chunk)
            if is_speech:
                self._active = True
                self._silence_frames = 0
                # Pre-pad with ring buffer
                self._speech_buffer = list(self._ring_buffer)
                self._speech_buffer.append(chunk)
                logger.debug("Speech started")
        else:
            # -- ACTIVE state: collecting speech --
            self._speech_buffer.append(chunk)
            if is_speech:
                self._silence_frames = 0
            else:
                self._silence_frames += 1
                if self._silence_frames >= self._silence_frames_needed:
                    # End of utterance
                    utterance = b"".join(self._speech_buffer)
                    self._reset()
                    # Discard very short utterances
                    total_frames = len(utterance) // (
                        2 * int(self.sample_rate * self.frame_duration_ms / 1000)
                    )
                    if total_frames < self._min_frames:
                        logger.debug("Utterance too short (%d frames) - discarded", total_frames)
                        return None
                    logger.debug(
                        "Speech ended - %d bytes (%.1f s)",
                        len(utterance),
                        len(utterance) / (2 * self.sample_rate),
                    )
                    return utterance
        return None

    def reset(self) -> None:
        """Manually reset state (e.g. when switching modes)."""
        self._reset()

    def _reset(self) -> None:
        self._active = False
        self._speech_buffer.clear()
        self._silence_frames = 0
        self._ring_buffer.clear()
