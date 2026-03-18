"""
tests/test_gestures.py - Unit tests for modules/gestures.py

Tests gesture classification using synthetic landmark arrays (no webcam
required).  Each static gesture test calls recognize() multiple times
to satisfy the stability buffer requirement.
"""

import pytest
import config
from modules.hand_tracker import (
    HandData,
    WRIST, THUMB_TIP, THUMB_IP, THUMB_MCP,
    INDEX_TIP, INDEX_PIP, INDEX_MCP,
    MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP,
    RING_TIP, RING_PIP, RING_MCP,
    PINKY_TIP, PINKY_PIP, PINKY_MCP,
)
from modules.gestures import GestureRecognizer, GestureType

config.load()

# Number of times to call recognize() to fill the stability buffer
# (must be >= gestures.stability_frames from settings)
STABILITY_REPS = 5


def _make_landmarks(
    thumb_ext=False,
    index_ext=False,
    middle_ext=False,
    ring_ext=False,
    pinky_ext=False,
    pinch_distance=0.1,
) -> list[tuple[float, float, float]]:
    """
    Create a synthetic set of 21 landmarks with the specified finger
    states.  Extended fingers have tip.y < pip.y; curled fingers have
    tip.y > pip.y.  Separation is large enough to exceed the hysteresis
    margin (0.03).
    """
    lm = [(0.5, 0.5, 0.0)] * 21

    # Wrist
    lm[WRIST] = (0.5, 0.9, 0.0)

    # -- Thumb (x-based for right hand: extended = tip.x < pip.x) --
    if thumb_ext:
        lm[THUMB_MCP] = (0.45, 0.7, 0.0)
        lm[THUMB_IP] = (0.40, 0.6, 0.0)
        lm[THUMB_TIP] = (0.30, 0.55, 0.0)  # Well left of IP for hysteresis
    else:
        lm[THUMB_MCP] = (0.45, 0.7, 0.0)
        lm[THUMB_IP] = (0.40, 0.6, 0.0)
        lm[THUMB_TIP] = (0.45, 0.75, 0.0)  # Right of IP = curled

    # -- Index --
    lm[INDEX_MCP] = (0.50, 0.7, 0.0)
    lm[INDEX_PIP] = (0.50, 0.55, 0.0)
    if index_ext:
        lm[INDEX_TIP] = (0.50 + pinch_distance / 2, 0.25, 0.0)  # Well above PIP
    else:
        lm[INDEX_TIP] = (0.55, 0.70, 0.0)  # Well below PIP

    # -- Middle --
    lm[MIDDLE_MCP] = (0.55, 0.7, 0.0)
    lm[MIDDLE_PIP] = (0.55, 0.55, 0.0)
    if middle_ext:
        lm[MIDDLE_TIP] = (0.55, 0.25, 0.0)
    else:
        lm[MIDDLE_TIP] = (0.55, 0.70, 0.0)

    # -- Ring --
    lm[RING_MCP] = (0.60, 0.7, 0.0)
    lm[RING_PIP] = (0.60, 0.55, 0.0)
    if ring_ext:
        lm[RING_TIP] = (0.60, 0.25, 0.0)
    else:
        lm[RING_TIP] = (0.60, 0.70, 0.0)

    # -- Pinky --
    lm[PINKY_MCP] = (0.65, 0.7, 0.0)
    lm[PINKY_PIP] = (0.65, 0.55, 0.0)
    if pinky_ext:
        lm[PINKY_TIP] = (0.65, 0.25, 0.0)
    else:
        lm[PINKY_TIP] = (0.65, 0.70, 0.0)

    return lm


def _hand(lm) -> HandData:
    return HandData(
        landmarks=lm,
        handedness="Right",
        pixel_landmarks=[(int(x * 640), int(y * 480)) for x, y, z in lm],
    )


def _recognize_stable(recognizer, lm, reps=STABILITY_REPS):
    """Call recognize() multiple times to fill the stability buffer."""
    result = None
    hand = _hand(lm)
    for _ in range(reps):
        result = recognizer.recognize(hand)
    return result


@pytest.fixture
def recognizer():
    return GestureRecognizer()


class TestStaticGestures:
    def test_fist(self, recognizer):
        lm = _make_landmarks()  # All curled
        r = _recognize_stable(recognizer, lm)
        assert r.gesture == GestureType.FIST

    def test_palm_open(self, recognizer):
        lm = _make_landmarks(
            thumb_ext=True, index_ext=True, middle_ext=True,
            ring_ext=True, pinky_ext=True,
        )
        r = _recognize_stable(recognizer, lm)
        assert r.gesture == GestureType.PALM_OPEN

    def test_point(self, recognizer):
        lm = _make_landmarks(index_ext=True, pinch_distance=0.2)
        r = _recognize_stable(recognizer, lm)
        assert r.gesture == GestureType.POINT

    def test_two_fingers(self, recognizer):
        lm = _make_landmarks(index_ext=True, middle_ext=True, pinch_distance=0.2)
        r = _recognize_stable(recognizer, lm)
        assert r.gesture == GestureType.TWO_FINGERS

    def test_three_fingers(self, recognizer):
        lm = _make_landmarks(
            index_ext=True, middle_ext=True, ring_ext=True,
            pinch_distance=0.2,
        )
        r = _recognize_stable(recognizer, lm)
        assert r.gesture == GestureType.THREE_FINGERS

    def test_thumb_only(self, recognizer):
        lm = _make_landmarks(thumb_ext=True)
        r = _recognize_stable(recognizer, lm)
        assert r.gesture == GestureType.THUMB_ONLY

    def test_peace_thumb(self, recognizer):
        lm = _make_landmarks(
            thumb_ext=True, index_ext=True, middle_ext=True,
            pinch_distance=0.2,
        )
        r = _recognize_stable(recognizer, lm)
        assert r.gesture == GestureType.PEACE_THUMB


class TestStabilityBuffer:
    """Verify that the stability buffer prevents single-frame gestures."""

    def test_single_frame_stays_none(self, recognizer):
        """A single recognize() call should NOT change from NONE."""
        lm = _make_landmarks(index_ext=True, pinch_distance=0.2)
        r = recognizer.recognize(_hand(lm))
        # First call: buffer not full, should remain NONE
        assert r.gesture == GestureType.NONE

    def test_gesture_stabilises_after_n_frames(self, recognizer):
        """After stability_frames calls, the gesture should be confirmed."""
        lm = _make_landmarks(index_ext=True, pinch_distance=0.2)
        hand = _hand(lm)
        results = [recognizer.recognize(hand) for _ in range(STABILITY_REPS)]
        # Last result should be stable
        assert results[-1].gesture == GestureType.POINT

    def test_noise_frame_rejected(self, recognizer):
        """A single noisy frame amid consistent gesture should not flip."""
        lm_point = _make_landmarks(index_ext=True, pinch_distance=0.2)
        lm_fist = _make_landmarks()

        # Establish POINT
        for _ in range(STABILITY_REPS):
            recognizer.recognize(_hand(lm_point))

        # Single noise frame
        recognizer.recognize(_hand(lm_fist))

        # Should still be POINT
        r = recognizer.recognize(_hand(lm_point))
        assert r.gesture == GestureType.POINT


class TestGestureResult:
    def test_cursor_position(self, recognizer):
        lm = _make_landmarks(index_ext=True, pinch_distance=0.2)
        r = recognizer.recognize(_hand(lm))
        # cursor_x and cursor_y should match index tip coordinates
        assert r.cursor_x == lm[INDEX_TIP][0]
        assert r.cursor_y == lm[INDEX_TIP][1]

    def test_pinch_distance(self, recognizer):
        lm = _make_landmarks(index_ext=True, pinch_distance=0.2)
        r = recognizer.recognize(_hand(lm))
        assert r.pinch_distance > 0
