"""
Gesture recognition engine.

Analyses MediaPipe hand landmarks to classify the current hand pose into
one of the defined gestures and extract cursor/scroll parameters.

Includes temporal filtering (stability buffer), hysteresis thresholds,
and click cooldowns for robust detection.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass
from enum import Enum

import config
from modules.hand_tracker import (
    HandData,
    WRIST, THUMB_TIP, THUMB_IP, THUMB_MCP,
    INDEX_TIP, INDEX_PIP, INDEX_MCP,
    MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP,
    RING_TIP, RING_PIP, RING_MCP,
    PINKY_TIP, PINKY_PIP, PINKY_MCP,
)
from utils.geometry import distance_2d, is_finger_extended

logger = logging.getLogger(__name__)

class GestureType(Enum):
    NONE = "none"
    POINT = "point"              # Index up, others curled -> move cursor
    PINCH = "pinch"              # Index-Thumb close -> left click
    PINCH_HOLD = "pinch_hold"    # Sustained pinch -> double click
    FIST = "fist"                # All curled -> neutral / pause
    PALM_OPEN = "palm_open"      # All extended -> right click
    TWO_FINGERS = "two_fingers"  # Index+Middle -> scroll mode
    THREE_FINGERS = "three_fingers"  # Idx+Mid+Ring -> drag mode
    THUMB_ONLY = "thumb_only"    # Only thumb extended -> modifier
    PEACE_THUMB = "peace_thumb"  # Thumb+Index+Middle -> middle click
    SWIPE_LEFT = "swipe_left"    # Quick lateral wrist movement <-
    SWIPE_RIGHT = "swipe_right"  # Quick lateral wrist movement ->


@dataclass
class GestureResult:
    """Output of the gesture recogniser for a single frame."""

    gesture: GestureType
    cursor_x: float       # Normalised 0-1 (index tip x)
    cursor_y: float       # Normalised 0-1 (index tip y)
    confidence: float     # 0-1 - how clearly the gesture is formed
    pinch_distance: float  # Thumb-index distance (normalised)

class GestureRecognizer:
    """Clasifica gestul mâinii pe baza pozițiilor degetelor.
    Folosește un buffer de stabilitate, histereză și cooldown
    pentru a evita clasificări false.
    """

    def __init__(self) -> None:
        gcfg = config.get("gestures")
        self.pinch_threshold: float = gcfg.get("pinch_threshold", 0.05)
        self.pinch_release_threshold: float = gcfg.get("pinch_release_threshold", 0.07)
        self.click_hold_frames: int = gcfg.get("click_hold_frames", 4)
        self.stability_frames: int = gcfg.get("stability_frames", 3)
        self.click_cooldown_ms: float = gcfg.get("click_cooldown_ms", 300)
        self.finger_hysteresis: float = gcfg.get("finger_hysteresis", 0.03)

        # Swipe detection state
        self._wrist_history: collections.deque[tuple[float, float]] = collections.deque(
            maxlen=15
        )  # (x, timestamp)
        self._swipe_cooldown: float = 0.0

        # Pinch state (with hysteresis)
        self._pinch_active: bool = False
        self._pinch_frames: int = 0

        # Stability buffer: track recent raw classifications
        self._gesture_buffer: collections.deque[GestureType] = collections.deque(
            maxlen=max(self.stability_frames, 1)
        )
        self._confirmed_gesture: GestureType = GestureType.NONE

        # Click cooldown tracking
        self._last_click_time: float = 0.0
        self._last_rclick_time: float = 0.0
        self._last_mclick_time: float = 0.0
        self._last_dclick_time: float = 0.0

        # Previous finger states for hysteresis
        self._prev_fingers: list[bool] = [False] * 5

    def recognize(self, hand: HandData) -> GestureResult:
        """Classify the current hand pose with temporal filtering."""
        lm = hand.landmarks
        handedness = hand.handedness

        # Cursor position from index finger tip
        cursor_x = lm[INDEX_TIP][0]
        cursor_y = lm[INDEX_TIP][1]

        # Pinch distance (thumb tip <-> index tip, normalised)
        pinch_dist = distance_2d(lm[THUMB_TIP][:2], lm[INDEX_TIP][:2])

        # Also compute thumb-index MCP distance for gesture disambiguation
        thumb_index_spread = distance_2d(lm[THUMB_TIP][:2], lm[INDEX_MCP][:2])
        fingers = self._get_finger_states(lm, handedness)
        num_extended = sum(fingers)
        now = time.monotonic()
        wrist_x = lm[WRIST][0]
        self._wrist_history.append((wrist_x, now))

        swipe_gesture = self._detect_swipe(now)
        if swipe_gesture is not None:
            self._pinch_frames = 0
            self._pinch_active = False
            return GestureResult(
                gesture=swipe_gesture,
                cursor_x=cursor_x,
                cursor_y=cursor_y,
                confidence=0.9,
                pinch_distance=pinch_dist,
            )
        # Only consider pinch if index finger is NOT clearly pointing
        # (when pointing, thumb may rest near index but shouldn't trigger pinch)
        if not self._pinch_active:
            # Entering pinch: tighter threshold + require thumb moving toward index
            is_pinch = pinch_dist < self.pinch_threshold
        else:
            # Leaving pinch: use looser threshold (harder to exit)
            is_pinch = pinch_dist < self.pinch_release_threshold

        if is_pinch:
            self._pinch_active = True
            self._pinch_frames += 1
        else:
            if self._pinch_active:
                self._pinch_active = False
            self._pinch_frames = 0

        # Pinch hold detection (exactly on hold threshold)
        if self._pinch_frames > 0 and self._pinch_frames == self.click_hold_frames:
            if self._can_click(now, "double"):
                return GestureResult(
                    gesture=GestureType.PINCH_HOLD,
                    cursor_x=cursor_x,
                    cursor_y=cursor_y,
                    confidence=0.85,
                    pinch_distance=pinch_dist,
                )

        # First frame of pinch -> single click (with cooldown)
        if is_pinch and self._pinch_frames == 1:
            if self._can_click(now, "left"):
                return GestureResult(
                    gesture=GestureType.PINCH,
                    cursor_x=cursor_x,
                    cursor_y=cursor_y,
                    confidence=0.9,
                    pinch_distance=pinch_dist,
                )

        # Sustained pinch past hold -> freeze cursor (return FIST/neutral
        # so the cursor doesn't move while the user is pinching)
        if is_pinch:
            return GestureResult(
                gesture=GestureType.FIST,
                cursor_x=cursor_x,
                cursor_y=cursor_y,
                confidence=0.5,
                pinch_distance=pinch_dist,
            )
        raw_gesture = self._classify_static(
            fingers, num_extended, pinch_dist, thumb_index_spread,
        )

        # Push to stability buffer
        self._gesture_buffer.append(raw_gesture)

        # Determine stable gesture via majority vote
        stable_gesture = self._get_stable_gesture(raw_gesture)

        return GestureResult(
            gesture=stable_gesture,
            cursor_x=cursor_x,
            cursor_y=cursor_y,
            confidence=0.85 if stable_gesture != GestureType.NONE else 0.3,
            pinch_distance=pinch_dist,
        )

    def _get_finger_states(
        self, lm: list[tuple[float, float, float]], handedness: str,
    ) -> list[bool]:
        """
        Determine which fingers are extended, using hysteresis to prevent
        flickering at boundary positions.

        If a finger was previously extended, it stays extended until it
        clearly curls (using negative margin). If it was curled, it stays
        curled until it clearly extends (using positive margin).
        """
        margin = self.finger_hysteresis
        fingers = []

        # Thumb
        prev_thumb = self._prev_fingers[0]
        m = -margin if prev_thumb else margin
        thumb_ext = is_finger_extended(
            lm[THUMB_TIP][:2], lm[THUMB_IP][:2], lm[THUMB_MCP][:2],
            is_thumb=True, handedness=handedness, margin=m,
        )
        fingers.append(thumb_ext)

        # Index, Middle, Ring, Pinky
        finger_landmarks = [
            (INDEX_TIP, INDEX_PIP, INDEX_MCP),
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP, RING_PIP, RING_MCP),
            (PINKY_TIP, PINKY_PIP, PINKY_MCP),
        ]
        for i, (tip_idx, pip_idx, mcp_idx) in enumerate(finger_landmarks, start=1):
            prev = self._prev_fingers[i]
            m = -margin if prev else margin
            ext = is_finger_extended(
                lm[tip_idx][:2], lm[pip_idx][:2], lm[mcp_idx][:2],
                margin=m,
            )
            fingers.append(ext)

        self._prev_fingers = fingers
        return fingers

    def _get_stable_gesture(self, raw_gesture: GestureType) -> GestureType:
        """
        Return a stable gesture using majority vote over the buffer.

        The gesture changes from the _confirmed_gesture only if the new
        candidate has appeared in >= stability_frames of the last N frames.
        This prevents single-frame glitches from triggering actions.
        """
        if len(self._gesture_buffer) < self.stability_frames:
            return self._confirmed_gesture

        # Count occurrences in the buffer
        counts: dict[GestureType, int] = {}
        for g in self._gesture_buffer:
            counts[g] = counts.get(g, 0) + 1

        # Find the most common gesture
        most_common = max(counts, key=counts.get)
        most_count = counts[most_common]

        # Only switch if the new gesture dominates the buffer
        if most_count >= self.stability_frames:
            self._confirmed_gesture = most_common
        # Otherwise keep the current confirmed gesture

        return self._confirmed_gesture

    def _can_click(self, now: float, click_type: str) -> bool:
        """Check and update click cooldown. Returns True if click is allowed."""
        cooldown_s = self.click_cooldown_ms / 1000.0

        if click_type == "left":
            if now - self._last_click_time < cooldown_s:
                return False
            self._last_click_time = now
            return True
        elif click_type == "right":
            if now - self._last_rclick_time < cooldown_s:
                return False
            self._last_rclick_time = now
            return True
        elif click_type == "middle":
            if now - self._last_mclick_time < cooldown_s:
                return False
            self._last_mclick_time = now
            return True
        elif click_type == "double":
            if now - self._last_dclick_time < cooldown_s:
                return False
            self._last_dclick_time = now
            return True
        return True

    def _classify_static(
        self,
        fingers: list[bool],
        num_extended: int,
        pinch_dist: float = 1.0,
        thumb_index_spread: float = 1.0,
    ) -> GestureType:
        """
        Classify a static hand pose from finger states.

        Order matters! Specific gestures (fewer fingers) must be checked
        BEFORE broad ones (PALM_OPEN) to prevent swallowing.
        """
        thumb, index, middle, ring, pinky = fingers

        # Fist - all curled, OR only thumb slightly visible (common from
        # side angles where thumb reads as borderline extended)
        if num_extended == 0:
            return GestureType.FIST
        if num_extended == 1 and thumb and not index:
            if thumb_index_spread > 0.12:
                return GestureType.THUMB_ONLY
            return GestureType.FIST

        # Point - index extended, others curled (check BEFORE palm)
        if index and not middle and not ring and not pinky:
            if pinch_dist > self.pinch_threshold:
                return GestureType.POINT
            return GestureType.NONE

        # Two fingers - V sign (check BEFORE palm so it's not swallowed)
        if index and middle and not ring and not pinky:
            if thumb:
                return GestureType.PEACE_THUMB  # Thumb + V = middle click
            return GestureType.TWO_FINGERS  # V = scroll mode

        # Three fingers (check BEFORE palm)
        if index and middle and ring and not pinky:
            return GestureType.THREE_FINGERS  # Drag mode

        # Palm open - all core fingers extended (must come AFTER specific checks)
        # Accepts 4+ fingers OR all 5, since pinky often misdetects
        if num_extended >= 4 and index and middle and ring:
            return GestureType.PALM_OPEN

        return GestureType.NONE

    def _detect_swipe(self, now: float) -> GestureType | None:
        """Detect horizontal swipe from wrist X history."""
        if now < self._swipe_cooldown:
            return None
        if len(self._wrist_history) < 5:
            return None

        # Compare wrist positions within the last ~500 ms
        recent = [(x, t) for x, t in self._wrist_history if now - t < 0.50]
        if len(recent) < 3:
            return None

        dx = recent[-1][0] - recent[0][0]
        dt = recent[-1][1] - recent[0][1]
        if dt <= 0:
            return None

        velocity = dx / dt  # normalised-units per second
        displacement = abs(dx)  # total horizontal displacement

        # Require BOTH sufficient velocity AND minimum displacement
        # to avoid false triggers from gradual drift
        SWIPE_VELOCITY_THRESHOLD = 0.6
        SWIPE_MIN_DISPLACEMENT = 0.08

        if displacement < SWIPE_MIN_DISPLACEMENT:
            return None

        if velocity > SWIPE_VELOCITY_THRESHOLD:
            self._swipe_cooldown = now + 0.6
            self._wrist_history.clear()
            return GestureType.SWIPE_RIGHT
        if velocity < -SWIPE_VELOCITY_THRESHOLD:
            self._swipe_cooldown = now + 0.6
            self._wrist_history.clear()
            return GestureType.SWIPE_LEFT

        return None
if __name__ == "__main__":
    import cv2

    logging.basicConfig(level=logging.DEBUG)
    config.load()

    from modules.camera import Camera
    from modules.hand_tracker import HandTracker

    cam = Camera()
    cam.start()
    tracker = HandTracker()
    recognizer = GestureRecognizer()

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue
            hands = tracker.process(frame)
            for hand in hands:
                tracker.draw_landmarks(frame, hand)
                result = recognizer.recognize(hand)
                # Overlay gesture name
                label = f"{result.gesture.value}  pinch={result.pinch_distance:.3f}"
                cv2.putText(
                    frame, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                )
            cv2.imshow("Gesture Demo - press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cam.stop()
        cv2.destroyAllWindows()
