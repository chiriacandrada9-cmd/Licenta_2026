"""
MediaPipe Hand Landmarker wrapper.

Accepts a BGR frame, runs MediaPipe hand landmark detection, and returns
structured HandData objects with both normalised and pixel landmarks.

Uses the new MediaPipe Tasks API (HandLandmarker) which replaced the
legacy mp.solutions.hands interface.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

import config

logger = logging.getLogger(__name__)

# Model download URL (official Google MediaPipe model)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


@dataclass
class HandData:
    """Detection result for a single hand."""

    landmarks: list[tuple[float, float, float]]  # 21 points, normalised (0-1)
    handedness: str  # "Left" or "Right"
    pixel_landmarks: list[tuple[int, int]] = field(default_factory=list)
    detection_score: float = 0.0

    def landmark(self, index: int) -> tuple[float, float, float]:
        """Shorthand to access a landmark by its MediaPipe index."""
        return self.landmarks[index]

    def pixel(self, index: int) -> tuple[int, int]:
        """Shorthand to access a pixel-space landmark."""
        return self.pixel_landmarks[index]


# MediaPipe landmark index constants
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Hand connections for drawing (pairs of landmark indices)
HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
]


def _ensure_model() -> str:
    """Download the hand landmarker model if not already cached."""
    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    assets_dir.mkdir(exist_ok=True)
    model_path = assets_dir / "hand_landmarker.task"

    if model_path.exists():
        return str(model_path)

    logger.info("Downloading hand landmarker model (first run only)...")
    urllib.request.urlretrieve(_MODEL_URL, str(model_path))
    logger.info("Model saved to %s", model_path)
    return str(model_path)


class HandTracker:
    """
    Wrapper around MediaPipe HandLandmarker (Tasks API) that converts
    raw detections into :class:`HandData` objects.
    """

    def __init__(
        self,
        max_hands: int | None = None,
        detection_confidence: float | None = None,
        tracking_confidence: float | None = None,
    ) -> None:
        ht_cfg = config.get("hand_tracking")
        self.max_hands = max_hands or ht_cfg["max_hands"]
        self.det_conf = detection_confidence or ht_cfg["detection_confidence"]
        self.trk_conf = tracking_confidence or ht_cfg["tracking_confidence"]

        model_path = _ensure_model()

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.det_conf,
            min_hand_presence_confidence=self.det_conf,
            min_tracking_confidence=self.trk_conf,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

        logger.info(
            "HandTracker ready (max_hands=%d, det=%.2f, trk=%.2f)",
            self.max_hands, self.det_conf, self.trk_conf,
        )

    def process(self, frame: np.ndarray) -> list[HandData]:
        """
        Detect hands in a BGR frame.

        Returns a list of :class:`HandData` (empty if no hand detected).
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        hands: list[HandData] = []
        if not result.hand_landmarks:
            return hands

        for hand_idx, hand_lms in enumerate(result.hand_landmarks):
            landmarks = [
                (lm.x, lm.y, lm.z) for lm in hand_lms
            ]
            pixel_landmarks = [
                (int(lm.x * w), int(lm.y * h)) for lm in hand_lms
            ]

            # Handedness
            handedness = "Right"
            score = 0.0
            if result.handedness and hand_idx < len(result.handedness):
                cat = result.handedness[hand_idx][0]
                handedness = cat.category_name
                score = cat.score

            hands.append(
                HandData(
                    landmarks=landmarks,
                    handedness=handedness,
                    pixel_landmarks=pixel_landmarks,
                    detection_score=score,
                )
            )

        return hands

    def draw_landmarks(
        self,
        frame: np.ndarray,
        hand_data: HandData,
    ) -> np.ndarray:
        """Draw the hand skeleton on the frame (for the preview window)."""
        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(hand_data.pixel_landmarks) and end_idx < len(hand_data.pixel_landmarks):
                pt1 = hand_data.pixel_landmarks[start_idx]
                pt2 = hand_data.pixel_landmarks[end_idx]
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw landmarks as dots
        for px, py in hand_data.pixel_landmarks:
            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

        return frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        logger.info("HandTracker closed")
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    config.load()

    from modules.camera import Camera

    cam = Camera()
    cam.start()
    tracker = HandTracker()

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue
            hands = tracker.process(frame)
            for hand in hands:
                tracker.draw_landmarks(frame, hand)
                # Show landmark indices
                for idx, (px, py) in enumerate(hand.pixel_landmarks):
                    cv2.putText(
                        frame, str(idx), (px, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1,
                    )
            cv2.imshow("Hand Tracker - press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cam.stop()
        cv2.destroyAllWindows()
