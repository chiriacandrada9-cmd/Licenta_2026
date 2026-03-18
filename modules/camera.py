"""
Webcam capture thread.

Continuously reads frames from the webcam in a dedicated thread and
exposes the latest frame to consumers via get_frame().
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class Camera:
    """
    Thread-safe webcam capture.

    Usage
    -----
    >>> cam = Camera()
    >>> cam.start()
    >>> frame = cam.get_frame()   # Latest BGR ndarray
    >>> cam.stop()
    """

    def __init__(
        self,
        device_index: int | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        flip_horizontal: bool | None = None,
    ) -> None:
        cam_cfg = config.get("camera")
        self.device_index = device_index if device_index is not None else cam_cfg["device_index"]
        self.width = width or cam_cfg["width"]
        self.height = height or cam_cfg["height"]
        self.fps = fps or cam_cfg["fps"]
        self.flip = flip_horizontal if flip_horizontal is not None else cam_cfg["flip_horizontal"]

        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Open the capture device and launch the reading thread."""
        if self._running:
            logger.warning("Camera already running")
            return

        logger.info(
            "Opening camera %d (%dx%d @ %d fps)",
            self.device_index, self.width, self.height, self.fps,
        )
        self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device {self.device_index}. "
                "Check that your webcam is connected and not in use by another app."
            )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraThread")
        self._thread.start()
        logger.info("Camera started")

    def stop(self) -> None:
        """Signal the capture thread to stop and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera stopped")

    def get_frame(self) -> np.ndarray | None:
        """Return the most recent frame (thread-safe), or None if unavailable."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def _capture_loop(self) -> None:
        """Continuously grab frames until stop() is called."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                break
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Frame read failed - retrying in 100 ms")
                time.sleep(0.1)
                continue
            if self.flip:
                frame = cv2.flip(frame, 1)  # Horizontal flip (mirror)
            with self._lock:
                self._frame = frame

        logger.debug("Capture loop exited")
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    config.load()

    cam = Camera()
    cam.start()
    try:
        while True:
            frame = cam.get_frame()
            if frame is not None:
                cv2.imshow("Camera Test - press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
