"""
Central coordinator.

Manages the hand-tracking, voice-control, and input-simulation pipelines
across multiple threads and dispatches gestures / voice commands to the
appropriate controllers.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from enum import Enum

import cv2
import numpy as np

import config
from modules.camera import Camera
from modules.hand_tracker import HandTracker, HandData
from modules.gestures import GestureRecognizer, GestureResult, GestureType
from modules.mouse_controller import MouseController
from modules.keyboard_controller import KeyboardController, VK_CONTROL, VK_MENU, VK_LWIN, VK_F4, VK_TAB
from modules.command_parser import CommandParser, ParsedCommand, ActionType
from modules.window_manager import WindowManager
from utils.smoothing import Smoother
from utils.geometry import normalize_to_screen

logger = logging.getLogger(__name__)


class AppMode(Enum):
    HAND_ONLY = "hand"
    VOICE_ONLY = "voice"
    COMBINED = "combined"


class Orchestrator:
    """Coordoneaza toate subsistemele: camera, audio, recunoastere gesturi,
    comenzi vocale. Ruleaza pe thread-uri separate pentru camera, audio si
    procesare, iar thread-ul principal se ocupa de preview/tray.
    """

    def __init__(self) -> None:
        # Mode
        mode_str = config.get("app", "mode") or "combined"
        self._mode = AppMode(mode_str)
        self._running = False
        self._stop_event = threading.Event()

        # Queues
        self._gesture_queue: queue.Queue[GestureResult] = queue.Queue(maxsize=30)
        self._utterance_queue: queue.Queue[bytes] = queue.Queue(maxsize=10)

        # Subsystems (lazily initialised in start())
        self._camera: Camera | None = None
        self._hand_tracker: HandTracker | None = None
        self._gesture_recognizer: GestureRecognizer | None = None
        self._smoother: Smoother | None = None
        self._mouse: MouseController | None = None
        self._keyboard: KeyboardController | None = None
        self._command_parser: CommandParser | None = None
        self._window_manager: WindowManager | None = None

        # Voice subsystems (loaded only when needed)
        self._audio_capture = None
        self._vad = None
        self._speech_recognizer = None

        # State
        self._dragging = False
        self._dictation_mode = False
        self._last_action_gesture: GestureType | None = None  # Fire-once tracking

        # ROI settings for screen mapping
        ht_cfg = config.get("hand_tracking")
        self._roi_x_min = ht_cfg.get("roi_x_min", 0.12)
        self._roi_x_max = ht_cfg.get("roi_x_max", 0.88)
        self._roi_y_min = ht_cfg.get("roi_y_min", 0.10)
        self._roi_y_max = ht_cfg.get("roi_y_max", 0.90)

        # Scroll tracking
        self._prev_scroll_y: float | None = None
        self._scroll_accum: float = 0.0

        # Preview frame (rendered by main thread, written by camera thread)
        self._preview_frame: np.ndarray | None = None
        self._preview_lock = threading.Lock()
        self._show_preview = config.get("app", "show_preview")

        # Threads
        self._camera_thread: threading.Thread | None = None
        self._audio_thread: threading.Thread | None = None
        self._processing_thread: threading.Thread | None = None

    def start(self) -> None:
        """Initialise all subsystems and launch worker threads."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        logger.info("Orchestrator starting in '%s' mode...", self._mode.value)
        self._running = True
        self._stop_event.clear()

        # ---- Always-needed ----
        self._mouse = MouseController()
        self._keyboard = KeyboardController()
        self._command_parser = CommandParser()
        self._window_manager = WindowManager()

        # ---- Hand subsystems ----
        if self._mode in (AppMode.HAND_ONLY, AppMode.COMBINED):
            self._init_hand()
            self._camera_thread = threading.Thread(
                target=self._camera_loop, daemon=True, name="CameraThread",
            )
            self._camera_thread.start()

        # ---- Voice subsystems ----
        if self._mode in (AppMode.VOICE_ONLY, AppMode.COMBINED):
            self._init_voice()
            self._audio_thread = threading.Thread(
                target=self._audio_loop, daemon=True, name="AudioThread",
            )
            self._audio_thread.start()

        # ---- Processing thread ----
        self._processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True, name="ProcessingThread",
        )
        self._processing_thread.start()

        logger.info("Orchestrator started")

    def stop(self) -> None:
        """Stop all threads and release resources."""
        if not self._running:
            return
        logger.info("Orchestrator stopping...")
        self._running = False
        self._stop_event.set()

        # Stop subsystems
        if self._camera:
            self._camera.stop()
        if self._hand_tracker:
            self._hand_tracker.close()
        if self._audio_capture:
            self._audio_capture.stop()

        # Wait for threads
        for t in (self._camera_thread, self._audio_thread, self._processing_thread):
            if t and t.is_alive():
                t.join(timeout=3.0)

        cv2.destroyAllWindows()

        # Reclaim RAM from large ML models (MediaPipe, Whisper)
        import gc
        gc.collect()

        logger.info("Orchestrator stopped")

    def wait(self) -> None:
        """Block until the stop event is set."""
        self._stop_event.wait()

    def is_running(self) -> bool:
        return self._running

    def get_preview_frame(self) -> np.ndarray | None:
        """Return the latest annotated preview frame (thread-safe)."""
        with self._preview_lock:
            return self._preview_frame.copy() if self._preview_frame is not None else None

    def set_mode(self, mode: str) -> None:
        """Change the operating mode at runtime (triggers restart)."""
        new_mode = AppMode(mode)
        if new_mode == self._mode:
            return
        logger.info("Mode change: %s -> %s", self._mode.value, new_mode.value)
        was_running = self._running
        if was_running:
            self.stop()
        self._mode = new_mode
        config.set_value("app", "mode", mode)
        if was_running:
            self.start()

    def _init_hand(self) -> None:
        self._camera = Camera()
        self._camera.start()
        self._hand_tracker = HandTracker()
        self._gesture_recognizer = GestureRecognizer()
        ht_cfg = config.get("hand_tracking")
        self._smoother = Smoother(
            freq=config.get("camera", "fps"),
            min_cutoff=ht_cfg.get("one_euro_min_cutoff", 1.0),
            beta=ht_cfg.get("one_euro_beta", 0.007),
            deadzone=config.get("gestures", "deadzone_radius"),
        )

    def _init_voice(self) -> None:
        from modules.audio_capture import AudioCapture
        from modules.vad import VoiceActivityDetector
        from modules.speech_recognizer import SpeechRecognizer

        self._audio_capture = AudioCapture()
        self._audio_capture.start()
        self._vad = VoiceActivityDetector()
        self._speech_recognizer = SpeechRecognizer()
        # Pre-load the model in a background thread so it's ready
        threading.Thread(
            target=self._speech_recognizer.load_model,
            daemon=True,
            name="WhisperLoader",
        ).start()

    def _camera_loop(self) -> None:
        """Capture frames -> detect hands -> recognise gestures -> push to queue."""
        logger.info("Camera loop started")
        while self._running:
            frame = self._camera.get_frame() if self._camera else None
            if frame is None:
                time.sleep(0.01)
                continue

            hands = self._hand_tracker.process(frame)

            if hands:
                result = self._gesture_recognizer.recognize(hands[0])
                try:
                    self._gesture_queue.put_nowait(result)
                except queue.Full:
                    pass

                # Draw landmarks and gesture label for preview
                if self._show_preview:
                    self._hand_tracker.draw_landmarks(frame, hands[0])
                    label = result.gesture.value
                    cv2.putText(
                        frame, label, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                    )

            # Store preview frame for main thread to render
            if self._show_preview:
                with self._preview_lock:
                    self._preview_frame = frame

        logger.info("Camera loop exited")

    def _audio_loop(self) -> None:
        """Capture audio -> VAD -> push complete utterances to queue."""
        logger.info("Audio loop started")
        while self._running:
            if self._audio_capture is None:
                time.sleep(0.1)
                continue
            chunk = self._audio_capture.get_chunk(timeout=0.1)
            if chunk is None:
                continue
            utterance = self._vad.feed(chunk)
            if utterance is not None:
                try:
                    self._utterance_queue.put_nowait(utterance)
                except queue.Full:
                    logger.warning("Utterance queue full - dropping utterance")
        logger.info("Audio loop exited")

    def _processing_loop(self) -> None:
        """Poll gesture + utterance queues and dispatch actions."""
        logger.info("Processing loop started")
        while self._running:
            # ---- Gestures ----
            # Drain ALL queued gestures and act only on the LATEST one.
            # This prevents input lag from stale gesture data piling up
            # when the camera produces frames faster than we consume.
            if self._mode in (AppMode.HAND_ONLY, AppMode.COMBINED):
                latest_gesture = None
                while True:
                    try:
                        latest_gesture = self._gesture_queue.get_nowait()
                    except queue.Empty:
                        break
                if latest_gesture is not None:
                    self._handle_gesture(latest_gesture)

            # ---- Voice ----
            if self._mode in (AppMode.VOICE_ONLY, AppMode.COMBINED):
                try:
                    utterance = self._utterance_queue.get_nowait()
                    text = self._speech_recognizer.transcribe(utterance)
                    if text:
                        if self._dictation_mode:
                            self._handle_dictation(text)
                        else:
                            command = self._command_parser.parse(text)
                            self._handle_command(command)
                except queue.Empty:
                    pass

            time.sleep(1 / 120)  # ~120 Hz (lower latency gesture response)
        logger.info("Processing loop exited")

    def _handle_gesture(self, result: GestureResult) -> None:
        """Map a gesture result to a mouse/keyboard action."""
        screen_w = self._mouse.screen_w
        screen_h = self._mouse.screen_h

        # Reset fire-once tracking when gesture type changes
        if result.gesture != self._last_action_gesture:
            self._last_action_gesture = None

        match result.gesture:
            case GestureType.POINT:
                sx, sy = self._smooth_and_map(result.cursor_x, result.cursor_y, screen_w, screen_h)
                self._mouse.move_to(sx, sy)

            case GestureType.PINCH:
                self._mouse.left_click()

            case GestureType.PINCH_HOLD:
                self._mouse.double_click()

            case GestureType.PALM_OPEN:
                # Fire right-click only ONCE when entering palm gesture
                if self._last_action_gesture != GestureType.PALM_OPEN:
                    self._mouse.right_click()
                    self._last_action_gesture = GestureType.PALM_OPEN

            case GestureType.PEACE_THUMB:
                # Fire middle-click only ONCE when entering gesture
                if self._last_action_gesture != GestureType.PEACE_THUMB:
                    self._mouse.middle_click()
                    self._last_action_gesture = GestureType.PEACE_THUMB

            case GestureType.TWO_FINGERS:
                # Scroll: use anchor-based tracking for stability.
                # On first frame of scroll, record the Y position as anchor.
                # Then scroll based on cumulative displacement from anchor.
                if self._prev_scroll_y is None:
                    # Entering scroll mode  -  set anchor
                    self._prev_scroll_y = result.cursor_y
                    self._scroll_accum = 0.0
                else:
                    dy = result.cursor_y - self._prev_scroll_y
                    self._scroll_accum += dy
                    self._prev_scroll_y = result.cursor_y

                    # Only scroll when accumulated movement exceeds threshold
                    SCROLL_STEP = 0.02  # ~2% of camera height per scroll tick
                    if abs(self._scroll_accum) > SCROLL_STEP:
                        # Fingers down = page scrolls down, fingers up = page scrolls up
                        ticks = int(self._scroll_accum / SCROLL_STEP)
                        self._mouse.scroll(-ticks)  # Negative = down in Windows
                        self._scroll_accum -= ticks * SCROLL_STEP

            case GestureType.THREE_FINGERS:
                sx, sy = self._smooth_and_map(result.cursor_x, result.cursor_y, screen_w, screen_h)
                if not self._dragging:
                    self._mouse.left_down()
                    self._dragging = True
                self._mouse.move_to(sx, sy)

            case GestureType.FIST:
                if self._dragging:
                    self._mouse.left_up()
                    self._dragging = False

            case GestureType.SWIPE_LEFT:
                self._keyboard.hotkey(VK_LWIN, VK_CONTROL, 0x25)  # Win+Ctrl+Left

            case GestureType.SWIPE_RIGHT:
                self._keyboard.hotkey(VK_LWIN, VK_CONTROL, 0x27)  # Win+Ctrl+Right

            case _:
                pass

        # Reset scroll state when not in scroll gesture
        if result.gesture != GestureType.TWO_FINGERS:
            self._prev_scroll_y = None
            self._scroll_accum = 0.0

    def _smooth_and_map(
        self, raw_x: float, raw_y: float, screen_w: int, screen_h: int,
    ) -> tuple[int, int]:
        """Apply smoothing and ROI-aware mapping to screen pixels."""
        sx, sy = self._smoother.smooth(raw_x, raw_y)
        return normalize_to_screen(
            sx, sy, screen_w, screen_h,
            roi_x_min=self._roi_x_min,
            roi_x_max=self._roi_x_max,
            roi_y_min=self._roi_y_min,
            roi_y_max=self._roi_y_max,
        )

    def _handle_command(self, cmd: ParsedCommand) -> None:
        """Execute a parsed voice command."""
        match cmd.action:
            # Mouse
            case ActionType.CLICK:
                self._mouse.left_click()
            case ActionType.RIGHT_CLICK:
                self._mouse.right_click()
            case ActionType.DOUBLE_CLICK:
                self._mouse.double_click()
            case ActionType.SCROLL_UP:
                self._mouse.scroll(3)
            case ActionType.SCROLL_DOWN:
                self._mouse.scroll(-3)

            # Keyboard
            case ActionType.TYPE_TEXT:
                if cmd.argument:
                    self._keyboard.type_text(cmd.argument)
            case ActionType.PRESS_KEY:
                self._press_named_key(cmd.argument)
            case ActionType.HOTKEY:
                self._press_named_hotkey(cmd.argument)

            # Windows
            case ActionType.OPEN_APP:
                if cmd.argument:
                    self._window_manager.open_application(cmd.argument)
            case ActionType.MINIMIZE:
                self._window_manager.minimize_window()
            case ActionType.MAXIMIZE:
                self._window_manager.maximize_window()
            case ActionType.CLOSE_WINDOW:
                self._keyboard.hotkey(VK_MENU, VK_F4)
            case ActionType.NEXT_WINDOW:
                self._keyboard.hotkey(VK_MENU, VK_TAB)
            case ActionType.SHOW_DESKTOP:
                from modules.keyboard_controller import VK_LWIN
                self._keyboard.hotkey(VK_LWIN, ord("D"))

            # App control
            case ActionType.SET_MODE:
                if cmd.argument:
                    self.set_mode(cmd.argument)
            case ActionType.PAUSE_CONTROL:
                self.stop()
            case ActionType.RESUME_CONTROL:
                self.start()

            # Dictation
            case ActionType.DICTATION_ON:
                self._dictation_mode = True
                logger.info("Dictation mode ON")
            case ActionType.DICTATION_OFF:
                self._dictation_mode = False
                logger.info("Dictation mode OFF")

            case ActionType.UNKNOWN:
                logger.debug("Unrecognised voice command: '%s'", cmd.raw_text)

    def _handle_dictation(self, text: str) -> None:
        """In dictation mode, type everything as text (unless it's a stop command)."""
        # Check for stop command even in dictation mode
        cmd = self._command_parser.parse(text)
        if cmd.action == ActionType.DICTATION_OFF:
            self._dictation_mode = False
            logger.info("Dictation mode OFF (via voice)")
            return
        self._keyboard.type_text(text + " ")

    def _press_named_key(self, name: str | None) -> None:
        from modules.keyboard_controller import VK_RETURN, VK_ESCAPE, VK_TAB
        key_map = {
            "enter": VK_RETURN,
            "escape": VK_ESCAPE,
            "tab": VK_TAB,
        }
        if name and name in key_map:
            self._keyboard.press_key(key_map[name])

    def _press_named_hotkey(self, name: str | None) -> None:
        hotkey_map = {
            "copy": (VK_CONTROL, ord("C")),
            "paste": (VK_CONTROL, ord("V")),
            "undo": (VK_CONTROL, ord("Z")),
            "select_all": (VK_CONTROL, ord("A")),
        }
        if name and name in hotkey_map:
            self._keyboard.hotkey(*hotkey_map[name])
