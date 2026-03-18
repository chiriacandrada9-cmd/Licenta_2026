"""
Win32 mouse input simulation via ctypes.

Uses SendInput with INPUT_MOUSE structures for absolute positioning,
clicks, and scroll events.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import time

logger = logging.getLogger(__name__)
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000

WHEEL_DELTA = 120  # Standard Windows scroll unit
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    _anonymous_ = ("_union",)
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("_union", _INPUT_UNION),
    ]

class MouseController:
    """
    Move the mouse cursor and generate click / scroll events.

    All coordinates are in *screen pixels* (0,0 = top-left).
    """

    def __init__(self, screen_width: int | None = None, screen_height: int | None = None) -> None:
        user32 = ctypes.windll.user32
        self.screen_w = screen_width or user32.GetSystemMetrics(0)
        self.screen_h = screen_height or user32.GetSystemMetrics(1)
        self._last_click_time: float = 0.0
        self._click_debounce: float = 0.05  # 50 ms between clicks
        logger.info("MouseController ready - screen %dx%d", self.screen_w, self.screen_h)

    def move_to(self, x: int, y: int) -> None:
        """Move cursor to absolute screen position (x, y)."""
        # Normalise to 0-65535 range for MOUSEEVENTF_ABSOLUTE
        abs_x = int(x * 65535 / self.screen_w)
        abs_y = int(y * 65535 / self.screen_h)
        self._send(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, abs_x, abs_y)

    def move_relative(self, dx: int, dy: int) -> None:
        """Move cursor by a relative offset in pixels."""
        self._send(MOUSEEVENTF_MOVE, dx, dy)

    def left_click(self) -> None:
        if not self._debounce_ok():
            return
        self._send(MOUSEEVENTF_LEFTDOWN)
        self._send(MOUSEEVENTF_LEFTUP)

    def right_click(self) -> None:
        if not self._debounce_ok():
            return
        self._send(MOUSEEVENTF_RIGHTDOWN)
        self._send(MOUSEEVENTF_RIGHTUP)

    def middle_click(self) -> None:
        if not self._debounce_ok():
            return
        self._send(MOUSEEVENTF_MIDDLEDOWN)
        self._send(MOUSEEVENTF_MIDDLEUP)

    def double_click(self) -> None:
        if not self._debounce_ok():
            return
        self._send(MOUSEEVENTF_LEFTDOWN)
        self._send(MOUSEEVENTF_LEFTUP)
        time.sleep(0.03)
        self._send(MOUSEEVENTF_LEFTDOWN)
        self._send(MOUSEEVENTF_LEFTUP)

    def left_down(self) -> None:
        self._send(MOUSEEVENTF_LEFTDOWN)

    def left_up(self) -> None:
        self._send(MOUSEEVENTF_LEFTUP)

    def scroll(self, amount: int) -> None:
        """Scroll vertically.  Positive = up, negative = down."""
        inp = INPUT(type=INPUT_MOUSE)
        inp.mi.mouseData = ctypes.wintypes.DWORD(amount * WHEEL_DELTA)
        inp.mi.dwFlags = MOUSEEVENTF_WHEEL
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def _debounce_ok(self) -> bool:
        now = time.monotonic()
        if now - self._last_click_time < self._click_debounce:
            return False
        self._last_click_time = now
        return True

    def _send(self, flags: int, dx: int = 0, dy: int = 0) -> None:
        inp = INPUT(type=INPUT_MOUSE)
        inp.mi.dx = dx
        inp.mi.dy = dy
        inp.mi.dwFlags = flags
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
