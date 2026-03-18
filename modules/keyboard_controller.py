"""
Win32 keyboard input simulation via ctypes.

Uses SendInput with INPUT_KEYBOARD structures.  Supports individual key
presses, multi-key hotkeys, and Unicode text typing.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import time

logger = logging.getLogger(__name__)
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

# Common virtual-key codes
VK_RETURN = 0x0D
VK_ESCAPE = 0x1B
VK_TAB = 0x09
VK_BACK = 0x08
VK_DELETE = 0x2E
VK_SPACE = 0x20
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # Alt
VK_LWIN = 0x5B
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_F4 = 0x73

# Letter keys: ord("A") - ord("Z") = 0x41-0x5A
# Number keys: ord("0") - ord("9") = 0x30-0x39
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]

    _anonymous_ = ("_union",)
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("_union", _INPUT_UNION),
    ]

class KeyboardController:
    """Simulate keyboard input via Win32 SendInput."""

    def press_key(self, vk_code: int) -> None:
        """Press and release a single key."""
        self._key_down(vk_code)
        self._key_up(vk_code)

    def hotkey(self, *vk_codes: int) -> None:
        """
        Press a key combination (e.g. Ctrl+C).

        Keys are pressed in order and released in reverse order.
        """
        for vk in vk_codes:
            self._key_down(vk)
            time.sleep(0.01)
        for vk in reversed(vk_codes):
            self._key_up(vk)

    def key_down(self, vk_code: int) -> None:
        self._key_down(vk_code)

    def key_up(self, vk_code: int) -> None:
        self._key_up(vk_code)

    def type_text(self, text: str) -> None:
        """Tasteaza un string caracter cu caracter folosind Unicode
        (KEYEVENTF_UNICODE). Evita problemele de keyboard layout."""
        for char in text:
            self._unicode_char(char)
            time.sleep(0.005)  # Small delay for stability

    @staticmethod
    def _key_down(vk_code: int) -> None:
        inp = INPUT(type=INPUT_KEYBOARD)
        inp.ki.wVk = vk_code
        inp.ki.dwFlags = 0
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    @staticmethod
    def _key_up(vk_code: int) -> None:
        inp = INPUT(type=INPUT_KEYBOARD)
        inp.ki.wVk = vk_code
        inp.ki.dwFlags = KEYEVENTF_KEYUP
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    @staticmethod
    def _unicode_char(char: str) -> None:
        # Key down
        inp_down = INPUT(type=INPUT_KEYBOARD)
        inp_down.ki.wVk = 0
        inp_down.ki.wScan = ord(char)
        inp_down.ki.dwFlags = KEYEVENTF_UNICODE
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp_down), ctypes.sizeof(INPUT))
        # Key up
        inp_up = INPUT(type=INPUT_KEYBOARD)
        inp_up.ki.wVk = 0
        inp_up.ki.wScan = ord(char)
        inp_up.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp_up), ctypes.sizeof(INPUT))
