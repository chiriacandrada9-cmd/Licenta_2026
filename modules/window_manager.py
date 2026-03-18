"""
Win32 window management.

Enumerate, focus, minimize, maximize, and close windows.  Also launch
applications by name via configurable aliases.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass

import win32con
import win32gui
import win32process

import config

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    process_name: str
    is_visible: bool


class WindowManager:
    """High-level window operations backed by pywin32."""

    @staticmethod
    def list_windows() -> list[WindowInfo]:
        """Return all visible, titled top-level windows."""
        windows: list[WindowInfo] = []
        excluded = config.get("window_management", "excluded_processes") or []

        def _enum_cb(hwnd: int, _results: list) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return True
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                # We don't resolve the process name here to avoid
                # requiring elevated privileges.  Use PID instead.
                proc_name = str(pid)
            except Exception:
                proc_name = ""
            _results.append(
                WindowInfo(
                    hwnd=hwnd,
                    title=title,
                    process_name=proc_name,
                    is_visible=True,
                )
            )
            return True

        win32gui.EnumWindows(_enum_cb, windows)
        return windows

    @staticmethod
    def get_active_window() -> WindowInfo | None:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None
        title = win32gui.GetWindowText(hwnd)
        return WindowInfo(hwnd=hwnd, title=title, process_name="", is_visible=True)

    @staticmethod
    def focus_window(hwnd: int) -> None:
        try:
            # UIPI bypass: send synthetic Alt press so Windows allows
            # foreground change (otherwise it may just flash the taskbar).
            import ctypes
            ctypes.windll.user32.keybd_event(0x12, 0, 0, 0)       # Alt down
            ctypes.windll.user32.keybd_event(0x12, 0, 0x0002, 0)  # Alt up
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
        except Exception as exc:
            logger.warning("Failed to focus window %d: %s", hwnd, exc)

    @staticmethod
    def minimize_window(hwnd: int | None = None) -> None:
        hwnd = hwnd or win32gui.GetForegroundWindow()
        if hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            logger.info("Minimized window %d", hwnd)

    @staticmethod
    def maximize_window(hwnd: int | None = None) -> None:
        hwnd = hwnd or win32gui.GetForegroundWindow()
        if hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            logger.info("Maximized window %d", hwnd)

    @staticmethod
    def close_window(hwnd: int | None = None) -> None:
        hwnd = hwnd or win32gui.GetForegroundWindow()
        if hwnd:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            logger.info("Closed window %d", hwnd)

    @classmethod
    def find_window_by_title(cls, query: str) -> WindowInfo | None:
        """Find the first visible window whose title contains *query*."""
        query_lower = query.lower()
        for win in cls.list_windows():
            if query_lower in win.title.lower():
                return win
        return None

    @staticmethod
    def open_application(name: str) -> None:
        """
        Launch an application by friendly name.

        Checks the ``app_aliases`` map in settings first, then tries
        ``os.startfile`` as a fallback.
        """
        aliases: dict = config.get("window_management", "app_aliases") or {}
        exe = aliases.get(name.lower())

        if exe:
            logger.info("Launching alias '%s' -> %s", name, exe)
            try:
                subprocess.Popen(exe, shell=True)
            except Exception as exc:
                logger.error("Failed to launch %s: %s", exe, exc)
        else:
            logger.info("No alias for '%s' - trying os.startfile", name)
            try:
                os.startfile(name)
            except Exception as exc:
                logger.error("Failed to open '%s': %s", name, exc)
