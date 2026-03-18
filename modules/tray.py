"""
System tray icon & menu.

Uses pystray + Pillow to create a tray icon with Start/Stop, mode
switching, preview toggle, and Exit.  Must run on the main thread on
Windows.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw
import pystray

import config

if TYPE_CHECKING:
    from modules.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

def _make_icon(colour: str = "green", size: int = 64) -> Image.Image:
    """Create a simple coloured circle icon."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 4
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=colour,
        outline="white",
        width=2,
    )
    return img


_ICON_COLOURS = {
    "running": "#22c55e",   # green
    "paused": "#facc15",    # yellow
    "stopped": "#ef4444",   # red
    "voice_only": "#3b82f6",  # blue
}

class TrayApp:
    """Aplicatia de system tray - meniu cu Start/Stop, mod, preview, Exit."""

    def __init__(self, orchestrator: Orchestrator) -> None:
        self._orch = orchestrator
        self._icon: pystray.Icon | None = None

    def run(self) -> None:
        """Start the tray icon  -  **blocks** on the main thread."""
        menu = pystray.Menu(
            pystray.MenuItem("Start", self._on_start, default=True),
            pystray.MenuItem("Pause", self._on_pause),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Mode",
                pystray.Menu(
                    pystray.MenuItem("Hand only", self._set_hand, checked=self._is_hand),
                    pystray.MenuItem("Voice only", self._set_voice, checked=self._is_voice),
                    pystray.MenuItem("Combined", self._set_combined, checked=self._is_combined),
                ),
            ),
            pystray.MenuItem(
                "Show preview",
                self._toggle_preview,
                checked=lambda _item: config.get("app", "show_preview"),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open settings", self._open_settings),
            pystray.MenuItem("Exit", self._on_exit),
        )

        self._icon = pystray.Icon(
            name="HandVoiceControl",
            icon=_make_icon("green"),
            title="Hand & Voice Control",
            menu=menu,
        )
        logger.info("System tray running")
        self._icon.run()  # Blocking

    def update_icon(self, state: str) -> None:
        """Change the tray icon colour (running/paused/stopped/voice_only)."""
        colour = _ICON_COLOURS.get(state, "#22c55e")
        if self._icon:
            self._icon.icon = _make_icon(colour)

    def _on_start(self, icon, item) -> None:
        if not self._orch.is_running():
            threading.Thread(target=self._orch.start, daemon=True, name="OrchestratorThread").start()
            self.update_icon("running")
            logger.info("Started via tray")

    def _on_pause(self, icon, item) -> None:
        if self._orch.is_running():
            self._orch.stop()
            self.update_icon("paused")
            logger.info("Paused via tray")

    def _set_hand(self, icon, item) -> None:
        self._orch.set_mode("hand")
        self.update_icon("running")

    def _set_voice(self, icon, item) -> None:
        self._orch.set_mode("voice")
        self.update_icon("voice_only")

    def _set_combined(self, icon, item) -> None:
        self._orch.set_mode("combined")
        self.update_icon("running")

    def _is_hand(self, item) -> bool:
        return config.get("app", "mode") == "hand"

    def _is_voice(self, item) -> bool:
        return config.get("app", "mode") == "voice"

    def _is_combined(self, item) -> bool:
        return config.get("app", "mode") == "combined"

    def _toggle_preview(self, icon, item) -> None:
        current = config.get("app", "show_preview")
        config.set_value("app", "show_preview", not current)
        logger.info("Preview toggled to %s", not current)

    @staticmethod
    def _open_settings(icon, item) -> None:
        import os
        os.startfile(str(config.SETTINGS_PATH))

    def _on_exit(self, icon, item) -> None:
        logger.info("Exit requested from tray")
        self._orch.stop()
        if self._icon:
            self._icon.stop()
