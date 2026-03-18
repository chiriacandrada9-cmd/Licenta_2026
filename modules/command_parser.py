"""
Voice command -> action mapper.

Parses transcribed text (Romanian or English) into structured
ParsedCommand objects that the orchestrator can execute.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ActionType(Enum):
    # Mouse
    CLICK = "click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    # Keyboard
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    HOTKEY = "hotkey"
    # Windows
    OPEN_APP = "open_app"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    CLOSE_WINDOW = "close_window"
    NEXT_WINDOW = "next_window"
    SHOW_DESKTOP = "show_desktop"
    # App control
    SET_MODE = "set_mode"
    PAUSE_CONTROL = "pause_control"
    RESUME_CONTROL = "resume_control"
    # Dictation
    DICTATION_ON = "dictation_on"
    DICTATION_OFF = "dictation_off"
    # Fallback
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    """Result of parsing a voice transcription."""

    action: ActionType
    argument: str | None = None   # e.g. text to type, app name, mode name
    raw_text: str = ""            # Original transcription

def _normalize(text: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace."""
    text = text.lower().strip()
    # Remove diacritics (ă->a, ț->t, etc.) for fuzzy matching
    nfkd = unicodedata.normalize("NFKD", text)
    without_diacritics = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", without_diacritics)

# Each entry: (list_of_trigger_phrases, action_type, has_argument)
# Phrases with {arg} capture everything after the trigger as the argument.

_COMMANDS: list[tuple[list[str], ActionType, bool]] = [
    # --- Dictation ---
    (["dictare", "dictation", "start dictation", "incepe dictare"], ActionType.DICTATION_ON, False),
    (["stop dictare", "stop dictation", "opreste dictare"], ActionType.DICTATION_OFF, False),

    # --- App control ---
    (["opreste", "stop", "pauza", "pause"], ActionType.PAUSE_CONTROL, False),
    (["porneste", "start", "resume", "continua"], ActionType.RESUME_CONTROL, False),
    (["mod voce", "voice mode", "mod vocal"], ActionType.SET_MODE, True),  # arg = "voice"
    (["mod mana", "hand mode", "mod maina"], ActionType.SET_MODE, True),   # arg = "hand"
    (["mod combinat", "combined mode"], ActionType.SET_MODE, True),         # arg = "combined"

    # --- Mouse (multi-word first) ---
    (["dublu click", "double click", "dublu clic"], ActionType.DOUBLE_CLICK, False),
    (["click dreapta", "right click", "clic dreapta"], ActionType.RIGHT_CLICK, False),
    (["scroll sus", "scroll up", "deruleaza sus"], ActionType.SCROLL_UP, False),
    (["scroll jos", "scroll down", "deruleaza jos"], ActionType.SCROLL_DOWN, False),
    (["click", "clic", "apasa"], ActionType.CLICK, False),

    # --- Keyboard shortcuts ---
    (["copiaza", "copy"], ActionType.HOTKEY, True),     # arg = "copy"
    (["lipeste", "paste"], ActionType.HOTKEY, True),     # arg = "paste"
    (["anuleaza", "undo"], ActionType.HOTKEY, True),     # arg = "undo"
    (["selecteaza tot", "select all"], ActionType.HOTKEY, True),  # arg = "select_all"
    (["enter"], ActionType.PRESS_KEY, True),              # arg = "enter"
    (["escape", "inchide"], ActionType.PRESS_KEY, True),  # arg = "escape"
    (["tab"], ActionType.PRESS_KEY, True),                # arg = "tab"

    # --- Windows ---
    (["minimizeaza", "minimize"], ActionType.MINIMIZE, False),
    (["maximizeaza", "maximize"], ActionType.MAXIMIZE, False),
    (["inchide fereastra", "close window"], ActionType.CLOSE_WINDOW, False),
    (["urmatoarea fereastra", "next window"], ActionType.NEXT_WINDOW, False),
    (["desktop", "show desktop", "arata desktop"], ActionType.SHOW_DESKTOP, False),

    # --- Open app (must be last  -  captures argument) ---
    (["deschide", "open"], ActionType.OPEN_APP, True),

    # --- Type text (must be last  -  captures argument) ---
    (["tasteaza", "scrie", "type"], ActionType.TYPE_TEXT, True),
]

class CommandParser:
    """Parse a transcribed voice string into a :class:`ParsedCommand`."""

    def parse(self, text: str) -> ParsedCommand:
        raw = text
        norm = _normalize(text)

        if not norm:
            return ParsedCommand(action=ActionType.UNKNOWN, raw_text=raw)

        for triggers, action, has_arg in _COMMANDS:
            for trigger in triggers:
                if norm.startswith(trigger):
                    argument = None
                    if has_arg:
                        rest = norm[len(trigger):].strip()
                        argument = rest if rest else self._default_arg(action, trigger)
                    cmd = ParsedCommand(action=action, argument=argument, raw_text=raw)
                    logger.info("Parsed: '%s' -> %s (arg=%s)", raw, action.value, argument)
                    return cmd

        # No match
        logger.debug("Unknown command: '%s'", raw)
        return ParsedCommand(action=ActionType.UNKNOWN, raw_text=raw)

    @staticmethod
    def _default_arg(action: ActionType, trigger: str) -> str | None:
        """Supply a default argument when the trigger matches but no text follows."""
        defaults = {
            # Mode triggers carry their mode name
            "mod voce": "voice",
            "voice mode": "voice",
            "mod vocal": "voice",
            "mod mana": "hand",
            "hand mode": "hand",
            "mod maina": "hand",
            "mod combinat": "combined",
            "combined mode": "combined",
            # Hotkey names
            "copiaza": "copy",
            "copy": "copy",
            "lipeste": "paste",
            "paste": "paste",
            "anuleaza": "undo",
            "undo": "undo",
            "selecteaza tot": "select_all",
            "select all": "select_all",
            # Key names
            "enter": "enter",
            "escape": "escape",
            "inchide": "escape",
            "tab": "tab",
        }
        return defaults.get(trigger)
