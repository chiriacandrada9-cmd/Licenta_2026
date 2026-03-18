"""
config.py - Manager global de configurari.

Loads settings from settings.json, provides typed access to every parameter,
and exposes a save() function so the tray UI can persist changes at runtime.
"""

import json
import logging
import os
import copy
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT_DIR / "settings.json"
ASSETS_DIR = ROOT_DIR / "assets"
LOG_DIR = ROOT_DIR / "logs"
MODEL_DIR = Path(os.environ.get("LOCALAPPDATA", ROOT_DIR)) / "HandVoiceControl" / "models"
DEFAULTS: dict = {
    "camera": {
        "device_index": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "flip_horizontal": True,
    },
    "hand_tracking": {
        "max_hands": 1,
        "detection_confidence": 0.7,
        "tracking_confidence": 0.7,
        "smoothing_factor": 0.4,
        "cursor_speed_multiplier": 1.5,
    },
    "gestures": {
        "click_hold_frames": 3,
        "scroll_sensitivity": 40,
        "pinch_threshold": 0.04,
        "deadzone_radius": 0.01,
    },
    "voice": {
        "sample_rate": 16000,
        "vad_aggressiveness": 2,
        "silence_duration_ms": 800,
        "whisper_model": "base",
        "whisper_language": "ro",
        "whisper_device": "cpu",
        "wake_word": None,
    },
    "window_management": {
        "excluded_processes": ["explorer.exe"],
        "app_aliases": {
            "chrome": "chrome.exe",
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "explorer": "explorer.exe",
            "terminal": "wt.exe",
            "paint": "mspaint.exe",
        },
    },
    "app": {
        "mode": "combined",
        "show_preview": True,
        "start_minimized": False,
        "log_level": "INFO",
    },
}
_settings: dict = {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load(path: Path | None = None) -> dict:
    """
    Load settings from disk, merging with DEFAULTS so that any missing keys
    are filled in.  If the file does not exist, DEFAULTS are written out.
    """
    global _settings
    path = path or SETTINGS_PATH

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_settings = json.load(f)
            _settings = _deep_merge(DEFAULTS, user_settings)
            logger.info("Settings loaded from %s", path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load %s (%s) - using defaults", path, exc)
            _settings = copy.deepcopy(DEFAULTS)
    else:
        logger.info("No settings file found - creating %s with defaults", path)
        _settings = copy.deepcopy(DEFAULTS)
        save(path)

    return _settings


def save(path: Path | None = None) -> None:
    """Persist current settings to disk."""
    path = path or SETTINGS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_settings, f, indent=2, ensure_ascii=False)
    logger.info("Settings saved to %s", path)


def get(section: str, key: str | None = None):
    """
    Retrieve a setting value.

    Examples
    --------
    >>> config.get("camera", "width")
    640
    >>> config.get("camera")
    {"device_index": 0, "width": 640, ...}
    """
    if not _settings:
        load()
    if key is None:
        return _settings.get(section, {})
    return _settings.get(section, {}).get(key)


def set_value(section: str, key: str, value) -> None:
    """Update a single setting in memory (call save() to persist)."""
    if section not in _settings:
        _settings[section] = {}
    _settings[section][key] = value


def ensure_dirs() -> None:
    """Create required directories if they don't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
