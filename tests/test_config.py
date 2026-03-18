"""
tests/test_config.py - Teste pentru config.py
"""

import json
import pytest
import tempfile
from pathlib import Path

import config


class TestConfigLoad:
    def test_load_existing(self):
        cfg = config.load()
        assert "camera" in cfg
        assert "hand_tracking" in cfg
        assert "gestures" in cfg
        assert "voice" in cfg
        assert "window_management" in cfg
        assert "app" in cfg

    def test_default_values(self):
        cfg = config.load()
        assert cfg["camera"]["width"] == 640
        assert cfg["camera"]["height"] == 480
        assert cfg["hand_tracking"]["max_hands"] == 1
        assert cfg["voice"]["sample_rate"] == 16000

    def test_load_creates_file_when_missing(self, tmp_path):
        path = tmp_path / "test_settings.json"
        assert not path.exists()
        cfg = config.load(path)
        assert path.exists()
        assert "camera" in cfg

    def test_load_with_partial_settings(self, tmp_path):
        path = tmp_path / "partial.json"
        path.write_text('{"camera": {"width": 1280}}', encoding="utf-8")
        cfg = config.load(path)
        # Overridden value
        assert cfg["camera"]["width"] == 1280
        # Merged default
        assert cfg["camera"]["height"] == 480
        assert "voice" in cfg

    def test_load_with_corrupt_json(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("NOT VALID JSON {{{", encoding="utf-8")
        cfg = config.load(path)
        # Should fall back to defaults
        assert cfg["camera"]["width"] == 640


class TestConfigGet:
    def test_get_section(self):
        config.load()
        cam = config.get("camera")
        assert isinstance(cam, dict)
        assert "width" in cam

    def test_get_key(self):
        config.load()
        assert config.get("camera", "fps") == 30

    def test_get_missing_key(self):
        config.load()
        assert config.get("camera", "nonexistent") is None

    def test_get_missing_section(self):
        config.load()
        assert config.get("nonexistent") == {}


class TestConfigSet:
    def test_set_and_get(self):
        config.load()
        config.set_value("app", "mode", "voice")
        assert config.get("app", "mode") == "voice"
        # Reset
        config.set_value("app", "mode", "combined")

    def test_set_creates_section(self):
        config.load()
        config.set_value("new_section", "key", "value")
        assert config.get("new_section", "key") == "value"


class TestConfigSave:
    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "saved.json"
        config.load(path)
        config.set_value("camera", "width", 1920)
        config.save(path)

        # Reload from disk
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["camera"]["width"] == 1920


class TestDeepMerge:
    def test_nested_override(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 99}}
        result = config._deep_merge(base, override)
        assert result["a"]["b"] == 99
        assert result["a"]["c"] == 2  # Preserved
        assert result["d"] == 3       # Preserved

    def test_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        result = config._deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"] == 2
