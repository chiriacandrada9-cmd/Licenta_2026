"""
tests/test_command_parser.py - Teste pentru modules/command_parser.py
"""

import pytest
import config
from modules.command_parser import CommandParser, ActionType, ParsedCommand

# Ensure config is loaded before parser is used
config.load()


@pytest.fixture
def parser():
    return CommandParser()


class TestMouseCommands:
    def test_click(self, parser):
        r = parser.parse("click")
        assert r.action == ActionType.CLICK

    def test_click_romanian(self, parser):
        r = parser.parse("apasă")
        assert r.action == ActionType.CLICK

    def test_right_click(self, parser):
        r = parser.parse("right click")
        assert r.action == ActionType.RIGHT_CLICK

    def test_right_click_romanian(self, parser):
        r = parser.parse("click dreapta")
        assert r.action == ActionType.RIGHT_CLICK

    def test_double_click(self, parser):
        r = parser.parse("double click")
        assert r.action == ActionType.DOUBLE_CLICK

    def test_double_click_romanian(self, parser):
        r = parser.parse("dublu click")
        assert r.action == ActionType.DOUBLE_CLICK

    def test_scroll_up(self, parser):
        r = parser.parse("scroll up")
        assert r.action == ActionType.SCROLL_UP

    def test_scroll_down_romanian(self, parser):
        r = parser.parse("scroll jos")
        assert r.action == ActionType.SCROLL_DOWN


class TestKeyboardCommands:
    def test_enter(self, parser):
        r = parser.parse("enter")
        assert r.action == ActionType.PRESS_KEY
        assert r.argument == "enter"

    def test_escape(self, parser):
        r = parser.parse("escape")
        assert r.action == ActionType.PRESS_KEY
        assert r.argument == "escape"

    def test_tab(self, parser):
        r = parser.parse("tab")
        assert r.action == ActionType.PRESS_KEY
        assert r.argument == "tab"

    def test_copy(self, parser):
        r = parser.parse("copy")
        assert r.action == ActionType.HOTKEY
        assert r.argument == "copy"

    def test_paste_romanian(self, parser):
        r = parser.parse("lipește")
        assert r.action == ActionType.HOTKEY
        assert r.argument == "paste"

    def test_undo(self, parser):
        r = parser.parse("undo")
        assert r.action == ActionType.HOTKEY
        assert r.argument == "undo"

    def test_select_all(self, parser):
        r = parser.parse("select all")
        assert r.action == ActionType.HOTKEY
        assert r.argument == "select_all"

    def test_type_text(self, parser):
        r = parser.parse("type hello world")
        assert r.action == ActionType.TYPE_TEXT
        assert r.argument == "hello world"

    def test_type_text_romanian(self, parser):
        r = parser.parse("tastează salut lume")
        assert r.action == ActionType.TYPE_TEXT
        assert r.argument == "salut lume"


class TestWindowCommands:
    def test_open_app(self, parser):
        r = parser.parse("open notepad")
        assert r.action == ActionType.OPEN_APP
        assert r.argument == "notepad"

    def test_open_app_romanian(self, parser):
        r = parser.parse("deschide chrome")
        assert r.action == ActionType.OPEN_APP
        assert r.argument == "chrome"

    def test_minimize(self, parser):
        r = parser.parse("minimize")
        assert r.action == ActionType.MINIMIZE

    def test_maximize_romanian(self, parser):
        r = parser.parse("maximizează")
        assert r.action == ActionType.MAXIMIZE

    def test_close_window(self, parser):
        r = parser.parse("close window")
        assert r.action == ActionType.CLOSE_WINDOW

    def test_next_window(self, parser):
        r = parser.parse("next window")
        assert r.action == ActionType.NEXT_WINDOW

    def test_show_desktop(self, parser):
        r = parser.parse("desktop")
        assert r.action == ActionType.SHOW_DESKTOP


class TestAppControlCommands:
    def test_stop(self, parser):
        r = parser.parse("stop")
        assert r.action == ActionType.PAUSE_CONTROL

    def test_stop_romanian(self, parser):
        r = parser.parse("oprește")
        assert r.action == ActionType.PAUSE_CONTROL

    def test_start(self, parser):
        r = parser.parse("start")
        assert r.action == ActionType.RESUME_CONTROL

    def test_voice_mode(self, parser):
        r = parser.parse("voice mode")
        assert r.action == ActionType.SET_MODE
        assert r.argument == "voice"

    def test_hand_mode_romanian(self, parser):
        r = parser.parse("mod mână")
        assert r.action == ActionType.SET_MODE
        assert r.argument == "hand"

    def test_combined_mode(self, parser):
        r = parser.parse("combined mode")
        assert r.action == ActionType.SET_MODE
        assert r.argument == "combined"


class TestDictationCommands:
    def test_dictation_on(self, parser):
        r = parser.parse("dictation")
        assert r.action == ActionType.DICTATION_ON

    def test_dictation_off(self, parser):
        r = parser.parse("stop dictation")
        assert r.action == ActionType.DICTATION_OFF

    def test_dictation_off_romanian(self, parser):
        r = parser.parse("stop dictare")
        assert r.action == ActionType.DICTATION_OFF


class TestEdgeCases:
    def test_empty_string(self, parser):
        r = parser.parse("")
        assert r.action == ActionType.UNKNOWN

    def test_whitespace_only(self, parser):
        r = parser.parse("   ")
        assert r.action == ActionType.UNKNOWN

    def test_unknown_command(self, parser):
        r = parser.parse("foobar gibberish")
        assert r.action == ActionType.UNKNOWN

    def test_case_insensitive(self, parser):
        r = parser.parse("CLICK")
        assert r.action == ActionType.CLICK

    def test_preserves_raw_text(self, parser):
        r = parser.parse("Open Notepad")
        assert r.raw_text == "Open Notepad"

    def test_extra_whitespace(self, parser):
        r = parser.parse("  scroll  up  ")
        assert r.action == ActionType.SCROLL_UP
