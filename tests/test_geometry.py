"""
tests/test_geometry.py - Teste pentru utils/geometry.py
"""

import math
import pytest
from utils.geometry import distance_2d, distance_3d, angle_between, is_finger_extended, normalize_to_screen


class TestDistance2D:
    def test_zero_distance(self):
        assert distance_2d((0, 0), (0, 0)) == 0.0

    def test_known_triangle(self):
        assert distance_2d((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_negative_coords(self):
        assert distance_2d((-1, -1), (2, 3)) == pytest.approx(5.0)

    def test_floats(self):
        assert distance_2d((0.1, 0.2), (0.4, 0.6)) == pytest.approx(0.5)


class TestDistance3D:
    def test_zero(self):
        assert distance_3d((0, 0, 0), (0, 0, 0)) == 0.0

    def test_known(self):
        assert distance_3d((0, 0, 0), (1, 2, 2)) == pytest.approx(3.0)


class TestAngleBetween:
    def test_right_angle(self):
        # 90° angle at origin
        angle = angle_between((1, 0), (0, 0), (0, 1))
        assert angle == pytest.approx(90.0, abs=0.1)

    def test_straight_line(self):
        angle = angle_between((-1, 0), (0, 0), (1, 0))
        assert angle == pytest.approx(180.0, abs=0.1)

    def test_zero_angle(self):
        angle = angle_between((1, 0), (0, 0), (2, 0))
        assert angle == pytest.approx(0.0, abs=0.1)

    def test_45_degrees(self):
        angle = angle_between((1, 0), (0, 0), (1, 1))
        assert angle == pytest.approx(45.0, abs=0.1)

    def test_zero_length_vector(self):
        # Same point -> should return 0 and not crash
        assert angle_between((0, 0), (0, 0), (1, 1)) == 0.0


class TestIsFingerExtended:
    def test_finger_extended(self):
        # Tip is above (lower y) PIP -> extended
        tip = (0.5, 0.2)
        pip = (0.5, 0.5)
        mcp = (0.5, 0.7)
        assert is_finger_extended(tip, pip, mcp) is True

    def test_finger_curled(self):
        # Tip is below (higher y) PIP -> curled
        tip = (0.5, 0.8)
        pip = (0.5, 0.5)
        mcp = (0.5, 0.3)
        assert is_finger_extended(tip, pip, mcp) is False

    def test_thumb_right_hand_extended(self):
        # Right hand: thumb tip x < pip x -> extended
        tip = (0.2, 0.5)
        pip = (0.4, 0.5)
        mcp = (0.5, 0.5)
        assert is_finger_extended(tip, pip, mcp, is_thumb=True, handedness="Right") is True

    def test_thumb_right_hand_curled(self):
        tip = (0.6, 0.5)
        pip = (0.4, 0.5)
        mcp = (0.3, 0.5)
        assert is_finger_extended(tip, pip, mcp, is_thumb=True, handedness="Right") is False

    def test_thumb_left_hand_extended(self):
        # Left hand: thumb tip x > pip x -> extended
        tip = (0.7, 0.5)
        pip = (0.5, 0.5)
        mcp = (0.4, 0.5)
        assert is_finger_extended(tip, pip, mcp, is_thumb=True, handedness="Left") is True


class TestNormalizeToScreen:
    def test_center(self):
        x, y = normalize_to_screen(0.5, 0.5, 1920, 1080)
        assert x == 960
        assert y == 540

    def test_origin(self):
        x, y = normalize_to_screen(0.0, 0.0, 1920, 1080)
        assert x == 0
        assert y == 0

    def test_clamp_high(self):
        x, y = normalize_to_screen(1.5, 1.5, 1920, 1080)
        assert x == 1919
        assert y == 1079

    def test_clamp_low(self):
        x, y = normalize_to_screen(-0.1, -0.1, 1920, 1080)
        assert x == 0
        assert y == 0
