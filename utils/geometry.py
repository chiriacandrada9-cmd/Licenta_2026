"""
Math helpers for hand landmark analysis.

Provides Euclidean distance, angle calculations, finger extension checks,
and ROI-aware coordinate normalization for screen mapping.
"""

from __future__ import annotations

import math


def distance_2d(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def distance_3d(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
) -> float:
    """Euclidean distance between two 3-D points."""
    return math.sqrt(
        (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2
    )


def angle_between(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> float:
    """
    Angle (in degrees) at vertex *p2* formed by the line segments p1->p2 and
    p2->p3.
    """
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def is_finger_extended(
    tip: tuple[float, float],
    pip: tuple[float, float],
    mcp: tuple[float, float],
    *,
    is_thumb: bool = False,
    handedness: str = "Right",
    margin: float = 0.0,
) -> bool:
    """
    Determine whether a finger is extended.

    For non-thumb fingers the tip must be above (lower y) the PIP joint
    by at least ``margin``.  For the thumb, we compare x-coordinates
    (direction depends on handedness) with the same margin.

    The *margin* parameter adds hysteresis: use a positive margin to
    require a clearer separation before declaring extended, and a
    negative margin to make it easier to remain extended.
    """
    if is_thumb:
        if handedness == "Right":
            return tip[0] < pip[0] - margin
        else:
            return tip[0] > pip[0] + margin
    return tip[1] < pip[1] - margin


def normalize_to_screen(
    x: float,
    y: float,
    screen_w: int,
    screen_h: int,
    *,
    roi_x_min: float = 0.0,
    roi_x_max: float = 1.0,
    roi_y_min: float = 0.0,
    roi_y_max: float = 1.0,
) -> tuple[int, int]:
    """
    Convert normalised hand coordinates to pixel screen coordinates.

    The ROI (Region of Interest) parameters define the usable sub-region
    of the camera frame where the hand typically moves.  Coordinates
    within [roi_min, roi_max] are stretched to fill the full screen,
    so the cursor can reach all four corners.

    Coordinates outside the ROI are clamped to the screen edges.
    """
    # Remap from ROI to 0-1
    roi_w = roi_x_max - roi_x_min
    roi_h = roi_y_max - roi_y_min

    if roi_w > 0:
        nx = (x - roi_x_min) / roi_w
    else:
        nx = x

    if roi_h > 0:
        ny = (y - roi_y_min) / roi_h
    else:
        ny = y

    # Clamp to [0, 1]
    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))

    # Scale to screen pixels
    sx = int(max(0, min(screen_w - 1, nx * screen_w)))
    sy = int(max(0, min(screen_h - 1, ny * screen_h)))
    return sx, sy
