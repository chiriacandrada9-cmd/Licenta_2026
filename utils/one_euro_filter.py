"""
Adaptive low-pass filter for real-time signals.

The One-Euro Filter (1€ Filter) is an adaptive low-pass filter designed
for real-time signal smoothing in human-computer interaction. It adapts
its cutoff frequency based on signal velocity:

    - Slow/still -> low cutoff -> heavy smoothing (removes jitter)
    - Fast motion -> high cutoff -> responsive tracking (no lag)

Reference: Casiez, Roussel, Vogel (2012)
           "1€ Filter: A Simple Speed-Based Low-Pass Filter for Noisy
            Input in Interactive Systems" (CHI '12)
           https://gery.casiez.net/1euro/
"""

from __future__ import annotations

import math


class LowPassFilter:
    """Simple first-order low-pass filter."""

    __slots__ = ("_y", "_alpha", "_initialized")

    def __init__(self, alpha: float = 0.5) -> None:
        self._y: float = 0.0
        self._alpha = alpha
        self._initialized = False

    def filter(self, value: float, alpha: float | None = None) -> float:
        if not self._initialized:
            self._y = value
            self._initialized = True
            return value
        a = alpha if alpha is not None else self._alpha
        self._y = a * value + (1.0 - a) * self._y
        return self._y

    def last(self) -> float:
        return self._y

    def reset(self) -> None:
        self._initialized = False


class OneEuroFilter:
    """Filtru trece-jos adaptiv bazat pe viteza semnalului.

    Args:
        freq: frecventa de esantionare (Hz), implicit ~30 pt camera
        min_cutoff: cutoff minim (Hz) - cat de mult netezeste la repaus
        beta: coeficient viteza - cat de repede reactioneaza la miscare
        d_cutoff: cutoff pt filtrul derivatei (de obicei 1.0)
    """

    __slots__ = ("_freq", "_min_cutoff", "_beta", "_d_cutoff",
                 "_x_filter", "_dx_filter", "_last_time")

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self._freq = freq
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._last_time: float | None = None

    @staticmethod
    def _alpha(cutoff: float, rate: float) -> float:
        """Compute smoothing factor from cutoff frequency and sample rate."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, timestamp: float | None = None) -> float:
        """Filtreaza o valoare. Daca se da timestamp, calculeaza rata
        de esantionare dinamic. Returneaza valoarea filtrata."""
        if self._last_time is not None and timestamp is not None:
            dt = timestamp - self._last_time
            if dt > 0:
                self._freq = 1.0 / dt
        self._last_time = timestamp

        # Estimate derivative (velocity)
        prev = self._x_filter.last() if self._x_filter._initialized else x
        dx = (x - prev) * self._freq

        # Filter the derivative
        alpha_d = self._alpha(self._d_cutoff, self._freq)
        edx = self._dx_filter.filter(dx, alpha_d)

        # Adaptive cutoff: higher speed -> higher cutoff -> less smoothing
        cutoff = self._min_cutoff + self._beta * abs(edx)

        # Filter the signal
        alpha_x = self._alpha(cutoff, self._freq)
        return self._x_filter.filter(x, alpha_x)

    def reset(self) -> None:
        """Clear internal state."""
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = None
