"""
Filtre de netezire pentru cursor.

Implementează un smoother de coordonate bazat pe Filtrul One-Euro (1€ Filter)
cu supresor de zonă moartă și rejecție outlier. Filtrul 1€ este un filtru
trece-jos adaptiv care ajustează frecvența de tăiere în funcție de viteza
semnalului:

    - Mână lentă/statică -> cutoff mic -> netezire puternică (cursor stabil)
    - Mișcare rapidă -> cutoff mare -> tracking responsiv (fără lag)

Rejecția outlier-ilor detectează salturi imposibile (ex. glitch-uri de tracking)
și le limitează pentru a preveni teleportarea cursorului.
"""

from __future__ import annotations

import math

from utils.one_euro_filter import OneEuroFilter


class Smoother:
    """Smoother de coordonate bazat pe filtrul 1€, cu zona moarta
    si rejectie outlier.

    Args:
        freq: frecventa esantionare (Hz)
        min_cutoff: cutoff minim, stabilitate la repaus
        beta: coeficient viteza, reactivitate la miscare
        deadzone: miscari mai mici decat asta sunt ignorate (spatiu 0-1)
        max_jump: salt maxim per cadru inainte de clampare (anti-teleportare)
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 2.5,
        beta: float = 0.8,
        deadzone: float = 0.003,
        max_jump: float = 0.15,
    ) -> None:
        self.deadzone = deadzone
        self.max_jump = max_jump
        self._filter_x = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta)
        self._filter_y = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta)

        self._prev_raw_x: float | None = None
        self._prev_raw_y: float | None = None
        self._prev_x: float | None = None
        self._prev_y: float | None = None

    def reset(self) -> None:
        """Reseteaza starea interna (cand mana dispare din cadru)."""
        self._prev_raw_x = None
        self._prev_raw_y = None
        self._prev_x = None
        self._prev_y = None
        self._filter_x.reset()
        self._filter_y.reset()

    def smooth(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        """Aplica filtrul 1€ pe coordonate raw normalizate.
        Include rejectie outlier (salturi > max_jump sunt limitate)
        si zona moarta (miscari prea mici sunt suprimate la output).
        Returneaza (smoothed_x, smoothed_y) in spatiu 0-1."""
        # --- Outlier rejection ---
        # Cap impossibly large single-frame jumps (tracking glitches)
        if self._prev_raw_x is not None and self._prev_raw_y is not None:
            dx = raw_x - self._prev_raw_x
            dy = raw_y - self._prev_raw_y
            dist = math.hypot(dx, dy)

            if dist > self.max_jump:
                # Clamp to max_jump distance in the same direction
                scale = self.max_jump / dist
                raw_x = self._prev_raw_x + dx * scale
                raw_y = self._prev_raw_y + dy * scale

        self._prev_raw_x = raw_x
        self._prev_raw_y = raw_y

        # --- 1€ Filter ---
        # Always run the filter so it sees every sample (critical for
        # accurate velocity estimation and smooth transitions).
        filtered_x = self._filter_x(raw_x)
        filtered_y = self._filter_y(raw_y)

        # Clamp to [0, 1]
        filtered_x = max(0.0, min(1.0, filtered_x))
        filtered_y = max(0.0, min(1.0, filtered_y))

        if self._prev_x is None or self._prev_y is None:
            # First sample  -  pass through.
            self._prev_x = filtered_x
            self._prev_y = filtered_y
            return filtered_x, filtered_y

        # --- Dead-zone ---
        # Suppress output if filtered position barely moved.
        dx = filtered_x - self._prev_x
        dy = filtered_y - self._prev_y
        dist = math.hypot(dx, dy)

        if dist < self.deadzone:
            return self._prev_x, self._prev_y

        self._prev_x = filtered_x
        self._prev_y = filtered_y

        return filtered_x, filtered_y
