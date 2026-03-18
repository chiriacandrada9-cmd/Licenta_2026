"""
tests/test_smoothing.py - Teste pentru utils/smoothing.py (filtrul 1€)
"""

import pytest
from utils.smoothing import Smoother


class TestSmoother:
    def test_first_sample_passes_through(self):
        s = Smoother(freq=30, deadzone=0.01)
        x, y = s.smooth(0.3, 0.7)
        assert x == pytest.approx(0.3)
        assert y == pytest.approx(0.7)

    def test_deadzone_suppression(self):
        s = Smoother(freq=30, deadzone=0.05)
        s.smooth(0.5, 0.5)  # Init

        # Tiny movement within deadzone  -  should be suppressed
        x, y = s.smooth(0.51, 0.51)
        assert x == pytest.approx(0.5)
        assert y == pytest.approx(0.5)

    def test_movement_beyond_deadzone(self):
        s = Smoother(freq=30, deadzone=0.01)
        s.smooth(0.5, 0.5)  # Init

        # Larger movement  -  should produce a different result
        x, y = s.smooth(0.7, 0.7)
        assert x != pytest.approx(0.5)
        assert y != pytest.approx(0.5)

    def test_smoothing_converges(self):
        s = Smoother(freq=30, deadzone=0.0, min_cutoff=5.0, beta=0.5)
        s.smooth(0.0, 0.0)

        # Feed the same target repeatedly  -  should converge
        for _ in range(50):
            x, y = s.smooth(1.0, 1.0)

        assert x == pytest.approx(1.0, abs=0.05)
        assert y == pytest.approx(1.0, abs=0.05)

    def test_clamp_output(self):
        s = Smoother(freq=30, deadzone=0.0, min_cutoff=100.0, beta=1.0)
        s.smooth(0.0, 0.0)

        # Very large jump  -  should be clamped to [0, 1]
        x, y = s.smooth(1.0, 1.0)
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0

    def test_reset(self):
        s = Smoother(freq=30, deadzone=0.01)
        s.smooth(0.5, 0.5)
        s.reset()

        # After reset, next sample should pass through cleanly
        x, y = s.smooth(0.8, 0.2)
        assert x == pytest.approx(0.8)
        assert y == pytest.approx(0.2)


class TestOneEuroFilter:
    """Tests for the underlying One-Euro Filter."""

    def test_filter_reduces_jitter(self):
        """Static input with noise should be smoothed to near-constant."""
        from utils.one_euro_filter import OneEuroFilter
        f = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.0)

        # Seed with 0.5
        f(0.5)

        # Add jitter around 0.5
        outputs = []
        import random
        random.seed(42)
        for _ in range(100):
            noisy = 0.5 + random.gauss(0, 0.02)
            outputs.append(f(noisy))

        # Output variance should be much less than input variance
        mean_out = sum(outputs) / len(outputs)
        var_out = sum((o - mean_out) ** 2 for o in outputs) / len(outputs)
        assert var_out < 0.02 ** 2  # Less than input noise variance

    def test_filter_tracks_fast_movement(self):
        """Large step should be tracked quickly with high beta."""
        from utils.one_euro_filter import OneEuroFilter
        f = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=1.0)

        f(0.0)
        # Step to 1.0  -  high beta should make filter responsive
        for _ in range(10):
            result = f(1.0)

        assert result > 0.9  # Should have tracked most of the way
