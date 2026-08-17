"""严格选股层测试：候选信号必须满足买卖点的结构定义。"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chanlun_core import (  # noqa: E402
    Direction, Divergence, Fractal, FractalType, Pivot, Segment, Stroke,
    TradeSignal,
)
from scan_b123 import _strict_signal_check  # noqa: E402


def _segment(idx: int, start: float, end: float, *, confirmed: bool = True) -> Segment:
    direction = Direction.UP if end > start else Direction.DOWN
    start_type = FractalType.BOTTOM if direction == Direction.UP else FractalType.TOP
    end_type = FractalType.TOP if direction == Direction.UP else FractalType.BOTTOM
    start_fx = Fractal(start_type, idx * 2, idx * 2, idx * 2,
                       start, start, f"2024-01-{idx * 2 + 1:02d}")
    end_fx = Fractal(end_type, idx * 2 + 1, idx * 2 + 1, idx * 2 + 1,
                     end, end, f"2024-01-{idx * 2 + 2:02d}")
    stroke = Stroke(idx, direction, start_fx, end_fx)
    return Segment(idx, direction, [stroke], break_type=1 if confirmed else 0)


def _downtrend_fixture():
    segments = [
        _segment(0, 35, 20), _segment(1, 20, 30), _segment(2, 30, 22),
        _segment(3, 19, 10), _segment(4, 10, 18), _segment(5, 18, 12),
        _segment(6, 12, 16), _segment(7, 16, 9), _segment(8, 14, 8),
    ]
    pivots = [
        Pivot(0, segments[0:3], zg=30, zd=22, gg=35, dd=20,
              leaving_segment=segments[3]),
        Pivot(1, segments[3:6], zg=18, zd=12, gg=19, dd=10,
              leaving_segment=segments[6]),
    ]
    div = Divergence(5, 8, Direction.DOWN, 10, 3, 10, 8, -1.2, -0.5)
    sig = TradeSignal(0, "B1", segments[8].end_fx.dt, 8, 8, divergence=div)
    return segments, pivots, sig


class TestStrictSignalCheck(unittest.TestCase):

    def test_b1_requires_two_separated_downtrend_pivots(self):
        segments, pivots, sig = _downtrend_fixture()
        ok, _, context = _strict_signal_check(
            sig, [sig], segments, pivots, np.array([9.0]), 0, 0, 5)
        self.assertTrue(ok)
        self.assertEqual(context["趋势中枢数"], 2)

        ok, reason, _ = _strict_signal_check(
            sig, [sig], segments, pivots[:1], np.array([9.0]), 0, 0, 5)
        self.assertFalse(ok)
        self.assertIn("两个", reason)

    def test_unconfirmed_segment_is_not_actionable(self):
        segments, pivots, sig = _downtrend_fixture()
        segments[8].break_type = 0
        ok, reason, _ = _strict_signal_check(
            sig, [sig], segments, pivots, np.array([9.0]), 0, 0, 5)
        self.assertFalse(ok)
        self.assertIn("尚未", reason)

    def test_b3_requires_completed_first_retest_above_zg(self):
        members = [_segment(0, 10, 20), _segment(1, 20, 15), _segment(2, 15, 18)]
        leaving = _segment(3, 19, 25)
        retest = _segment(4, 25, 20)
        segments = members + [leaving, retest]
        pivot = Pivot(0, members, zg=18, zd=15, gg=20, dd=10,
                      leaving_segment=leaving)
        sig = TradeSignal(0, "B3", retest.end_fx.dt, 20, 4, pivot_idx=0)

        ok, _, _ = _strict_signal_check(
            sig, [sig], segments, [pivot], np.array([21.0]), 0, 0, 5)
        self.assertTrue(ok)

        retest.break_type = 0
        ok, reason, _ = _strict_signal_check(
            sig, [sig], segments, [pivot], np.array([21.0]), 0, 0, 5)
        self.assertFalse(ok)
        self.assertIn("尚未", reason)


if __name__ == "__main__":
    unittest.main()
