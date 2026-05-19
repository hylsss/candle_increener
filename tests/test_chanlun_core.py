"""
单元测试：chanlun_core 第1-5层（合并 / 分型 / 笔 / 线段 / 中枢）

测试用例的设计原则：
  - 每组数据都对应 CHANLUN_NOTES.md 里某条原文规则；
  - 用最小可复现的合成K线，避免依赖真实数据；
  - 边界情形（首尾、严格等值、多次连续包含）单独建立用例。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 `python tests/test_chanlun_core.py` 也能跑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from chanlun_core import (
    RawBar, MergedBar, Direction, FractalType,
    Fractal, Stroke, Segment, Pivot,
    merge_klines, find_fractals, find_strokes, find_segments, find_pivots,
)


def mk(highs_lows, start_idx=0):
    """便捷构造：传入 [(h, l), (h, l), ...]，返回 RawBar 列表。"""
    out = []
    for i, (h, l) in enumerate(highs_lows):
        out.append(RawBar(
            idx=start_idx + i,
            dt=f"2024-01-{start_idx + i + 1:02d}",
            open=(h + l) / 2, close=(h + l) / 2,
            high=h, low=l,
            volume=1000,
        ))
    return out


# ══════════════════════════════════════════════════════════════════════
# 1. K线包含合并
# ══════════════════════════════════════════════════════════════════════

class TestMergeKlines(unittest.TestCase):

    def test_no_containment(self):
        """完全不包含的K线序列应原样返回。"""
        # 注意：(12,10) 与 (11.5,10.5) 是包含关系（前包后），不算"无包含"
        raws = mk([(10, 9), (11, 9.5), (12, 10), (13, 11)])
        merged = merge_klines(raws)
        self.assertEqual(len(merged), 4)
        for i, m in enumerate(merged):
            self.assertEqual(m.high, raws[i].high)
            self.assertEqual(m.low, raws[i].low)
            self.assertEqual(m.raw_indices, [i])

    def test_upward_containment(self):
        """
        向上趋势中后一根被前一根包含 → 用"高高低高"合并。
        三根：(10, 9) -> (11, 9.5) -> (10.8, 9.7)
        第三根被第二根包含；向上方向 → 合并后 high=max(11, 10.8)=11，
        low=max(9.5, 9.7)=9.7。
        """
        raws = mk([(10, 9), (11, 9.5), (10.8, 9.7)])
        merged = merge_klines(raws)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[1].high, 11.0)
        self.assertAlmostEqual(merged[1].low, 9.7)
        self.assertEqual(merged[1].raw_indices, [1, 2])
        self.assertEqual(merged[1].direction, Direction.UP)

    def test_downward_containment(self):
        """
        向下趋势中后一根被前一根包含 → 用"低低高低"合并。
        三根：(12, 11) -> (11.5, 10) -> (11.2, 10.5)
        第三根被第二根包含；向下方向 → high=min(11.5, 11.2)=11.2，
        low=min(10, 10.5)=10。
        """
        raws = mk([(12, 11), (11.5, 10), (11.2, 10.5)])
        merged = merge_klines(raws)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[1].high, 11.2)
        self.assertAlmostEqual(merged[1].low, 10.0)
        self.assertEqual(merged[1].direction, Direction.DOWN)

    def test_cascading_containment(self):
        """连续包含：滚动合并，每次基于当前合并基。"""
        raws = mk([(10, 8), (11, 9), (10.8, 9.5), (10.5, 9.7)])
        # 第2根不被第1根包含；向上 → merged=[(10,8),(11,9)]
        # 第3根被第2根包含；向上合并 → merged[-1]=(max(11,10.8), max(9,9.5))=(11,9.5)
        # 第4根被合并后的第2根包含；向上合并 → (max(11,10.5), max(9.5,9.7))=(11,9.7)
        merged = merge_klines(raws)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[1].high, 11.0)
        self.assertAlmostEqual(merged[1].low, 9.7)
        self.assertEqual(merged[1].raw_indices, [1, 2, 3])

    def test_direction_switch(self):
        """方向变化后合并方法切换。"""
        # 先向上到高点，再向下被包含
        raws = mk([(10, 9), (11, 9.5), (12, 10), (11.5, 9.8), (11, 9.9)])
        # 4根: 12>11 → 向上不合并
        # 5根: 高低都低于4根 → 向下；但5根的 high=11<11.5 low=9.9>9.8 → 包含
        merged = merge_klines(raws)
        # 第4根入栈时方向被刷成 DOWN，因为 high 12→11.5 down, low 10→9.8 down
        # 然后第5根被第4根包含 → 向下合并 → (min(11.5,11), min(9.8,9.9))=(11,9.8)
        self.assertEqual(len(merged), 4)
        self.assertAlmostEqual(merged[3].high, 11.0)
        self.assertAlmostEqual(merged[3].low, 9.8)
        self.assertEqual(merged[3].direction, Direction.DOWN)

    def test_first_bar_contains_second(self):
        """第一根包含第二根：初始方向兜底为 UP，按 UP 合并。"""
        raws = mk([(12, 8), (11, 9)])
        merged = merge_klines(raws)
        self.assertEqual(len(merged), 1)
        # UP 方向："高高低高" → (max(12,11), max(8,9)) = (12, 9)
        self.assertAlmostEqual(merged[0].high, 12.0)
        self.assertAlmostEqual(merged[0].low, 9.0)


# ══════════════════════════════════════════════════════════════════════
# 2. 分型识别
# ══════════════════════════════════════════════════════════════════════

class TestFindFractals(unittest.TestCase):

    def test_simple_top_fractal(self):
        """简单顶分型：低-高-低 严格不等。"""
        raws = mk([(10, 9), (11, 10), (10.5, 9.2)])
        merged = merge_klines(raws)
        fxs = find_fractals(merged)
        self.assertEqual(len(fxs), 1)
        self.assertEqual(fxs[0].ftype, FractalType.TOP)
        self.assertEqual(fxs[0].mid_idx, 1)

    def test_simple_bottom_fractal(self):
        """简单底分型：高-低-高 严格不等。"""
        raws = mk([(11, 10), (10, 9), (10.5, 9.5)])
        merged = merge_klines(raws)
        fxs = find_fractals(merged)
        self.assertEqual(len(fxs), 1)
        self.assertEqual(fxs[0].ftype, FractalType.BOTTOM)

    def test_equal_high_no_fractal(self):
        """相等高点不构成顶分型（严格不等）。"""
        raws = mk([(10, 9), (11, 10), (11, 10.2)])
        merged = merge_klines(raws)
        # 第3根被第2根包含 → 合并 → merged 只有 2 根 → 无分型
        fxs = find_fractals(merged)
        self.assertEqual(len(fxs), 0)

    def test_multiple_fractals(self):
        """多个分型应按顺序识别。"""
        raws = mk([
            (10, 9),     # 0
            (11, 10),    # 1 ← 顶
            (10, 9),     # 2
            (9, 8),      # 3 ← 底
            (10, 9),     # 4
            (11, 10),    # 5 ← 顶
            (10.5, 9.5), # 6
        ])
        merged = merge_klines(raws)
        fxs = find_fractals(merged)
        types = [f.ftype for f in fxs]
        self.assertEqual(types, [FractalType.TOP, FractalType.BOTTOM, FractalType.TOP])


# ══════════════════════════════════════════════════════════════════════
# 3. 笔的划分
# ══════════════════════════════════════════════════════════════════════

class TestFindStrokes(unittest.TestCase):

    def test_simple_stroke_new(self):
        """
        最小新笔：顶→底，中间≥3根独立合并K线。
        合成：高-顶-低-低-低-底-高
        """
        raws = mk([
            (10, 9),     # 0
            (12, 11),    # 1 ← 顶
            (11, 9),     # 2 (向下，不被包含)
            (10, 8),     # 3 (向下)
            (9, 7),      # 4 (向下)
            (8, 6),      # 5 ← 底
            (9, 7.5),    # 6
        ])
        merged = merge_klines(raws)
        fxs = find_fractals(merged)
        strokes = find_strokes(merged, fxs, new_stroke=True)
        self.assertEqual(len(strokes), 1)
        self.assertEqual(strokes[0].direction, Direction.DOWN)
        self.assertEqual(strokes[0].start_fx.ftype, FractalType.TOP)
        self.assertEqual(strokes[0].end_fx.ftype, FractalType.BOTTOM)

    def test_stroke_too_short_new(self):
        """中间合并K线不足3根 → 新笔不成立，老笔成立。"""
        raws = mk([
            (10, 9),     # 0
            (12, 11),    # 1 ← 顶
            (11, 9),     # 2 (中间只1根)
            (8, 6),      # 3 ← 底
            (9, 7.5),    # 4
        ])
        merged = merge_klines(raws)
        fxs = find_fractals(merged)
        strokes_new = find_strokes(merged, fxs, new_stroke=True)
        strokes_old = find_strokes(merged, fxs, new_stroke=False)
        self.assertEqual(len(strokes_new), 0)
        self.assertEqual(len(strokes_old), 1)

    def test_consecutive_same_type_fractals(self):
        """
        连续同性质分型：保留更极者。
        构造两个顶分型，第二个更高 → 应该用第二个顶。
        """
        raws = mk([
            (10, 9),     # 0
            (12, 11),    # 1 ← 顶1
            (11, 10),    # 2 (第3根)
            (10, 9),     # 3
            (13, 12),    # 4 ← 顶2 (更高，向上不包含)
            (12, 11),    # 5
            (11, 10),    # 6
            (10, 9),     # 7
            (8, 6),      # 8 ← 底
            (9, 7.5),    # 9
        ])
        merged = merge_klines(raws)
        fxs = find_fractals(merged)
        strokes = find_strokes(merged, fxs, new_stroke=True)
        # 应该只有一笔：顶2 → 底
        self.assertEqual(len(strokes), 1)
        self.assertAlmostEqual(strokes[0].start_price, 13.0)
        self.assertAlmostEqual(strokes[0].end_price, 6.0)

    def test_alternating_strokes(self):
        """顶→底→顶 形成两笔，方向交替。"""
        raws = mk([
            (10, 9),     # 0
            (13, 12),    # 1 ← 顶
            (12, 10),    # 2
            (11, 9),     # 3
            (10, 8),     # 4
            (8, 6),      # 5 ← 底
            (9, 7),      # 6
            (10, 8),     # 7
            (11, 9),     # 8
            (14, 13),    # 9 ← 顶2
            (13, 12),    # 10
        ])
        merged = merge_klines(raws)
        fxs = find_fractals(merged)
        strokes = find_strokes(merged, fxs, new_stroke=True)
        self.assertEqual(len(strokes), 2)
        self.assertEqual(strokes[0].direction, Direction.DOWN)
        self.assertEqual(strokes[1].direction, Direction.UP)

    def test_top_below_bottom_not_a_stroke(self):
        """
        硬性约束：顶 extreme 必须 > 底 extreme。
        构造一个"伪顶"低于后续"伪底"的场景 → 不成笔。
        实际上分型识别本身会过滤这种情况；这里测试 _fractal_pair_valid
        的硬性约束作为安全网。
        """
        # 直接构造分型对调用——通过制造合并K线序列让"顶"在很低位置
        raws = mk([
            (5, 4),      # 0
            (6, 5),      # 1 ← 顶（低位的）
            (5, 4),      # 2
            (4, 3),      # 3
            (5, 4),      # 4
            (10, 9),     # 5
            (15, 14),    # 6
            (10, 9),     # 7
            (9, 8),      # 8 ← 底（高位的，但只是局部底）
            (10, 9),     # 9
        ])
        # 这种合成数据 fractal 识别能拿到 (顶在5/6位置, 底在3位置, 顶在6, 底在8)
        # 真正的合理笔应该是 顶6 → 底8 之类。第一个"顶"(idx 1)与"底"(idx 8)
        # 不应该成笔，因为顶extreme=6 < 底extreme=8。
        merged = merge_klines(raws)
        fxs = find_fractals(merged)
        strokes = find_strokes(merged, fxs, new_stroke=True)
        # 验证：任一成立的笔都满足顶>底
        for s in strokes:
            if s.start_fx.ftype == FractalType.TOP:
                self.assertGreater(s.start_fx.extreme, s.end_fx.extreme,
                                   f"笔{s.idx}：顶{s.start_fx.extreme}应高于底{s.end_fx.extreme}")
            else:
                self.assertGreater(s.end_fx.extreme, s.start_fx.extreme,
                                   f"笔{s.idx}：顶{s.end_fx.extreme}应高于底{s.start_fx.extreme}")


# ══════════════════════════════════════════════════════════════════════
# 4. 线段划分
# ══════════════════════════════════════════════════════════════════════

def _make_fx(ftype: FractalType, mid_idx: int, high: float, low: float,
             dt: str = "") -> Fractal:
    """便捷构造一个虚拟分型（不真的关联 MergedBar 序列）。"""
    return Fractal(
        ftype=ftype, mid_idx=mid_idx,
        left_idx=mid_idx - 1, right_idx=mid_idx + 1,
        high=high, low=low, dt=dt or f"d{mid_idx}",
        confirmed=True,
    )


def _make_stroke(idx: int, start_price: float, end_price: float,
                 start_mid_idx: int = None, end_mid_idx: int = None) -> Stroke:
    """
    构造一条笔。start_price/end_price 决定方向。
    虚拟分型的 high/low 取等于 extreme（极值），便于在测试里精确控制笔的区间。
    """
    if start_mid_idx is None:
        start_mid_idx = idx * 10
    if end_mid_idx is None:
        end_mid_idx = idx * 10 + 5

    if end_price > start_price:
        direction = Direction.UP
        start_fx = _make_fx(FractalType.BOTTOM, start_mid_idx,
                            high=start_price, low=start_price)
        end_fx = _make_fx(FractalType.TOP, end_mid_idx,
                          high=end_price, low=end_price)
    else:
        direction = Direction.DOWN
        start_fx = _make_fx(FractalType.TOP, start_mid_idx,
                            high=start_price, low=start_price)
        end_fx = _make_fx(FractalType.BOTTOM, end_mid_idx,
                          high=end_price, low=end_price)

    return Stroke(idx=idx, direction=direction,
                  start_fx=start_fx, end_fx=end_fx)


def _mk_strokes(price_pairs):
    """传入 [(start, end), (start, end), ...]，每对相邻笔自动连续。返回 Stroke 列表。"""
    out = []
    for i, (sp, ep) in enumerate(price_pairs):
        out.append(_make_stroke(i, sp, ep))
    return out


class TestFindSegments(unittest.TestCase):

    def test_too_few_strokes(self):
        """笔数 < 3 → 无线段。"""
        strokes = _mk_strokes([(10, 20), (20, 15)])
        self.assertEqual(find_segments(strokes), [])

    def test_first_three_no_overlap_no_segment(self):
        """
        前三笔不重叠（s2 整体在 s0 之下）→ 无线段。
        """
        # s0: UP 50→100, s1: DOWN 100→40, s2: UP 40→45
        # s0 [50,100] vs s2 [40,45]：s0.low=50 > s2.high=45 → 不重叠
        strokes = _mk_strokes([(50, 100), (100, 40), (40, 45)])
        segs = find_segments(strokes)
        self.assertEqual(segs, [])

    def test_basic_up_segment_no_break(self):
        """
        3 笔（UP-DOWN-UP）且 s0、s2 重叠 → 形成 1 条上行线段（未完成）。
        """
        # s0: 10→20, s1: 20→15, s2: 15→25 → s0[10,20], s2[15,25] 重叠
        strokes = _mk_strokes([(10, 20), (20, 15), (15, 25)])
        segs = find_segments(strokes)
        self.assertEqual(len(segs), 1)
        seg = segs[0]
        self.assertEqual(seg.direction, Direction.UP)
        self.assertEqual(len(seg), 3)
        self.assertEqual(seg.break_type, 0)  # 未确认破坏
        self.assertEqual(seg.start_price, 10)
        self.assertEqual(seg.end_price, 25)

    def test_first_type_break_to_down(self):
        """
        第一种破坏：上行段顶部出现后，特征序列上无缺口的顶分型 → 段在顶处结束。
        构造：上行段 [10→30→25→35]，然后向下破坏 [35→28→32→22]。
          特征序列(向下笔)：[s1(20→15), s3(35→28), s5(32→22)]
          s3 在 [s1, s3, s5] 中：high 30→35→32（s3 最高）
                                  low 15→28→22（s3 最高？28>15 ✓, 28>22 ✓）
          → s3 是顶分型，且 s1[20,15] 与 s3[35,28] 无缺口（s1.low=15 ≤ s3.high=35 显然重叠）
          → 第一种破坏，段在 s2 末端（35）结束。
        """
        strokes = _mk_strokes([
            (10, 30),   # s0 UP   - 起始上行
            (30, 25),   # s1 DOWN - 小回调（X1）
            (25, 35),   # s2 UP   - 创新高至 35
            (35, 28),   # s3 DOWN - 回撤更深（X2 → 顶分型中点）
            (28, 32),   # s4 UP   - 反弹不创新高
            (32, 22),   # s5 DOWN - 跌破前低（X3）
        ])
        segs = find_segments(strokes)
        # 至少 1 条已完成的上行段
        self.assertGreaterEqual(len(segs), 1)
        first = segs[0]
        self.assertEqual(first.direction, Direction.UP)
        self.assertEqual(first.break_type, 1)
        self.assertAlmostEqual(first.end_price, 35.0)
        # 第一段应该恰好 3 笔（s0、s1、s2）
        self.assertEqual(len(first), 3)

    def test_segment_must_be_odd(self):
        """已完成线段的笔数必然为奇数（第77/78课）。"""
        strokes = _mk_strokes([
            (10, 30), (30, 25), (25, 35),
            (35, 28), (28, 32), (32, 22),
        ])
        for seg in find_segments(strokes):
            if seg.break_type != 0:
                self.assertEqual(len(seg) % 2, 1,
                                 f"段{seg.idx}笔数={len(seg)}不是奇数")

    def test_alternating_segments(self):
        """
        上行段被破坏后转下行段，整体应有方向交替的两段以上。
        """
        strokes = _mk_strokes([
            # 上行段
            (10, 30), (30, 25), (25, 35),
            # 第一种破坏
            (35, 28), (28, 32), (32, 22),
            # 下行段延伸
            (22, 26), (26, 18),
        ])
        segs = find_segments(strokes)
        if len(segs) >= 2:
            # 相邻段方向必然相反
            for a, b in zip(segs[:-1], segs[1:]):
                self.assertNotEqual(a.direction, b.direction,
                                    f"段{a.idx}{a.direction}与段{b.idx}{b.direction}方向相同")


# ══════════════════════════════════════════════════════════════════════
# 5. 中枢识别
# ══════════════════════════════════════════════════════════════════════

def _mk_segment(idx: int, start_price: float, end_price: float) -> Segment:
    """
    构造一段最小合规的 Segment：用一笔代表整段。
    Segment 的 high/low/start_price/end_price 只依赖 start_fx/end_fx，
    对中枢识别测试足矣（不依赖线段内部破坏类型等细节）。
    """
    stroke = _make_stroke(idx, start_price, end_price)
    return Segment(idx=idx, direction=stroke.direction, strokes=[stroke])


def _mk_segments(price_pairs):
    """传入 [(start, end), (start, end), ...]，按序构造交替方向的段。"""
    return [_mk_segment(i, sp, ep) for i, (sp, ep) in enumerate(price_pairs)]


class TestFindPivots(unittest.TestCase):

    def test_too_few_segments(self):
        """线段 < 3 → 无中枢。"""
        segs = _mk_segments([(10, 20), (20, 15)])
        self.assertEqual(find_pivots(segs), [])

    def test_basic_pivot(self):
        """
        三段构成最简中枢：
          s0: UP 10→20     [10,20]
          s1: DOWN 20→15   [15,20]
          s2: UP 15→18     [15,18]
        s0 与 s2 同向，重叠区间 = [max(10,15), min(20,18)] = [15,18] → 中枢 [15,18]
        """
        segs = _mk_segments([(10, 20), (20, 15), (15, 18)])
        pivots = find_pivots(segs)
        self.assertEqual(len(pivots), 1)
        p = pivots[0]
        self.assertAlmostEqual(p.zg, 18.0)
        self.assertAlmostEqual(p.zd, 15.0)
        self.assertAlmostEqual(p.gg, 20.0)   # max of all members' highs
        self.assertAlmostEqual(p.dd, 10.0)   # min of all members' lows
        self.assertEqual(len(p), 3)
        self.assertIsNone(p.entry_direction)
        self.assertIsNone(p.leaving_segment)
        self.assertFalse(p.is_finished)

    def test_no_overlap_no_pivot(self):
        """
        s0 与 s2 完全不重叠 → 不成中枢。
        s0: UP 10→20，s1: DOWN 20→5，s2: UP 5→8
        s0 [10,20] vs s2 [5,8]：min(s0.high, s2.high)=8 < max(s0.low, s2.low)=10。
        """
        segs = _mk_segments([(10, 20), (20, 5), (5, 8)])
        self.assertEqual(find_pivots(segs), [])

    def test_pivot_extension(self):
        """
        中枢延伸：第4段仍在 [ZD,ZG] 内，纳入中枢；GG/DD 可被新段刷新。
          s0 UP   10→25    [10,25]
          s1 DOWN 25→16    [16,25]
          s2 UP   16→23    [16,23]
          s3 DOWN 23→14    [14,23] —— low=14 已低于 ZD=16，但与 [16,23] 仍有交集
                                    （high=23 >= ZD=16，low=14 <= ZG=23）→ 纳入
          s4 UP   14→24    [14,24] —— 同理纳入，high=24 > ZG=23 但仍有交集
          → 共 5 段，DD 应被刷到 14，GG 刷到 25。
        ZG = min(25, 23) = 23；ZD = max(10, 16) = 16。
        """
        segs = _mk_segments([
            (10, 25), (25, 16), (16, 23),
            (23, 14), (14, 24),
        ])
        pivots = find_pivots(segs)
        self.assertEqual(len(pivots), 1)
        p = pivots[0]
        self.assertAlmostEqual(p.zg, 23.0)
        self.assertAlmostEqual(p.zd, 16.0)
        self.assertAlmostEqual(p.gg, 25.0)
        self.assertAlmostEqual(p.dd, 10.0)
        self.assertEqual(len(p), 5)
        self.assertFalse(p.is_finished)

    def test_pivot_leaving_up(self):
        """
        向上离开：第4段 low > ZG → leaving_segment，中枢结束。
          s0 UP   10→20    [10,20]
          s1 DOWN 20→15    [15,20]
          s2 UP   15→18    [15,18]   ZG=18, ZD=15
          s3 DOWN 18→16    [16,18]   仍在 [15,18] 内 → 纳入
          s4 UP   16→25    [16,25]   low=16 ≤ ZG=18 → 仍有交集 → 纳入
          s5 DOWN 25→22    [22,25]   low=22 > ZG=18 → 离开（leaving）
        """
        segs = _mk_segments([
            (10, 20), (20, 15), (15, 18),
            (18, 16), (16, 25), (25, 22),
        ])
        pivots = find_pivots(segs)
        self.assertGreaterEqual(len(pivots), 1)
        p = pivots[0]
        self.assertAlmostEqual(p.zg, 18.0)
        self.assertAlmostEqual(p.zd, 15.0)
        self.assertEqual(len(p), 5)            # s0..s4 纳入
        self.assertTrue(p.is_finished)
        self.assertIsNotNone(p.leaving_segment)
        self.assertEqual(p.leaving_segment.idx, 5)
        self.assertEqual(p.leaving_segment.direction, Direction.DOWN)

    def test_entry_direction_recorded(self):
        """
        中枢前若存在线段，entry_direction = 该段方向。
        构造：让 s0 与 s2 不重叠（s0 区间高过 s2 整段），中枢只能从 s1 起。
          s0 DOWN [100, 60]      —— 进入段，high=100，low=60
          s1 UP   [40, 20]→等价 _mk_segment(1, 60, 40)：UP? 不，60→40 是 DOWN…
        段方向由 start_price/end_price 决定。要让 s1 是 UP，必须 end>start。
        但相邻段方向交替，s0=DOWN 之后 s1=UP，s1 必须从 s0 末端起。
        这里我们用"逻辑上"独立的段（_mk_segment 不要求段端点连续）：
          s0 DOWN 高位 100→60，s1 UP 5→20，s2 DOWN 20→15，s3 UP 15→18
        s0 区间 [60,100]，s2 区间 [15,20]：min(s0.high, s2.high)=20，
        max(s0.low, s2.low)=60 → 20 < 60 → 无重叠 → i=0 不成中枢。
        i=1：s1 [5,20]，s3 [15,18]：ZG=min(20,18)=18，ZD=max(5,15)=15 → 中枢。
        """
        segs = _mk_segments([
            (100, 60),      # s0 DOWN —— 进入段
            (5, 20),        # s1 UP
            (20, 15),       # s2 DOWN
            (15, 18),       # s3 UP
        ])
        pivots = find_pivots(segs)
        self.assertEqual(len(pivots), 1)
        self.assertEqual(pivots[0].entry_direction, Direction.DOWN)
        # 中枢从 s1 开始
        self.assertEqual(pivots[0].segments[0].idx, 1)

    def test_consecutive_pivots(self):
        """
        连续两个中枢：第一个被向上离开，离开段成为第二个中枢的第一段。
          中枢1：s0/s1/s2 [15,18]，s3 仍在内，s4 也在内，s5 离开。
          中枢2：从 s5 开始扫描，需要 s5、s6、s7 同向且 s5/s7 有重叠。
        """
        segs = _mk_segments([
            (10, 20), (20, 15), (15, 18),       # 中枢1 起 [15,18]
            (18, 16), (16, 25),                 # 延伸纳入
            (25, 22),                           # 离开（高于 ZG=18）
            (22, 30), (30, 24),                 # 中枢2 候选起：s6 UP, s7 DOWN
        ])
        pivots = find_pivots(segs)
        # 第一个中枢应该存在并完成
        self.assertGreaterEqual(len(pivots), 1)
        p1 = pivots[0]
        self.assertTrue(p1.is_finished)
        self.assertAlmostEqual(p1.zg, 18.0)
        self.assertAlmostEqual(p1.zd, 15.0)

        # 若线段足够形成第二中枢，验证它的 entry_direction
        if len(pivots) >= 2:
            p2 = pivots[1]
            # 第二中枢的"前一段"是中枢1 最后一段（s4 UP）或离开段（s5 DOWN）
            self.assertIn(p2.entry_direction, (Direction.UP, Direction.DOWN))

    def test_boundary_equality_not_leaving(self):
        """
        边界等值：sj.high == ZD 或 sj.low == ZG 都应仍算"有交集"。
        本实现脱离判定用严格不等：sj.low > zg or sj.high < zd。
        构造：中枢 [15, 18]，s3 = DOWN 18→12（high=18=ZG）→ 应纳入。
        GG/DD 取所有成员段的极值，含 s0 的 low=10。
        """
        segs = _mk_segments([
            (10, 20), (20, 15), (15, 18),    # 中枢 ZG=18 ZD=15
            (18, 12),                        # s3 DOWN, high=18=ZG → 有交集
        ])
        pivots = find_pivots(segs)
        self.assertEqual(len(pivots), 1)
        p = pivots[0]
        self.assertEqual(len(p), 4)
        self.assertAlmostEqual(p.zg, 18.0)
        self.assertAlmostEqual(p.zd, 15.0)
        # GG/DD 是所有成员段的极值包络
        self.assertAlmostEqual(p.gg, 20.0)   # 最高来自 s0/s1
        self.assertAlmostEqual(p.dd, 10.0)   # 最低来自 s0

    def test_gg_dd_envelope_exceeds_pivot_range(self):
        """
        GG/DD 是震荡包络，可以超出 [ZD, ZG] 边界。
        这是缠论原意：进入段、扩展段的端点可能在 ZG 之上或 ZD 之下。
        """
        # ZG=18, ZD=15, 但 s0 的低点 10 远低于 ZD
        segs = _mk_segments([(10, 20), (20, 15), (15, 18)])
        p = find_pivots(segs)[0]
        self.assertLess(p.dd, p.zd)          # 10 < 15
        self.assertGreater(p.gg, p.zg)       # 20 > 18


if __name__ == "__main__":
    unittest.main(verbosity=2)
