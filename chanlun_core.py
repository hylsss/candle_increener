"""
chanlun_core.py
────────────────────────────────────
缠论核心引擎：原始K线 → 合并K线 → 分型 → 笔 → 线段 → 中枢 → 买卖点

实现严格依据《教你炒股票》108课原文。所有边界条件与硬性规则在
CHANLUN_NOTES.md 中有完整引用，本文件每一步实现都标注对应课次。

本文件实现进度：
  Phase 1.1  merge_klines     ── K线包含合并（第62/65/77课）         ✅
  Phase 1.2  find_fractals    ── 顶/底分型识别（第62/77课）          ✅
  Phase 1.3  find_strokes     ── 笔的划分（第77课3步法）             ✅
  Phase 1.4  find_segments    ── 线段划分（第67/71/77/78课）         ✅
  Phase 1.5  find_pivots      ── 中枢识别（第17/20/29课）            ✅
  Phase 1.6  detect_divergence── MACD背驰（第24/25/27/50课）         ✅
  Phase 1.7  find_signals     ── 1/2/3类买卖点（第17/20/24/53课）    ✅
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════

class Direction(Enum):
    """趋势方向 — 用于合并方向状态机与笔/线段方向标注。"""
    UP = 1
    DOWN = -1


class FractalType(Enum):
    TOP = "TOP"      # 顶分型
    BOTTOM = "BOT"   # 底分型


@dataclass
class RawBar:
    """原始K线。idx 是它在原始序列里的下标（用于追溯）。"""
    idx: int
    dt: str           # 日期字符串，例如 "2024-01-15"；缠论本身不关心精确时间
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self):
        if self.high < self.low:
            raise ValueError(f"RawBar idx={self.idx}: high({self.high}) < low({self.low})")


@dataclass
class MergedBar:
    """
    合并K线（处理过包含关系后的等价K线）。
    high / low 是合并区间；raw_indices 记录由哪些原始K线合成而来；
    direction 是该合并发生时的方向（第65课：先定方向再合）。
    """
    idx: int                           # 合并序列里的下标
    dt: str                            # 取该合并段最后一根原始K线的日期
    high: float
    low: float
    direction: Direction               # 合并发生时的滚动方向
    raw_indices: list[int] = field(default_factory=list)

    @property
    def raw_count(self) -> int:
        return len(self.raw_indices)


@dataclass
class Fractal:
    """
    分型（基于合并K线）。
    mid_idx 指向合并序列里的中间那根；left_idx / right_idx 指两侧。
    high / low 是中间那根的高低（顶分型用 high，底分型用 low 作"极值"）。
    confirmed 表示右侧K线是否已经收完（True = 已确认；False = 潜在）。
    """
    ftype: FractalType
    mid_idx: int                       # 在合并序列里的下标
    left_idx: int
    right_idx: int
    high: float                        # 中间合并K线的最高
    low: float                         # 中间合并K线的最低
    dt: str                            # 中间合并K线的日期
    confirmed: bool = True

    @property
    def extreme(self) -> float:
        """该分型的特征价：顶分型取 high，底分型取 low。"""
        return self.high if self.ftype == FractalType.TOP else self.low


@dataclass
class Stroke:
    """
    笔。方向：先底后顶 = UP；先顶后底 = DOWN（第77课）。
    start_fx / end_fx 是构成本笔的两端分型。
    """
    idx: int                           # 笔序列里的下标
    direction: Direction
    start_fx: Fractal
    end_fx: Fractal

    @property
    def start_price(self) -> float:
        return self.start_fx.extreme

    @property
    def end_price(self) -> float:
        return self.end_fx.extreme

    @property
    def high(self) -> float:
        return max(self.start_price, self.end_price)

    @property
    def low(self) -> float:
        return min(self.start_price, self.end_price)


@dataclass
class Segment:
    """
    线段。方向 = 起点到终点的方向（向上=底→顶，向下=顶→底）。
    strokes 是构成本线段的笔列表（笔数为奇数，至少 3）。
    break_type: 1 = 第一种破坏（无缺口），2 = 第二种破坏（有缺口），
                0 = 仍在延伸/未完成。
    """
    idx: int
    direction: Direction
    strokes: list[Stroke]
    break_type: int = 0

    @property
    def start_fx(self) -> Fractal:
        return self.strokes[0].start_fx

    @property
    def end_fx(self) -> Fractal:
        return self.strokes[-1].end_fx

    @property
    def start_price(self) -> float:
        return self.start_fx.extreme

    @property
    def end_price(self) -> float:
        return self.end_fx.extreme

    @property
    def high(self) -> float:
        return max(self.start_price, self.end_price)

    @property
    def low(self) -> float:
        return min(self.start_price, self.end_price)

    def __len__(self) -> int:
        return len(self.strokes)


@dataclass
class Pivot:
    """
    中枢（第17/20/29课）。

    segments       构成中枢的连续线段（不含 leaving_segment）。
    zg / zd        中枢上沿 / 下沿。第20课公式：ZG=min(g1,g2)，ZD=max(d1,d2)，
                   只用"前两段同向次级别走势"的高低点决定。本实现中线段在
                   时间上自然交替方向，因此 g1/g2 取 segments[0] 与 segments[2]
                   的 high；d1/d2 同理。"定中枢以后不再因后续 Z 段而改变"。
    gg / dd        中枢震荡的最高 / 最低边界。GG=max(gn)，DD=min(dn)，遍历
                   中枢中所有 Z 段（含 segments[0..]，本实现遍历全部成员段）。
    entry_direction 进入中枢前一段线段的方向；UP=上升中枢、DOWN=下跌中枢。
                   若中枢从序列开头开始（无前驱段），值为 None。
    leaving_segment 离开中枢的那一段线段（首段满足 low>ZG 或 high<ZD）。
                   未离开（中枢仍在延伸到序列末尾）时为 None。
    """
    idx: int
    segments: list[Segment]
    zg: float
    zd: float
    gg: float
    dd: float
    entry_direction: Direction | None = None
    leaving_segment: Segment | None = None

    @property
    def high(self) -> float:
        """中枢上沿（= ZG）。买卖点判定时所用的"硬边界"。"""
        return self.zg

    @property
    def low(self) -> float:
        """中枢下沿（= ZD）。"""
        return self.zd

    @property
    def start_dt(self) -> str:
        return self.segments[0].start_fx.dt

    @property
    def end_dt(self) -> str:
        """中枢最后一段结束时间。若已有 leaving，则中枢边界仍取最后成员段。"""
        return self.segments[-1].end_fx.dt

    @property
    def is_finished(self) -> bool:
        """是否已出现离开段。"""
        return self.leaving_segment is not None

    def __len__(self) -> int:
        return len(self.segments)


# ══════════════════════════════════════════════════════════════════════
# 1. K线包含合并 — 第62/65/77课
# ══════════════════════════════════════════════════════════════════════

def _is_contained(a_high: float, a_low: float,
                  b_high: float, b_low: float) -> bool:
    """
    判定 a 和 b 两根K线（任一为合并K线）是否构成包含关系。
    a 包含 b 或 b 包含 a 都算。等价边界（高=高 且 低=低）也算包含。
    """
    return (a_high >= b_high and a_low <= b_low) or \
           (b_high >= a_high and b_low <= a_low)


def _merge_pair(prev_high: float, prev_low: float,
                cur_high: float, cur_low: float,
                direction: Direction) -> tuple[float, float]:
    """
    第65课：
      向上时，取 [max(low), max(high)] —— 高高低高（两根中较高的高，较高的低）；
      向下时，取 [min(low), min(high)] —— 高低低低（两根中较低的高，较低的低）。
    返回合并后的 (high, low)。
    """
    if direction == Direction.UP:
        return max(prev_high, cur_high), max(prev_low, cur_low)
    else:
        return min(prev_high, cur_high), min(prev_low, cur_low)


def merge_klines(raws: Iterable[RawBar]) -> list[MergedBar]:
    """
    把原始K线序列做包含关系合并。

    算法（第65课）：
      - 顺序滚动：用 (merged[-1], raw[i]) 判断；若有包含则按方向合并成新的
        merged[-1]，否则把 raw[i] 作为新的 merged 元素接上。
      - 方向：用"合并基"与"前一个合并基"的高低关系决定方向。第一个合并基没有
        前向方向，规则上初始方向不重要（之后立刻会被两根独立K线刷新）；
        这里给一个安全默认：第一根之后若与第二根不包含，则用两者高低关系定向。

    注意：方向必须在"判定是否合并"时已经确定，所以方向更新发生在每次"新元素
    入栈"之后，而不是合并时。
    """
    raws = list(raws)
    if not raws:
        return []

    merged: list[MergedBar] = []
    direction = Direction.UP  # 兜底初值，会在第二个元素就被覆盖

    for raw in raws:
        if not merged:
            merged.append(MergedBar(
                idx=0, dt=raw.dt,
                high=raw.high, low=raw.low,
                direction=direction,
                raw_indices=[raw.idx],
            ))
            continue

        last = merged[-1]

        if _is_contained(last.high, last.low, raw.high, raw.low):
            # 包含 → 用当前方向合并到 last
            new_high, new_low = _merge_pair(
                last.high, last.low, raw.high, raw.low, direction,
            )
            last.high = new_high
            last.low = new_low
            last.dt = raw.dt
            last.raw_indices.append(raw.idx)
            # 方向不变（方向只在"独立K线接进来"时刷新）
        else:
            # 不包含 → 先用 last 与 raw 的关系刷新方向，再入栈
            if raw.high > last.high and raw.low > last.low:
                direction = Direction.UP
            elif raw.high < last.high and raw.low < last.low:
                direction = Direction.DOWN
            # else: 理论上不会进这里（不包含 + 不高高低低 + 不低低高高 不存在）
            merged.append(MergedBar(
                idx=len(merged), dt=raw.dt,
                high=raw.high, low=raw.low,
                direction=direction,
                raw_indices=[raw.idx],
            ))

    return merged


# ══════════════════════════════════════════════════════════════════════
# 2. 顶/底分型识别 — 第62/77课
# ══════════════════════════════════════════════════════════════════════

def find_fractals(merged: list[MergedBar],
                  include_unconfirmed: bool = False) -> list[Fractal]:
    """
    扫描合并K线，识别顶/底分型。

    第62课定义：
      顶分型：第二K线高点是相邻三K线高点中最高的，且低点也是相邻三K线低点中最高的。
      底分型：第二K线低点是相邻三K线低点中最低的，且高点也是相邻三K线高点中最低的。

    严格起见：高/低的"最高""最低"用 ">" / "<"（严格不等）。等高等低不算分型——
    这样可以避免合并K线序列中出现的退化情形。

    include_unconfirmed=True 时，若序列最后只有 2 根合并K线已收，但前一根具备
    单边极值条件（左侧严格不等成立，右侧未知），输出 confirmed=False 的潜在分型。
    """
    fractals: list[Fractal] = []
    n = len(merged)

    for i in range(1, n - 1):
        left, mid, right = merged[i - 1], merged[i], merged[i + 1]

        # 顶分型：mid.high 严格高于两侧 high，且 mid.low 严格高于两侧 low
        if mid.high > left.high and mid.high > right.high \
                and mid.low > left.low and mid.low > right.low:
            fractals.append(Fractal(
                ftype=FractalType.TOP,
                mid_idx=i, left_idx=i - 1, right_idx=i + 1,
                high=mid.high, low=mid.low, dt=mid.dt,
                confirmed=True,
            ))
            continue

        # 底分型：mid.low 严格低于两侧 low，且 mid.high 严格低于两侧 high
        if mid.low < left.low and mid.low < right.low \
                and mid.high < left.high and mid.high < right.high:
            fractals.append(Fractal(
                ftype=FractalType.BOTTOM,
                mid_idx=i, left_idx=i - 1, right_idx=i + 1,
                high=mid.high, low=mid.low, dt=mid.dt,
                confirmed=True,
            ))

    # 潜在分型（最后一根尚未走完时也给个推断）
    if include_unconfirmed and n >= 2:
        i = n - 1
        left, mid = merged[i - 1], merged[i]
        # 潜在顶：mid 比 left 更高、更高
        if mid.high > left.high and mid.low > left.low:
            fractals.append(Fractal(
                ftype=FractalType.TOP,
                mid_idx=i, left_idx=i - 1, right_idx=i,  # right 未知，占位
                high=mid.high, low=mid.low, dt=mid.dt,
                confirmed=False,
            ))
        elif mid.low < left.low and mid.high < left.high:
            fractals.append(Fractal(
                ftype=FractalType.BOTTOM,
                mid_idx=i, left_idx=i - 1, right_idx=i,
                high=mid.high, low=mid.low, dt=mid.dt,
                confirmed=False,
            ))

    return fractals


# ══════════════════════════════════════════════════════════════════════
# 3. 笔的划分 — 第77课 3 步法 + 新笔 / 老笔配置
# ══════════════════════════════════════════════════════════════════════

def _bars_between(merged: list[MergedBar],
                  fx_a: Fractal, fx_b: Fractal) -> int:
    """两个分型中间的合并K线数（不含两端的 mid_idx 那根）。"""
    return abs(fx_b.mid_idx - fx_a.mid_idx) - 1


def _fractal_pair_valid(merged: list[MergedBar],
                        fx_a: Fractal, fx_b: Fractal,
                        new_stroke: bool) -> bool:
    """
    判断两个相邻异性质分型能否构成一笔。

    硬性条件（第77课）：
      a) 顶K高点的区间必须至少有一部分高于底K低点的区间，否则不成笔。
         —— 这里转化为：顶分型的 mid.low > 底分型的 mid.high 严格成立？
         不对。原文意思更宽：顶K高点 ≥ 底K高点（即顶K区间不能完全在底K区间内）。
         实现上等价于：顶分型 extreme > 底分型 extreme，且顶K mid.high > 底K mid.low
         （只要顶K区间和底K区间不是"顶在底之下"）。
      b) 顶分型 mid_idx 与底分型 mid_idx 不能相邻（否则两个分型必共用K线）。

    新笔（new_stroke=True，第81课）：
      c) 顶K和底K不能共享原始K线（mid_idx 不同就基本保证；进一步保险检查
         raw_indices 无交集）。
      d) 两端 mid 之间的合并K线数 ≥ 3（即整段≥5根合并K线）。

    老笔（new_stroke=False，第62课）：
      d') 两端 mid 之间的合并K线数 ≥ 1（即整段≥3根合并K线）。
    """
    # 确认是异性质
    if fx_a.ftype == fx_b.ftype:
        return False

    if fx_a.ftype == FractalType.TOP:
        top, bot = fx_a, fx_b
    else:
        top, bot = fx_b, fx_a

    # (a) 顶 extreme > 底 extreme（顶必须真的高于底）
    if top.extreme <= bot.extreme:
        return False
    # 顶K的高点区间必须至少触及底K低点之上 → 顶K mid.high > 底K mid.low
    if top.high <= bot.low:
        return False

    # (b) mid_idx 不能相邻
    if abs(top.mid_idx - bot.mid_idx) < 2:
        return False

    # (c) 顶/底 mid 不能共用合并K线（mid_idx 不等已确保）
    top_mid_raw = set(merged[top.mid_idx].raw_indices)
    bot_mid_raw = set(merged[bot.mid_idx].raw_indices)
    if top_mid_raw & bot_mid_raw:
        return False

    # (d) 中间合并K线数
    gap = _bars_between(merged, fx_a, fx_b)
    min_gap = 3 if new_stroke else 1
    if gap < min_gap:
        return False

    return True


def find_strokes(merged: list[MergedBar],
                 fractals: list[Fractal],
                 new_stroke: bool = True) -> list[Stroke]:
    """
    第77课3步法：
      1. 把所有合格分型按时间排列（已经按 mid_idx 顺序）；
      2. 同性质相邻：保留更极值者
         —— 连续顶：保留较高的；连续底：保留较低的；
      3. 剩下分型若相邻是顶↔底则构成一笔；若仍同性质，再合并极值。

    new_stroke=True（默认）：新笔，中间≥3根合并K线；
    new_stroke=False：老笔，中间≥1根合并K线。

    本实现采取"先化简到严格交替序列，再扫描成笔"的策略：
      - 严格交替之后，相邻分型必为异性质；
      - 再对每对相邻分型应用 _fractal_pair_valid；
      - 不通过的对：丢弃后一个分型（保留前一个），继续匹配下一个。
        这是把"分型不足以成笔"等价为"该分型不算确认笔端"的处理。

    返回的 Stroke 列表，相邻两笔方向必然相反。
    """
    if not fractals:
        return []

    # ── Step 1+2: 同性质相邻合并到极值 ──────────────────────────────
    simplified: list[Fractal] = []
    for fx in fractals:
        if not simplified:
            simplified.append(fx)
            continue
        last = simplified[-1]
        if fx.ftype == last.ftype:
            # 同性质 → 比较极值，保留更极的
            if fx.ftype == FractalType.TOP:
                if fx.extreme > last.extreme:
                    simplified[-1] = fx
            else:  # BOTTOM
                if fx.extreme < last.extreme:
                    simplified[-1] = fx
        else:
            simplified.append(fx)

    # ── Step 3: 异性质对扫描成笔 ────────────────────────────────────
    strokes: list[Stroke] = []
    anchor: Fractal | None = None        # 当前已认定的"上一个笔端分型"

    for fx in simplified:
        if anchor is None:
            anchor = fx
            continue

        if _fractal_pair_valid(merged, anchor, fx, new_stroke):
            # 成笔
            direction = (Direction.UP if anchor.ftype == FractalType.BOTTOM
                         else Direction.DOWN)
            strokes.append(Stroke(
                idx=len(strokes),
                direction=direction,
                start_fx=anchor,
                end_fx=fx,
            ))
            anchor = fx
            continue

        # ── 不成笔：按"唯一性"原则做撤回 ─────────────────────────
        # 情况 A：fx 与 anchor 同性质 → 第77课"同性相邻保留更极者"
        if fx.ftype == anchor.ftype:
            more_extreme = (
                (fx.ftype == FractalType.TOP and fx.extreme > anchor.extreme)
                or (fx.ftype == FractalType.BOTTOM and fx.extreme < anchor.extreme)
            )
            if more_extreme:
                # 若 anchor 是上一笔的终点 → 上一笔的终点要替换成 fx，
                # 即"上一笔的起点 → fx" 才是真正的那一笔。
                # 实现方式：撤回上一笔，用 popped.start_fx 与 fx 配对重建。
                if strokes and strokes[-1].end_fx is anchor:
                    popped = strokes.pop()
                    new_start = popped.start_fx
                    if _fractal_pair_valid(merged, new_start, fx, new_stroke):
                        direction = (Direction.UP if new_start.ftype == FractalType.BOTTOM
                                     else Direction.DOWN)
                        strokes.append(Stroke(
                            idx=len(strokes),
                            direction=direction,
                            start_fx=new_start,
                            end_fx=fx,
                        ))
                        anchor = fx
                    else:
                        # 重建配对失败（极少见，例如 mid_idx 相邻）→ 回退
                        # 让 popped.start_fx 充当新 anchor，等下一个异性 fx
                        anchor = new_start
                else:
                    # 无上一笔可撤 → 仅替换 anchor（原先逻辑）
                    anchor = fx
            # else: 丢弃 fx
            continue

        # 情况 B：fx 与 anchor 异性质但 _fractal_pair_valid 不通过（K线不够等）
        #   → 直接丢弃 fx（第77课"中间那些都 X 掉"的精神）。

    return strokes


# ══════════════════════════════════════════════════════════════════════
# 4. 线段划分 — 第67/71/77/78课
# ══════════════════════════════════════════════════════════════════════
#
# 算法概述：
#   ① 找起点：连续三笔有价格重叠 + 第一笔与第三笔同向 → 该方向即线段方向。
#   ② 特征序列：与线段方向相反的笔，按顺序构成序列；元素用 (high, low) 描述。
#   ③ 破坏：在特征序列上识别"反向分型"——
#       第一种破坏（无缺口）：第1、2 元素之间没有缺口（区间有重叠）。
#                            两侧不做包含合并；分型在第2元素出现即结束线段。
#       第二种破坏（有缺口）：第1、2 元素之间有缺口。
#                            从假定转折点开始，对反向特征序列做包含合并；
#                            反向特征序列出现分型即结束线段（不再细分缺口）。
#   ④ 第三笔完全在第一笔范围内（第71课）：先不确认破坏，等突破第一笔的端点。
#   ⑤ 古怪线段（第78课）：第一种破坏后，反向没有形成新线段时，重新合并回原方向。
#
# 实现采取"对每个候选转折点都先记录，等右侧足够 K 线后再确认"的方式。
# 简化点：第二种破坏的"反向特征序列包含合并"采用同一套 _merge_feat_elems。
# ══════════════════════════════════════════════════════════════════════


@dataclass
class _FeatElem:
    """特征序列元素（一条或多条同向笔合并后的等价笔）。"""
    high: float
    low: float
    stroke_start_idx: int        # 在 strokes 列表中的起始下标
    stroke_end_idx: int          # 在 strokes 列表中的结束下标（含）
    direction: Direction         # 该笔/合并组的方向（与所属线段方向相反）


def _stroke_overlap(s_a: Stroke, s_b: Stroke) -> bool:
    """两条笔的价格区间 [low, high] 是否有重叠（含端点相等）。"""
    return s_a.low <= s_b.high and s_b.low <= s_a.high


def _elem_from_stroke(s: Stroke) -> _FeatElem:
    return _FeatElem(
        high=s.high, low=s.low,
        stroke_start_idx=s.idx, stroke_end_idx=s.idx,
        direction=s.direction,
    )


def _merge_feat_elems(prev: _FeatElem, cur: _FeatElem,
                      seg_direction: Direction) -> _FeatElem | None:
    """
    特征序列内部的包含合并（第67/71课）。
    seg_direction 是所属线段方向；特征序列元素方向与之相反。
      上行段（seg_direction=UP）的特征序列由向下笔组成 → 取较低的高、较低的低？
      实际原文：特征序列元素也按"笔的方向"做包含——上行段特征序列由下行笔组成，
      特征序列包含合并采用"向下"取法：[min(high), min(low)]。
      下行段反之：[max(high), max(low)]。
    若 prev 和 cur 不构成包含关系（即彼此独立），返回 None 表示不合并。
    """
    contained = (
        (prev.high >= cur.high and prev.low <= cur.low)
        or (cur.high >= prev.high and cur.low <= prev.low)
    )
    if not contained:
        return None

    if seg_direction == Direction.UP:
        new_high = min(prev.high, cur.high)
        new_low = min(prev.low, cur.low)
    else:
        new_high = max(prev.high, cur.high)
        new_low = max(prev.low, cur.low)

    return _FeatElem(
        high=new_high, low=new_low,
        stroke_start_idx=prev.stroke_start_idx,
        stroke_end_idx=cur.stroke_end_idx,
        direction=prev.direction,
    )


def _has_gap(prev: _FeatElem, cur: _FeatElem,
             seg_direction: Direction) -> bool:
    """
    特征序列相邻两元素之间是否存在缺口（即区间完全不重叠）。
    seg_direction = UP 时，特征序列方向 = DOWN：
        缺口 = prev.low > cur.high（前元素的低区仍高于后元素的高区）。
    seg_direction = DOWN 时（特征序列方向 = UP）：
        缺口 = prev.high < cur.low。
    """
    if seg_direction == Direction.UP:
        return prev.low > cur.high
    return prev.high < cur.low


def _is_feat_fractal(e1: _FeatElem, e2: _FeatElem, e3: _FeatElem,
                     seg_direction: Direction) -> bool:
    """
    特征序列三元素是否构成"反向分型"（中间元素是端点）。
    上行段（seg_direction=UP）：寻找"顶分型"——e2.high 是相邻三者中最高，
                                且 e2.low 也是相邻三者中最高（严格不等）。
    下行段：寻找"底分型"——e2.low 是最低，且 e2.high 也是最低。
    严格不等避免退化情形。
    """
    if seg_direction == Direction.UP:
        return (e2.high > e1.high and e2.high > e3.high
                and e2.low > e1.low and e2.low > e3.low)
    return (e2.low < e1.low and e2.low < e3.low
            and e2.high < e1.high and e2.high < e3.high)


def find_segments(strokes: list[Stroke]) -> list[Segment]:
    """
    把笔序列划成线段。
    返回所有完成的线段（最后一段如果未确认破坏，也作为"未完成"返回，
    用 break_type=0 标识；其余 1=第一种破坏，2=第二种破坏）。

    实现是状态机式滚动：维护当前线段的方向、构成笔，以及反向特征序列。
    """
    if len(strokes) < 3:
        return []

    segments: list[Segment] = []

    # ── 找起点：连续三笔有重叠且第1、3笔同向 ───────────────────────
    start_i = -1
    for i in range(len(strokes) - 2):
        s1, s2, s3 = strokes[i], strokes[i + 1], strokes[i + 2]
        if s1.direction == s3.direction and _stroke_overlap(s1, s3):
            start_i = i
            break
    if start_i < 0:
        return []

    # 当前线段状态
    seg_dir = strokes[start_i].direction
    seg_strokes: list[Stroke] = list(strokes[start_i: start_i + 3])

    # 特征序列：由"反向笔"构成；初始只有起始三笔中的第二笔
    feat_seq: list[_FeatElem] = [_elem_from_stroke(strokes[start_i + 1])]

    # 第二种破坏的"反向特征序列"暂存：
    #   pending_break_elem 是假定线段终止点（特征序列分型的极值元素）；
    #   反向特征序列为 reverse_feat_seq；当它出现分型，第二种破坏确认。
    pending_break_elem: _FeatElem | None = None
    reverse_feat_seq: list[_FeatElem] = []
    # 第二种破坏判断时是否需要包含合并（第78课要求）
    # 第一种破坏：特征序列内不合并；第二种破坏：合并。
    # 我们对两套序列分别处理。

    def finish_segment(break_type: int, end_stroke_pos: int):
        """
        把当前累积的 seg_strokes 截到 end_stroke_pos（含）为止，提交为一段。
        然后用余下的笔重置线段方向。
        """
        nonlocal seg_strokes, seg_dir, feat_seq
        nonlocal pending_break_elem, reverse_feat_seq

        # end_stroke_pos 是 strokes 全局下标
        confirmed = [s for s in seg_strokes if s.idx <= end_stroke_pos]
        # 必须 ≥3 笔且奇数（第77课）；若不足或偶数，尝试回退一笔
        while len(confirmed) >= 3 and len(confirmed) % 2 == 0:
            confirmed = confirmed[:-1]
        if len(confirmed) >= 3:
            segments.append(Segment(
                idx=len(segments),
                direction=seg_dir,
                strokes=confirmed,
                break_type=break_type,
            ))

    i = start_i + 3
    while i < len(strokes):
        nxt = strokes[i]

        if pending_break_elem is None:
            # ── 主流程：在原方向特征序列里找破坏 ───────────────
            if nxt.direction == seg_dir:
                # 与线段方向同向 → 加入线段，更新端点
                seg_strokes.append(nxt)
                i += 1
                continue

            # 反向笔 → 进入特征序列
            new_elem = _elem_from_stroke(nxt)

            # 试包含合并（仅当与上一元素构成包含 + 当前不是潜在破坏点时）
            # 简化：在没有候选破坏的情况下，feat_seq 不做合并——这样
            # "缺口"信息得到保留；一旦出现分型再回头看缺口。
            feat_seq.append(new_elem)
            seg_strokes.append(nxt)

            # 检查分型
            if len(feat_seq) >= 3:
                e1, e2, e3 = feat_seq[-3], feat_seq[-2], feat_seq[-1]
                if _is_feat_fractal(e1, e2, e3, seg_dir):
                    has_gap = _has_gap(e1, e2, seg_dir)
                    if not has_gap:
                        # 第一种破坏：立即结束线段
                        # 终点 = e2 对应的反向笔之前的最后一根同向笔的终点
                        # e2 是反向笔，它的"起点"即原线段的实际高/低点
                        end_stroke_idx = e2.stroke_start_idx - 1
                        if end_stroke_idx < start_i:
                            end_stroke_idx = e2.stroke_start_idx
                        finish_segment(1, end_stroke_idx)

                        # 从 e2 起点之后重启新线段（方向反转）
                        new_start = e2.stroke_start_idx
                        # 重新初始化状态
                        if new_start + 2 < len(strokes):
                            seg_dir = strokes[new_start].direction
                            seg_strokes = list(strokes[new_start: new_start + 3])
                            feat_seq = [_elem_from_stroke(strokes[new_start + 1])]
                            start_i = new_start
                            i = new_start + 3
                            pending_break_elem = None
                            reverse_feat_seq = []
                            continue
                        else:
                            # 后续笔不足以构成新段
                            return segments
                    else:
                        # 第二种破坏候选：进入"等反向特征序列分型"阶段
                        pending_break_elem = e2
                        reverse_feat_seq = []
            i += 1
            continue

        # ── 第二种破坏候选：等反向特征序列分型 ─────────────────
        # 反向特征序列方向 = 原线段方向（因为反向后线段方向反过来，特征序列再反过来 = 原方向）
        # 简化：把 nxt 加入 reverse_feat_seq，做包含合并，找分型
        new_elem = _elem_from_stroke(nxt)
        seg_strokes.append(nxt)

        # 反向特征序列内部包含合并（第78课要求）
        reverse_seg_dir = Direction.DOWN if seg_dir == Direction.UP else Direction.UP
        if reverse_feat_seq and nxt.direction != reverse_seg_dir:
            # 与反向特征序列同向（即与原线段同向）的笔，不进入反向特征序列
            i += 1
            continue
        if nxt.direction == reverse_seg_dir:
            # 反向方向的笔（即原方向同向笔，作反向线段的特征元素）
            if reverse_feat_seq:
                merged = _merge_feat_elems(reverse_feat_seq[-1], new_elem, reverse_seg_dir)
                if merged is not None:
                    reverse_feat_seq[-1] = merged
                else:
                    reverse_feat_seq.append(new_elem)
            else:
                reverse_feat_seq.append(new_elem)

            if len(reverse_feat_seq) >= 3:
                e1, e2, e3 = reverse_feat_seq[-3], reverse_feat_seq[-2], reverse_feat_seq[-1]
                if _is_feat_fractal(e1, e2, e3, reverse_seg_dir):
                    # 第二种破坏确认
                    end_stroke_idx = pending_break_elem.stroke_start_idx - 1
                    finish_segment(2, end_stroke_idx)

                    new_start = pending_break_elem.stroke_start_idx
                    if new_start + 2 < len(strokes):
                        seg_dir = strokes[new_start].direction
                        seg_strokes = list(strokes[new_start: new_start + 3])
                        feat_seq = [_elem_from_stroke(strokes[new_start + 1])]
                        start_i = new_start
                        i = new_start + 3
                        pending_break_elem = None
                        reverse_feat_seq = []
                        continue
                    else:
                        return segments

        # 同时检测原方向延续：若反向探索失败，原线段创新高/新低 → 取消候选
        if seg_dir == Direction.UP and nxt.direction == Direction.UP \
                and nxt.high > pending_break_elem.high:
            pending_break_elem = None
            reverse_feat_seq = []
        elif seg_dir == Direction.DOWN and nxt.direction == Direction.DOWN \
                and nxt.low < pending_break_elem.low:
            pending_break_elem = None
            reverse_feat_seq = []

        i += 1

    # 收尾：未完成的线段也作为 break_type=0 提交
    if len(seg_strokes) >= 3:
        confirmed = list(seg_strokes)
        while len(confirmed) >= 3 and len(confirmed) % 2 == 0:
            confirmed = confirmed[:-1]
        if len(confirmed) >= 3:
            segments.append(Segment(
                idx=len(segments),
                direction=seg_dir,
                strokes=confirmed,
                break_type=0,
            ))

    return segments


# ══════════════════════════════════════════════════════════════════════
# 5. 中枢识别 — 第17/20/29课
# ══════════════════════════════════════════════════════════════════════
#
# 算法（最低不可分级别下，"次级别走势"用线段近似）：
#   ① 顺序扫描线段。对每个候选起点 i，考察 segments[i]、segments[i+1]、
#      segments[i+2] 三段。由于线段在时间上严格交替方向，segments[i] 与
#      segments[i+2] 必然同向，套用第20课公式：
#          ZG = min(segments[i].high, segments[i+2].high)
#          ZD = max(segments[i].low,  segments[i+2].low)
#      若 ZG > ZD（前两同向段有重叠），中枢成立，前三段全部纳入。
#   ② 向后扩展：对 j = i+3, i+4, ...，若 segments[j] 与 [ZD, ZG] 仍有
#      交集（即 segments[j].low <= ZG 且 segments[j].high >= ZD），
#      纳入并更新 GG=max(gn)、DD=min(dn)；否则该段就是 leaving_segment，
#      中枢在 j-1 处结束。
#   ③ 进入方向（entry_direction）取 segments[i-1].direction；i=0 时为 None。
#   ④ 下一中枢的扫描从 leaving 段开始（i=j），允许"离开段"成为下一中枢的
#      第一段（即趋势中两个同级别中枢之间共享一段过渡）。若无离开段（中枢
#      延伸到序列末尾），算法结束。
#
# 第20课关键边界：
#   - "ZD/ZG 只用前两段同向走势的高低点，定下后不再因后续 Z 段而改变"
#   - "GG/DD 遍历所有 Z 段"——用于判断扩展、级别升级、第3类买卖点等
#   - "若有 Zn，使得 dn>ZG 或 gn<ZD，则必然产生高级别的走势中枢或趋势"
#     —— 本实现把这种 Zn 标为 leaving_segment，不计入当前中枢
# ══════════════════════════════════════════════════════════════════════


def find_pivots(segments: list[Segment]) -> list[Pivot]:
    """
    把线段序列划分成若干中枢。

    返回 Pivot 列表，按时间顺序排列。最后一个中枢可能 is_finished=False
    （仍在延伸/未出现离开段）。
    """
    if len(segments) < 3:
        return []

    pivots: list[Pivot] = []
    i = 0
    n = len(segments)

    while i + 2 < n:
        s1, s3 = segments[i], segments[i + 2]
        # 第20课公式：仅用前两段同向走势确定中枢边界
        zg = min(s1.high, s3.high)
        zd = max(s1.low, s3.low)

        if zg <= zd:
            # 前两同向段无重叠 → 此处不成中枢，下一段开始扫描
            i += 1
            continue

        members: list[Segment] = list(segments[i:i + 3])
        gg = max(s.high for s in members)
        dd = min(s.low for s in members)

        # 向后扩展：纳入与 [ZD, ZG] 有交集的段
        leaving: Segment | None = None
        j = i + 3
        while j < n:
            sj = segments[j]
            # 完全脱离中枢区间 → 离开段
            if sj.low > zg or sj.high < zd:
                leaving = sj
                break
            members.append(sj)
            if sj.high > gg:
                gg = sj.high
            if sj.low < dd:
                dd = sj.low
            j += 1

        entry_dir = segments[i - 1].direction if i > 0 else None
        pivots.append(Pivot(
            idx=len(pivots),
            segments=members,
            zg=zg, zd=zd, gg=gg, dd=dd,
            entry_direction=entry_dir,
            leaving_segment=leaving,
        ))

        if leaving is None:
            # 中枢延伸到序列末尾，没有更多段可处理
            break
        # 离开段允许作为下一中枢的"第一段"参与扫描
        i = j

    return pivots


# ══════════════════════════════════════════════════════════════════════
# 6. MACD 与背驰检测 — 第24/25/27/50/65课
# ══════════════════════════════════════════════════════════════════════
#
# 实现层次：
#   ① calc_macd            标准 12/26/9 EMA → DIFF/DEA/柱(=2*(DIFF-DEA))
#   ② segment_macd_area    把段映射回原始K线区间，求方向匹配的柱面积绝对值
#   ③ detect_divergence    同向段两两对比：C 段创新极但面积 < A 段 → 背驰
#
# 关键原文：
#   - 第25课：MACD 参数 12/26/9，黄白线=DIFF/DEA，柱=2*(DIFF-DEA)
#   - 第24课："C 段的走势类型完成时对应的 MACD 柱子面积比 A 段对应的面积要小，
#             这时候就构成标准的背弛"
#   - 第27课："第一类买点都是在 0 轴之下背驰形成的"——本实现把"0 轴位置"
#             作为附加判据：UP 背驰要求 A 段最大 DIFF 在 0 轴上方；
#             DOWN 背驰要求 A 段最小 DIFF 在 0 轴下方。
#   - 第50课：MACD 是辅助——本模块只在已有段结构后调用，不直接驱动判断。
#   - 第65课：线段以下的背驰称"类背驰"，力度比较方法相同。
#
# 注意：
#   - "创新极"是背驰必要条件：UP 段要 high > 前 UP 段 high；DOWN 段反之。
#   - 背驰类型粒度本版只输出"段级类背驰"，趋势/盘整背驰需结合中枢上下文，
#     由 Phase 1.7 的买卖点逻辑判断时再叠加。
# ══════════════════════════════════════════════════════════════════════


@dataclass
class MACDPoint:
    """单根K线的 MACD 三元组。bar = 2*(diff - dea)。"""
    diff: float
    dea: float
    bar: float


@dataclass
class Divergence:
    """
    一次背驰记录（段级别力度比较）。
      seg_a_idx / seg_c_idx  对比的两段在 segments 列表中的下标
      direction              方向（UP=顶背驰，DOWN=底背驰）
      area_a / area_c        两段 MACD 柱面积（绝对值之和）
      price_a / price_c      两段的极值（UP=high，DOWN=low）
      diff_a / diff_c        两段对应区间的 DIFF 极值（UP取max，DOWN取min）
                             用于"0 轴判定"：第27课"第一类买点在 0 轴之下"。
    """
    seg_a_idx: int
    seg_c_idx: int
    direction: Direction
    area_a: float
    area_c: float
    price_a: float
    price_c: float
    diff_a: float
    diff_c: float

    @property
    def ratio(self) -> float:
        """C 段面积 / A 段面积。越小背驰越强。"""
        return self.area_c / self.area_a if self.area_a > 0 else 0.0


def _ema(values: list[float], period: int) -> list[float]:
    """
    指数移动平均（EMA）。
    标准递推：EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}，其中 alpha = 2/(period+1)。
    种子值取 values[0]（行业惯用，等同于通达信"先用首值再迭代"）。
    """
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * alpha + out[-1] * (1.0 - alpha))
    return out


def calc_macd(closes: list[float],
              fast: int = 12, slow: int = 26, signal: int = 9
              ) -> list[MACDPoint]:
    """
    标准 MACD（第25课参数 12/26/9）。
      DIFF = EMA(close, fast) - EMA(close, slow)
      DEA  = EMA(DIFF, signal)
      bar  = 2 * (DIFF - DEA)
    """
    if not closes:
        return []
    ef = _ema(closes, fast)
    es = _ema(closes, slow)
    diff = [f - s for f, s in zip(ef, es)]
    dea = _ema(diff, signal)
    return [MACDPoint(diff=d, dea=e, bar=2.0 * (d - e))
            for d, e in zip(diff, dea)]


def _segment_raw_range(merged: list[MergedBar], segment: Segment) -> tuple[int, int]:
    """
    把段映射回原始K线下标区间 [start_raw, end_raw]（闭区间）。
    用段首笔起点分型所在合并K线的首根原始下标，到段末笔终点分型所在合并K线
    的末根原始下标。
    """
    start_mid = segment.strokes[0].start_fx.mid_idx
    end_mid = segment.strokes[-1].end_fx.mid_idx
    start_raw = merged[start_mid].raw_indices[0]
    end_raw = merged[end_mid].raw_indices[-1]
    return start_raw, end_raw


def segment_macd_area(macd: list[MACDPoint],
                      merged: list[MergedBar],
                      segment: Segment) -> float:
    """
    求一段对应原始K线区间内、与段方向匹配的 MACD 柱的绝对值之和。
      UP 段：累加所有 bar > 0 的柱；
      DOWN 段：累加所有 bar < 0 的柱（取绝对值）。
    第24课原文用"红柱面积"、"绿柱面积"，对应这两种情形。
    """
    start_raw, end_raw = _segment_raw_range(merged, segment)
    area = 0.0
    for i in range(start_raw, min(end_raw + 1, len(macd))):
        b = macd[i].bar
        if segment.direction == Direction.UP and b > 0:
            area += b
        elif segment.direction == Direction.DOWN and b < 0:
            area += -b
    return area


def _segment_diff_extreme(macd: list[MACDPoint],
                          merged: list[MergedBar],
                          segment: Segment) -> float:
    """段对应原始K线区间内的 DIFF 极值：UP 取最大，DOWN 取最小。"""
    start_raw, end_raw = _segment_raw_range(merged, segment)
    span = macd[start_raw: min(end_raw + 1, len(macd))]
    if not span:
        return 0.0
    if segment.direction == Direction.UP:
        return max(p.diff for p in span)
    return min(p.diff for p in span)


def detect_divergence(segments: list[Segment],
                      merged: list[MergedBar],
                      raws: list[RawBar],
                      require_zero_axis: bool = True,
                      ) -> list[Divergence]:
    """
    段级别"类背驰"检测。

    算法：
      1. 用收盘价算 MACD。
      2. 对每个段 C（idx ≥ 2），向前找最近的"同向段" A。
         由于线段方向严格交替，A = segments[c_idx - 2]、segments[c_idx - 4] …
         本实现取最近的同向段（即 c_idx - 2）作为基准——满足"相邻两个同向
         走势段"的最简对比；多中枢趋势对比留给 Phase 1.7 整合后再扩展。
      3. C 必须创新极：UP → C.high > A.high；DOWN → C.low < A.low。
         不创新极不算背驰（第24课）。
      4. 面积比较：area_c < area_a → 背驰候选。
      5. 0 轴判定（require_zero_axis=True，第27课）：
           UP 背驰要求 A 段 DIFF 最大值 > 0（即"0 轴之上的顶背驰"）；
           DOWN 背驰要求 A 段 DIFF 最小值 < 0（"0 轴之下的底背驰"）。
         传入 False 可放宽（线段以下"类背驰"常无此约束）。

    返回所有满足条件的 Divergence 记录。
    """
    if len(segments) < 3 or not raws:
        return []

    closes = [r.close for r in raws]
    macd = calc_macd(closes)
    if not macd:
        return []

    # 预先缓存每段的面积与 DIFF 极值
    n_seg = len(segments)
    areas = [segment_macd_area(macd, merged, s) for s in segments]
    diffs = [_segment_diff_extreme(macd, merged, s) for s in segments]

    results: list[Divergence] = []
    for c_idx in range(2, n_seg):
        a_idx = c_idx - 2  # 最近的同向段
        if segments[a_idx].direction != segments[c_idx].direction:
            # 防御：理论上线段交替方向，差 2 必同向；防止异常数据
            continue
        seg_a, seg_c = segments[a_idx], segments[c_idx]

        # 创新极
        if seg_c.direction == Direction.UP:
            if seg_c.high <= seg_a.high:
                continue
        else:
            if seg_c.low >= seg_a.low:
                continue

        # 面积比较
        if areas[c_idx] >= areas[a_idx]:
            continue

        # 0 轴判定（可选）
        if require_zero_axis:
            if seg_c.direction == Direction.UP and diffs[a_idx] <= 0:
                continue
            if seg_c.direction == Direction.DOWN and diffs[a_idx] >= 0:
                continue

        results.append(Divergence(
            seg_a_idx=a_idx, seg_c_idx=c_idx,
            direction=seg_c.direction,
            area_a=areas[a_idx], area_c=areas[c_idx],
            price_a=seg_a.high if seg_c.direction == Direction.UP else seg_a.low,
            price_c=seg_c.high if seg_c.direction == Direction.UP else seg_c.low,
            diff_a=diffs[a_idx], diff_c=diffs[c_idx],
        ))

    return results


# ══════════════════════════════════════════════════════════════════════
# 7. 1/2/3 类买卖点 — 第17/20/24/27/53课
# ══════════════════════════════════════════════════════════════════════
#
# 识别策略（在当前级别上，"次级别"用相邻段近似）：
#
#   B1 / S1（第1类）：来自背驰 C 段的终点
#     - DOWN 方向背驰 → B1（下跌趋势/盘整背驰的终点 = 第一类买点）
#     - UP   方向背驰 → S1
#     - 严格意义需要"趋势背驰"（≥2 个同向中枢），本实现先输出所有段级
#       背驰对应的 1 类信号，并附带 divergence 引用供后续过滤
#
#   B2 / S2（第2类）：1 类后的次级别确认（第17课"买卖点定律一"）
#     - B2：B1 之后，下一段（UP）完成、再下一段（DOWN）回试且不破 B1 价位
#     - S2 镜像
#     - 信号点 = 那条 DOWN 段（或 UP 段）的终点
#
#   B3 / S3（第3类）：中枢离开后的回试不破 ZG/ZD（第20/53课）
#     - B3：已完成中枢的 leaving_segment 方向 = UP，且 leaving 之后的下
#           一段（DOWN 反向回试）low > ZG → 第一类买点
#     - S3 镜像，要求 high < ZD
#     - "必须是第一次"：算法只看 leaving 后紧邻的那一段，自然成立
#
# 关于"是否前向中枢"等更高阶约束（第29/53课的趋势/盘整背驰区分等）暂
# 不在此实现内细分，避免过度判定。可在调用层结合 pivots 数量/方向再做
# 二次筛选。
# ══════════════════════════════════════════════════════════════════════


@dataclass
class TradeSignal:
    """
    一个买卖点信号。

      signal_type  "B1"/"B2"/"B3"/"S1"/"S2"/"S3"
      dt           信号点对应的日期
      price        信号点价格（段末分型的 extreme）
      segment_idx  信号点所在段在 segments 列表中的下标
      pivot_idx    关联中枢下标（仅 B3/S3 有）；其他类型为 None
      divergence   关联背驰记录（仅 B1/S1 有）；其他为 None
      note         调试备注，可选
    """
    idx: int
    signal_type: str
    dt: str
    price: float
    segment_idx: int
    pivot_idx: int | None = None
    divergence: Divergence | None = None
    note: str = ""

    @property
    def is_buy(self) -> bool:
        return self.signal_type.startswith("B")

    @property
    def is_sell(self) -> bool:
        return self.signal_type.startswith("S")

    @property
    def level(self) -> int:
        """1, 2, 或 3。"""
        return int(self.signal_type[1])


def find_signals(segments: list[Segment],
                 pivots: list[Pivot],
                 divergences: list[Divergence]) -> list[TradeSignal]:
    """
    依据已识别的线段、中枢、背驰，统一输出 1/2/3 类买卖点。

    返回的 TradeSignal 列表按 segment_idx 升序排列。idx 字段在最终排序后
    重新赋值，保证可作为稳定外部引用。
    """
    signals: list[TradeSignal] = []
    n_seg = len(segments)

    # ── 1 类：来自 detect_divergence ─────────────────────────────
    for div in divergences:
        seg_c = segments[div.seg_c_idx]
        sig_type = "S1" if div.direction == Direction.UP else "B1"
        signals.append(TradeSignal(
            idx=0,
            signal_type=sig_type,
            dt=seg_c.end_fx.dt,
            price=seg_c.end_price,
            segment_idx=div.seg_c_idx,
            divergence=div,
            note=f"area C/A={div.ratio:.0%}",
        ))

        # ── 2 类：1 类后 +2 段回试不破 1 类价位 ───────────────
        # 段交替方向：c_idx 是 C 段，c_idx+1 反向，c_idx+2 与 C 同向
        c_idx = div.seg_c_idx
        if c_idx + 2 >= n_seg:
            continue
        seg_next = segments[c_idx + 1]
        seg_after = segments[c_idx + 2]

        if sig_type == "B1":
            # B1 的 C 段是 DOWN（创新低背驰）；之后 UP-DOWN 回试
            if (seg_next.direction == Direction.UP
                    and seg_after.direction == Direction.DOWN
                    and seg_after.end_price > seg_c.end_price):
                signals.append(TradeSignal(
                    idx=0, signal_type="B2",
                    dt=seg_after.end_fx.dt,
                    price=seg_after.end_price,
                    segment_idx=c_idx + 2,
                    note=f"after B1@{seg_c.end_fx.dt}",
                ))
        else:  # S1
            if (seg_next.direction == Direction.DOWN
                    and seg_after.direction == Direction.UP
                    and seg_after.end_price < seg_c.end_price):
                signals.append(TradeSignal(
                    idx=0, signal_type="S2",
                    dt=seg_after.end_fx.dt,
                    price=seg_after.end_price,
                    segment_idx=c_idx + 2,
                    note=f"after S1@{seg_c.end_fx.dt}",
                ))

    # ── 3 类：中枢已完成 + 离开段方向匹配 + 紧邻回试不破 ZG/ZD ────
    for p in pivots:
        if p.leaving_segment is None:
            continue
        leaving = p.leaving_segment
        leave_idx = leaving.idx
        if leave_idx + 1 >= n_seg:
            continue
        retest = segments[leave_idx + 1]
        # 段方向严格交替，retest 必与 leaving 反向 → 这里只做防御
        if retest.direction == leaving.direction:
            continue

        if leaving.direction == Direction.UP:
            # 中枢向上离开 → 等回试 DOWN 段低点是否 > ZG
            if retest.end_price > p.zg:
                signals.append(TradeSignal(
                    idx=0, signal_type="B3",
                    dt=retest.end_fx.dt,
                    price=retest.end_price,
                    segment_idx=leave_idx + 1,
                    pivot_idx=p.idx,
                    note=f"ZG={p.zg:.2f}",
                ))
        else:
            # 中枢向下离开 → 等回抽 UP 段高点是否 < ZD
            if retest.end_price < p.zd:
                signals.append(TradeSignal(
                    idx=0, signal_type="S3",
                    dt=retest.end_fx.dt,
                    price=retest.end_price,
                    segment_idx=leave_idx + 1,
                    pivot_idx=p.idx,
                    note=f"ZD={p.zd:.2f}",
                ))

    # 按时间（segment_idx）稳定排序；同段位 1/2/3 类按级别次序输出
    signals.sort(key=lambda s: (s.segment_idx, s.level))
    for i, s in enumerate(signals):
        s.idx = i
    return signals


# ══════════════════════════════════════════════════════════════════════
# 工具：从 pandas DataFrame 转 RawBar
# ══════════════════════════════════════════════════════════════════════

def from_dataframe(df, date_col="日期", open_col="开盘", high_col="最高",
                   low_col="最低", close_col="收盘", vol_col="成交量"):
    """
    把 akshare 风格的 DataFrame 转成 RawBar 列表。
    缺失列名会回落到英文（open/high/low/close/volume/date）。
    """
    cols = df.columns
    def pick(zh, en):
        return zh if zh in cols else en

    dc = pick(date_col, "date")
    oc = pick(open_col, "open")
    hc = pick(high_col, "high")
    lc = pick(low_col, "low")
    cc = pick(close_col, "close")
    vc = pick(vol_col, "volume")

    bars = []
    for i, row in df.reset_index(drop=True).iterrows():
        bars.append(RawBar(
            idx=int(i),
            dt=str(row[dc]),
            open=float(row[oc]),
            high=float(row[hc]),
            low=float(row[lc]),
            close=float(row[cc]),
            volume=float(row[vc]) if vc in cols else 0.0,
        ))
    return bars
