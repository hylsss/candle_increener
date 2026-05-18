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
  Phase 1.5  find_pivots      ── 中枢识别（第17/29课）               🚧
  Phase 1.6  detect_divergence── MACD背驰                            🚧
  Phase 1.7  find_signals     ── 1/2/3类买卖点                       🚧
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
        # 情况 A：fx 与 anchor 同性质（夹在它们之间的异性质分型已被废弃/合并）
        #   → 若 fx 更极，替换 anchor；若不如 anchor 极，丢弃 fx。
        if fx.ftype == anchor.ftype:
            more_extreme = (
                (fx.ftype == FractalType.TOP and fx.extreme > anchor.extreme)
                or (fx.ftype == FractalType.BOTTOM and fx.extreme < anchor.extreme)
            )
            if more_extreme:
                # 若 anchor 是上一笔的终点 → 上一笔需要撤回，因为终点不再是它
                if strokes and strokes[-1].end_fx is anchor:
                    popped = strokes.pop()
                    # 撤回后新 anchor 应回到上一笔的起点 → 由 fx 再去与之配对
                    anchor = popped.start_fx
                    # 不立刻 continue：让 fx 走一次主循环，可能与新 anchor 成笔
                    # 但本轮的 fx 与新 anchor 仍同性质（笔起点和终点必反），
                    # 实际上不会同性质——这分支不会出现。安全起见仍走完。
                anchor = fx
            # else: 丢弃 fx
            continue

        # 情况 B：fx 与 anchor 异性质但 _fractal_pair_valid 不通过（K线不够等）
        #   → 直接丢弃 fx（第77课"中间那些都 X 掉"的精神）。
        # （未来如需更精细，可在此插入"特殊处理"。）

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
