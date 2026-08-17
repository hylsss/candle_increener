"""
scan_b123.py  ── 缠论一/二/三类买点统一选股
────────────────────────────────────────────
扫描全市场（主板按成交额排序），找出近 N 个交易日内触发缠论
B1/B2/B3 信号且通过过滤条件的股票，按信号质量评分排列输出。

信号说明：
  B1  背驰买点：下跌段 MACD 面积背驰，信号最早但噪音较多
  B2  确认买点：B1 后回试不破，风险收益最均衡
  B3  突破买点：中枢向上离开后回踩 ZG 不破，最稳但信号最少

质量评分（0-100）：
  B1  背驰力度(C/A面积比) + 0轴位置 + MA60偏离 + 趋势持续度
  B2  回试幅度合理性 + 缩量确认 + 均线位置
  B3  中枢质量 + ZG安全边际 + 离开段强度 + 均线多头

用法：
  python scan_b123.py                      # 扫描成交额前 300 只（默认 B2+B3）
  B123_SAMPLE=500 python scan_b123.py      # 扩大范围
  B123_LOOKBACK=20 python scan_b123.py     # 缩短信号有效期（交易日）
  B123_TYPES=B1,B2,B3 python scan_b123.py # 指定信号类型
  B123_MIN_SCORE=40 python scan_b123.py   # 提高质量门槛
  SPREADSHEET_ID=xxx python scan_b123.py  # 写入 Google Sheets

环境变量：
  B123_SAMPLE    扫描股票数（成交额排序）   default 300
  B123_LOOKBACK  信号有效期（交易日）       default 5（严格模式）
  B123_DAYS      历史数据日历天数           default 800
  B123_TYPES     信号类型（逗号分隔）       default B2,B3
  B123_MIN_SCORE 最低质量分（0-100）        default 25
  B123_WORKERS   并发数                     default 1
  B123_STRICT    =1 启用严格缠论硬过滤       default 1
  SPREADSHEET_ID Google Sheets ID
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import (
    ThreadPoolExecutor, as_completed,
    TimeoutError as FuturesTimeout,
)
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import local_patch  # noqa: F401
import akshare as ak

from full_scan import get_universe
from chanlun_core import (
    Direction, TradeSignal,
    from_dataframe, merge_klines, find_fractals, find_strokes,
    find_segments, find_pivots, detect_divergence, find_signals,
)


def _pivot_end_idx(pivot) -> int:
    return max(s.idx for s in pivot.segments)


def _strict_trend_pivots(div, pivots) -> list:
    """返回背驰 C 段之前、构成同向趋势的已完成中枢。

    严格第一类买点的前提是趋势背驰，而不是任意相邻线段的 MACD
    面积缩小。下跌趋势要求后中枢 GG < 前中枢 DD；上涨趋势镜像。
    """
    eligible = [
        p for p in pivots
        if p.is_finished and _pivot_end_idx(p) < div.seg_c_idx
    ]
    eligible.sort(key=lambda p: p.segments[0].idx)
    trend = []
    for p in eligible:
        if not trend:
            trend.append(p)
            continue
        prev = trend[-1]
        separated = (
            p.gg < prev.dd if div.direction == Direction.DOWN
            else p.dd > prev.gg
        )
        if separated:
            trend.append(p)
        else:
            # 中枢区间/波动区间重叠意味着扩展或盘整，不能冒充趋势。
            trend = [p]
    return trend


def _strict_signal_check(sig, signals, segments, pivots,
                         closes: np.ndarray, sig_idx: int,
                         last_idx: int, max_age: int) -> tuple[bool, str, dict]:
    """按缠论定义做硬过滤；均线、量能不参与信号真伪判定。"""
    if last_idx - sig_idx > max_age:
        return False, "信号过旧", {}
    if sig.segment_idx >= len(segments):
        return False, "信号线段不存在", {}

    signal_seg = segments[sig.segment_idx]
    if signal_seg.break_type == 0:
        return False, "信号线段尚未被反向线段破坏确认", {}

    # 买点之后若已有更晚的同级别卖点，买点已完成其生命周期。
    if any(s.is_sell and s.segment_idx > sig.segment_idx for s in signals):
        return False, "之后已出现同级别卖点", {}

    cur = float(closes[-1])
    context: dict = {"结构合规": True, "结构级别": "线段"}

    if sig.signal_type == "B1":
        div = sig.divergence
        if div is None or div.direction != Direction.DOWN:
            return False, "缺少下跌趋势背驰", {}
        if div.diff_a >= 0:
            return False, "第一类买点不在MACD零轴下方", {}
        trend_pivots = _strict_trend_pivots(div, pivots)
        if len(trend_pivots) < 2:
            return False, "不足两个依次向下且不重叠的同级别中枢", {}
        if cur < sig.price:
            return False, "当前价已跌破第一类买点", {}
        context.update({
            "趋势中枢数": len(trend_pivots),
            "背驰比(C/A)": round(div.ratio, 3),
        })

    elif sig.signal_type == "B2":
        # 二买必须来自一个严格有效的一买，不能只凭“不创新低”命名。
        parent = next((
            s for s in signals
            if s.signal_type == "B1" and s.segment_idx == sig.segment_idx - 2
        ), None)
        if parent is None or parent.divergence is None:
            return False, "缺少对应第一类买点", {}
        trend_pivots = _strict_trend_pivots(parent.divergence, pivots)
        if len(trend_pivots) < 2 or parent.divergence.diff_a >= 0:
            return False, "前置第一类买点不是严格趋势背驰", {}
        rally = segments[sig.segment_idx - 1]
        if rally.break_type == 0:
            return False, "一买后的向上走势尚未完成", {}
        if sig.price <= parent.price:
            return False, "回试跌破第一类买点", {}
        if cur < sig.price:
            return False, "当前价已跌破第二类买点", {}
        context.update({"前置B1价": round(parent.price, 2), "趋势中枢数": len(trend_pivots)})

    elif sig.signal_type == "B3":
        if sig.pivot_idx is None or sig.pivot_idx >= len(pivots):
            return False, "缺少对应中枢", {}
        pivot = pivots[sig.pivot_idx]
        leaving = pivot.leaving_segment
        if not pivot.is_finished or leaving is None:
            return False, "中枢尚未完成离开", {}
        if leaving.direction != Direction.UP or leaving.break_type == 0:
            return False, "向上离开走势尚未完成", {}
        if leaving.idx + 1 != sig.segment_idx:
            return False, "不是离开中枢后的第一次回试", {}
        if leaving.low <= pivot.zg:
            return False, "离开走势未整体脱离中枢上沿", {}
        if signal_seg.direction != Direction.DOWN or signal_seg.low <= pivot.zg:
            return False, "回试跌破中枢上沿ZG", {}
        if cur <= pivot.zg:
            return False, "当前价已返回中枢", {}
        context.update({"ZG": round(pivot.zg, 2), "ZD": round(pivot.zd, 2),
                        "中枢段数": len(pivot.segments)})
    else:
        return False, "严格选股只接受B1/B2/B3", {}

    return True, "", context


def _strict_score(sig, context: dict) -> int:
    """只用缠论结构信息排序，不把均线/黄金分割伪装成买点依据。"""
    if sig.signal_type == "B1":
        ratio = float(context.get("背驰比(C/A)", 1.0))
        return max(60, min(100, round(105 - ratio * 50)))
    if sig.signal_type == "B2":
        return min(100, 80 + 5 * int(context.get("趋势中枢数", 2)))
    return min(100, 75 + 4 * int(context.get("中枢段数", 3)))


# ══════════════════════════════════════════════════════════════════════
# 信号质量评分
# ══════════════════════════════════════════════════════════════════════

def _score_b1(sig: TradeSignal, segments, hist_closes: np.ndarray,
              hist_volumes: np.ndarray | None, sig_idx: int) -> tuple[int, dict]:
    """
    B1 评分（0-100）：
      - 背驰力度（C/A 面积比，越小越强）             40 分
      - MACD 0 轴位置（A 段 DIFF < 0，真正底部背驰） 15 分
      - MA60 偏离度（信号日，不能太深也不太差）        25 分
      - 量比（信号日相对20日均量）                    10 分
      - 下跌段数量（趋势成熟度）                      10 分
    """
    score = 0
    extra: dict = {}

    # ── 背驰力度 ───────────────────────────────────────────
    div = sig.divergence
    ratio = div.ratio if div else 1.0
    extra["背驰比(C/A)"] = round(ratio, 2)
    if ratio < 0.30:
        score += 40
    elif ratio < 0.50:
        score += 28
    elif ratio < 0.70:
        score += 15
    elif ratio < 0.85:
        score += 5

    # ── 0 轴判定（A 段 DIFF 极值 < 0 = 真正底部背驰）──────
    if div and div.diff_a < 0:
        score += 15
    extra["A段DIFF极值"] = round(div.diff_a, 4) if div else 0.0

    # ── MA60 偏离 ──────────────────────────────────────────
    ma60 = float(hist_closes[max(0, sig_idx - 59):sig_idx + 1].mean())
    cur_close = float(hist_closes[sig_idx])
    ma60_pct = round((cur_close - ma60) / ma60 * 100, 1) if ma60 > 0 else 0.0
    extra["MA60偏离%"] = ma60_pct
    if ma60_pct > -3:
        score += 25
    elif ma60_pct > -8:
        score += 18
    elif ma60_pct > -15:
        score += 8

    # ── 量比（信号日 vs 20 日均量）────────────────────────
    if hist_volumes is not None and sig_idx >= 5:
        avg_vol = float(hist_volumes[max(0, sig_idx - 19):sig_idx].mean())
        vr = round(float(hist_volumes[sig_idx]) / avg_vol, 2) if avg_vol > 0 else 1.0
    else:
        vr = 1.0
    extra["量比(信号日)"] = vr
    if vr < 0.5:
        score += 3   # 极度缩量，底部可能，但也有风险
    elif vr <= 1.2:
        score += 10  # 温和量
    elif vr <= 2.0:
        score += 6

    # ── 趋势成熟度：信号段之前有几条向下段 ────────────────
    c_idx = sig.segment_idx
    down_segs_before = sum(
        1 for s in segments[:c_idx] if s.direction == Direction.DOWN
    )
    if down_segs_before >= 4:
        score += 10
    elif down_segs_before >= 2:
        score += 6

    extra["下跌段数"] = down_segs_before
    return min(score, 100), extra


def _score_b2(sig: TradeSignal, segments, hist_closes: np.ndarray,
              hist_volumes: np.ndarray | None, sig_idx: int) -> tuple[int, dict]:
    """
    B2 评分（0-100）：
      - 回试幅度（黄金回调比：0.38-0.65 最优）         35 分
      - 缩量确认（信号日量比 < 1）                      20 分
      - MA60 偏离度（信号日，距均线不能太远）            25 分
      - MA 趋势（MA20 > MA60 则趋势偏多）               20 分
    """
    score = 0
    extra: dict = {}

    # ── 回试幅度：计算 fibonacci 回调比 ──────────────────
    b2_seg_idx = sig.segment_idx          # DOWN 回试段
    rally_seg_idx = b2_seg_idx - 1        # UP 反弹段
    b1_seg_idx = b2_seg_idx - 2           # B1 所在 DOWN 段

    retest_ratio = 0.5  # 默认中性
    if b1_seg_idx >= 0 and rally_seg_idx >= 0:
        b1_price = segments[b1_seg_idx].end_price
        rally_high = segments[rally_seg_idx].high
        b2_price = sig.price
        span = rally_high - b1_price
        if span > 0:
            retest_ratio = round((rally_high - b2_price) / span, 2)

    extra["回试幅度"] = retest_ratio
    if 0.38 <= retest_ratio <= 0.65:
        score += 35  # 黄金比例回调
    elif 0.25 <= retest_ratio < 0.38 or 0.65 < retest_ratio <= 0.80:
        score += 20
    elif retest_ratio < 0.25 or retest_ratio > 0.80:
        score += 8   # 回调过浅或过深

    # ── 信号日量比 ────────────────────────────────────────
    if hist_volumes is not None and sig_idx >= 5:
        avg_vol = float(hist_volumes[max(0, sig_idx - 19):sig_idx].mean())
        vr = round(float(hist_volumes[sig_idx]) / avg_vol, 2) if avg_vol > 0 else 1.0
    else:
        vr = 1.0
    extra["量比(信号日)"] = vr
    if vr <= 0.5:
        score += 5   # 极度缩量，稍谨慎
    elif vr <= 0.8:
        score += 20  # 缩量最佳
    elif vr <= 1.2:
        score += 14

    # ── MA60 偏离 ──────────────────────────────────────────
    ma60 = float(hist_closes[max(0, sig_idx - 59):sig_idx + 1].mean())
    cur_close = float(hist_closes[sig_idx])
    ma60_pct = round((cur_close - ma60) / ma60 * 100, 1) if ma60 > 0 else 0.0
    extra["MA60偏离%"] = ma60_pct
    if ma60_pct > 5:
        score += 25
    elif ma60_pct > 0:
        score += 20
    elif ma60_pct > -5:
        score += 12
    elif ma60_pct > -10:
        score += 5

    # ── MA 趋势（均线多头排列）────────────────────────────
    ma20 = float(hist_closes[max(0, sig_idx - 19):sig_idx + 1].mean())
    if ma20 > ma60:
        score += 20
    elif ma20 > ma60 * 0.97:
        score += 10

    extra["MA20(信号日)"] = round(ma20, 2)
    extra["MA60(信号日)"] = round(ma60, 2)
    return min(score, 100), extra


def _score_b3(sig: TradeSignal, segments, pivots, hist_closes: np.ndarray,
              hist_volumes: np.ndarray | None, sig_idx: int) -> tuple[int, dict]:
    """
    B3 评分（0-100）：
      - 中枢成员段数（段数越多中枢越可靠）             25 分
      - ZG 安全边际（回试价高于 ZG 的距离）            25 分
      - 离开段强度（突破 ZG 的幅度）                   20 分
      - MA 趋势（MA20 > MA60 均线多头）                20 分
      - 离开段放量                                     10 分
    """
    score = 0
    extra: dict = {}

    # ── 中枢质量 ──────────────────────────────────────────
    pivot = pivots[sig.pivot_idx] if sig.pivot_idx is not None else None
    if pivot:
        n_segs = len(pivot.segments)
        extra["中枢段数"] = n_segs
        extra["ZG"] = round(pivot.zg, 2)
        extra["ZD"] = round(pivot.zd, 2)
        if n_segs >= 6:
            score += 25
        elif n_segs >= 4:
            score += 18
        elif n_segs >= 3:
            score += 10

        # ── ZG 安全边际 ───────────────────────────────────
        zg = pivot.zg
        b3_price = sig.price
        margin_pct = (b3_price - zg) / zg * 100 if zg > 0 else 0.0
        extra["ZG安全边际%"] = round(margin_pct, 1)
        if margin_pct > 3.0:
            score += 25
        elif margin_pct > 1.0:
            score += 18
        elif margin_pct > 0:
            score += 10

        # ── 离开段强度：leaving_segment high vs ZG ────────
        leaving = pivot.leaving_segment
        if leaving:
            leave_strength = (leaving.high - zg) / zg * 100 if zg > 0 else 0.0
            extra["离开段强度%"] = round(leave_strength, 1)
            if leave_strength > 8:
                score += 20
            elif leave_strength > 3:
                score += 13
            elif leave_strength > 0:
                score += 6
        else:
            extra["离开段强度%"] = 0.0
    else:
        extra["中枢段数"] = 0
        extra["ZG"] = 0.0
        extra["ZD"] = 0.0
        extra["ZG安全边际%"] = 0.0
        extra["离开段强度%"] = 0.0

    # ── MA 趋势 ───────────────────────────────────────────
    ma20 = float(hist_closes[max(0, sig_idx - 19):sig_idx + 1].mean())
    ma60 = float(hist_closes[max(0, sig_idx - 59):sig_idx + 1].mean())
    extra["MA20(信号日)"] = round(ma20, 2)
    extra["MA60(信号日)"] = round(ma60, 2)
    if ma20 > ma60:
        score += 20
    elif ma20 > ma60 * 0.97:
        score += 10

    # ── 离开段放量（用信号日前后均量代估）────────────────
    if hist_volumes is not None and sig_idx >= 5:
        avg_vol = float(hist_volumes[max(0, sig_idx - 19):sig_idx].mean())
        vr = round(float(hist_volumes[sig_idx]) / avg_vol, 2) if avg_vol > 0 else 1.0
    else:
        vr = 1.0
    extra["量比(信号日)"] = vr
    if vr > 1.5:
        score += 10
    elif vr > 1.1:
        score += 6

    return min(score, 100), extra


# ══════════════════════════════════════════════════════════════════════
# 单股检测
# ══════════════════════════════════════════════════════════════════════

def _detect_b123(code: str, name: str, cur_price,
                 hist: pd.DataFrame,
                 lookback_days: int,
                 target_types: set[str],
                 min_score: int,
                 strict_mode: bool = True) -> list[dict]:
    """
    对单只股票跑完整 pipeline，返回近 lookback_days 内
    符合 target_types 且 score >= min_score 的信号记录列表。
    每个信号独立输出（同一只股票可同时触发 B2+B3）。
    """
    if len(hist) < 120:
        return []

    try:
        raws = from_dataframe(hist)
        merged = merge_klines(raws)
        fxs = find_fractals(merged)
        strokes = find_strokes(merged, fxs, new_stroke=True)
        segments = find_segments(strokes)
        if len(segments) < 3:
            return []
        pivots = find_pivots(segments)
        divs = detect_divergence(
            segments, merged, raws, require_zero_axis=strict_mode)
        signals = find_signals(segments, pivots, divs)
    except Exception:
        return []

    date_to_idx = {str(d): i for i, d in enumerate(hist["日期"])}
    last_idx = len(hist) - 1
    closes = hist["收盘"].to_numpy(dtype=float)
    volumes = hist["成交量"].to_numpy(dtype=float) if "成交量" in hist.columns else None

    # 当前状态
    cur = float(closes[-1])
    cur_ma20 = float(closes[max(0, last_idx - 19):last_idx + 1].mean())
    cur_ma60 = float(closes[max(0, last_idx - 59):last_idx + 1].mean())

    records = []
    for sig in signals:
        if sig.signal_type not in target_types:
            continue

        sig_idx = date_to_idx.get(str(sig.dt))
        if sig_idx is None:
            continue
        dist = last_idx - sig_idx
        if dist > lookback_days:
            continue

        strict_context = {}
        if strict_mode:
            ok, _, strict_context = _strict_signal_check(
                sig, signals, segments, pivots, closes, sig_idx, last_idx,
                max_age=lookback_days,
            )
            if not ok:
                continue

        # 宽松兼容模式才使用均线/量能硬过滤；严格模式中它们不是缠论定义。
        ma60_at_sig = float(closes[max(0, sig_idx - 59):sig_idx + 1].mean())
        close_at_sig = float(closes[sig_idx])
        ma60_pct = round((close_at_sig - ma60_at_sig) / ma60_at_sig * 100, 1) if ma60_at_sig > 0 else 0.0

        if not strict_mode:
            ma60_limit = -15.0 if sig.signal_type == "B1" else -10.0
            if ma60_pct < ma60_limit:
                continue

        # 公共过滤：量比 > 0.5
        if volumes is not None and sig_idx >= 5:
            avg_vol = float(volumes[max(0, sig_idx - 19):sig_idx].mean())
            vr_check = float(volumes[sig_idx]) / avg_vol if avg_vol > 0 else 1.0
        else:
            vr_check = 1.0
        if not strict_mode and vr_check <= 0.5:
            continue

        # 评分
        if strict_mode:
            score, extra = _strict_score(sig, strict_context), strict_context
        elif sig.signal_type == "B1":
            score, extra = _score_b1(sig, segments, closes, volumes, sig_idx)
        elif sig.signal_type == "B2":
            score, extra = _score_b2(sig, segments, closes, volumes, sig_idx)
        else:  # B3
            score, extra = _score_b3(sig, segments, pivots, closes, volumes, sig_idx)

        if score < min_score:
            continue

        price_chg = round((cur - sig.price) / sig.price * 100, 1) if sig.price > 0 else 0.0

        # 仅供观察的行情辅助指标；严格模式不据此确认或排序买点。
        trend_score = 0
        if cur > cur_ma20:       trend_score += 30
        if cur_ma20 > cur_ma60:  trend_score += 30
        if cur > cur_ma60:       trend_score += 20
        if volumes is not None:
            avg20 = float(volumes[max(0, last_idx - 19):last_idx].mean())
            if avg20 > 0 and float(volumes[last_idx]) > avg20 * 1.1:
                trend_score += 10
        # distance penalty
        if dist <= 5:            trend_score += 10

        record = {
            "代码":         code,
            "名称":         name,
            "信号类型":     sig.signal_type,
            "信号日期":     sig.dt,
            "距今(交易日)": dist,
            "信号价":       round(sig.price, 2),
            "当前价":       round(cur, 2),
            "距信号涨幅%":  price_chg,
            "质量分":       score,
            "趋势评分":     trend_score,
            "当前MA20":     round(cur_ma20, 2),
            "当前MA60":     round(cur_ma60, 2),
            "MA60偏离%(信号日)": ma60_pct,
            "信号备注":     sig.note,
            "判定模式":     "严格缠论" if strict_mode else "兼容模式",
            "扫描时间":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        # 合并信号类型专有字段
        record.update({k: v for k, v in extra.items()})
        records.append(record)

    return records


# ══════════════════════════════════════════════════════════════════════
# 主扫描入口
# ══════════════════════════════════════════════════════════════════════

def scan_b123(sample_size: int = 300,
              lookback_days: int = 5,
              period_days: int = 800,
              target_types: set[str] | None = None,
              min_score: int = 25,
              workers: int = 1,
              strict_mode: bool = True) -> pd.DataFrame:
    """
    扫描全市场，返回满足条件的 B1/B2/B3 信号 DataFrame。
    同一只股票可出现多行（不同信号类型）。
    """
    if target_types is None:
        target_types = {"B2", "B3"}

    type_str = "+".join(sorted(target_types))
    print(f"▶ 缠论买点扫描: [{type_str}] | 历史 {period_days} 天 | "
          f"有效期 {lookback_days} 交易日 | 最低质量分 {min_score} | "
          f"模式 {'严格缠论' if strict_mode else '兼容'}")

    uni = get_universe(min_price=3.0, max_price=500.0,
                       min_turnover=0.5, main_board_only=True)
    uni = uni.sort_values("成交额_亿元", ascending=False).head(sample_size)
    name_map = dict(zip(uni["代码"].astype(str), uni["名称"].astype(str)))
    price_map = dict(zip(uni["代码"].astype(str), uni["当前价"]))
    codes = list(uni["代码"].astype(str))
    print(f"📋 股票池: {len(codes)} 只（主板按成交额排序）")

    histories = _fetch_histories(codes, period_days, workers)
    print(f"✅ 历史数据: {len(histories)} 只")

    all_records: list[dict] = []
    total = len(histories)
    for i, (code, hist) in enumerate(histories.items(), 1):
        recs = _detect_b123(
            code=code,
            name=name_map.get(code, ""),
            cur_price=price_map.get(code),
            hist=hist,
            lookback_days=lookback_days,
            target_types=target_types,
            min_score=min_score,
            strict_mode=strict_mode,
        )
        all_records.extend(recs)
        if i % 50 == 0:
            print(f"   ...进度 {i}/{total}，命中 {len(all_records)} 条信号")

    print(f"\n✅ 扫描完成: {total} 只 → 信号命中 {len(all_records)} 条")

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # 信号类型权重（B3 最稳排前，B2 次之，B1 最后）
    type_rank = {"B3": 0, "B2": 1, "B1": 2}
    df["_type_rank"] = df["信号类型"].map(type_rank).fillna(3)
    sort_cols = ["_type_rank", "质量分"]
    ascending = [True, False]
    if not strict_mode:
        sort_cols.append("趋势评分")
        ascending.append(False)
    df = df.sort_values(sort_cols, ascending=ascending).drop(columns=["_type_rank"])

    return df.reset_index(drop=True)


def _fetch_histories(codes: list[str], period_days: int, workers: int) -> dict:
    start = (datetime.now() - timedelta(days=int(period_days * 1.4))).strftime("%Y%m%d")
    end = datetime.now().strftime("%Y%m%d")

    def _one(code):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start, end_date=end, adjust="qfq",
            )
            if df is None or df.empty:
                return code, None
            cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
            df = df[[c for c in cols if c in df.columns]].copy()
            for c in ["开盘", "收盘", "最高", "最低", "成交量"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["开盘", "收盘", "最高", "最低"]).reset_index(drop=True)
            df["日期"] = df["日期"].astype(str)
            return code, df if len(df) >= 120 else None
        except Exception:
            return code, None

    results: dict = {}
    done = 0
    total = len(codes)
    # 整体超时兜底：即便某请求绕过了 socket 超时，也不让它拖死扫描。
    # 上限 = 每只票 8s 估算 + 60s 余量（socket 超时本身已是 25s 下限）。
    per_stock = float(os.environ.get("CB_FETCH_PER_STOCK", "8"))
    overall = max(120, int(total * per_stock / max(workers, 1)) + 60)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, c): c for c in codes}
        try:
            for f in as_completed(futs, timeout=overall):
                code, df = f.result()
                done += 1
                if df is not None:
                    results[code] = df
                if done % 50 == 0 or done == total:
                    print(f"   ...预拉历史 {done}/{total}")
        except FuturesTimeout:
            stuck = total - done
            print(f"   ⚠️  历史拉取整体超时（{overall}s），跳过未完成的 {stuck} 只")
    return results


# ══════════════════════════════════════════════════════════════════════
# Google Sheets 输出（复用 scan_b2 风格）
# ══════════════════════════════════════════════════════════════════════

# 每个信号类型的主要展示列（公共列 + 专有列）
_COMMON_COLS = [
    "代码", "名称", "信号类型", "信号日期", "距今(交易日)",
    "信号价", "当前价", "距信号涨幅%", "质量分", "趋势评分",
    "当前MA20", "当前MA60", "MA60偏离%(信号日)", "信号备注", "扫描时间",
]
_B1_EXTRA = ["背驰比(C/A)", "A段DIFF极值", "量比(信号日)", "下跌段数"]
_B2_EXTRA = ["回试幅度", "量比(信号日)", "MA20(信号日)", "MA60(信号日)"]
_B3_EXTRA = ["中枢段数", "ZG", "ZD", "ZG安全边际%", "离开段强度%",
             "量比(信号日)", "MA20(信号日)", "MA60(信号日)"]


def _ordered_cols(df: pd.DataFrame) -> list[str]:
    """返回有序列名：公共列优先，再补充信号专有列，最后其余列。"""
    extra = _B1_EXTRA + _B2_EXTRA + _B3_EXTRA
    seen = set()
    ordered = []
    for c in _COMMON_COLS + extra:
        if c in df.columns and c not in seen:
            ordered.append(c)
            seen.add(c)
    for c in df.columns:
        if c not in seen:
            ordered.append(c)
    return ordered


def write_to_sheet(df: pd.DataFrame, spreadsheet_id: str,
                   key_file: str = "service_account.json"):
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        print("⚠️  gspread 未安装")
        return

    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    try:
        creds = Credentials.from_service_account_file(key_file, scopes=SCOPES)
        gc = gspread.authorize(creds)
        book = gc.open_by_key(spreadsheet_id)
    except Exception as e:
        print(f"⚠️  Sheets 连接失败: {e}")
        return

    import math

    now = datetime.now().strftime("%Y-%m-%d %H%M")
    tab_name = f"🎯 缠论买点 {now}"
    ordered = _ordered_cols(df)
    df_out = df[ordered].copy()

    # 日期列强制转为字符串，防止 Sheets 把日期解析成序列号
    for col in df_out.columns:
        if "日期" in col or "时间" in col:
            df_out[col] = df_out[col].astype(str)

    # tab 已存在就清空复用，不存在则新建；若新建时仍冲突（极端并发）则再取一次
    try:
        ws = book.worksheet(tab_name)
        ws.clear()
    except Exception:
        try:
            ws = book.add_worksheet(title=tab_name,
                                    rows=max(len(df_out) + 5, 10),
                                    cols=len(df_out.columns))
        except Exception:
            ws = book.worksheet(tab_name)
            ws.clear()

    def _safe(v):
        if isinstance(v, float) and math.isnan(v):
            return ""
        try:
            import numpy as np
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v) if not math.isnan(float(v)) else ""
        except Exception:
            pass
        return str(v) if hasattr(v, 'isoformat') else v  # datetime → 字符串

    rows = [list(df_out.columns)]
    for _, r in df_out.iterrows():
        rows.append([_safe(v) for v in r])

    # RAW 模式写入防止 Sheets 自动转换格式
    ws.update(rows, "A1", value_input_option="RAW")

    sheet_id = ws._properties["sheetId"]
    n_cols = len(df_out.columns)
    n_rows = len(df_out)

    # 行颜色：B3=浅蓝，B2=浅绿，B1=浅黄
    color_map = {
        "B3": {"red": 0.82, "green": 0.90, "blue": 0.98},
        "B2": {"red": 0.85, "green": 0.97, "blue": 0.85},
        "B1": {"red": 0.99, "green": 0.97, "blue": 0.82},
    }
    type_col_idx = list(df_out.columns).index("信号类型") if "信号类型" in df_out.columns else -1

    requests = [
        {"repeatCell": {
            "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1,
                      "startColumnIndex": 0, "endColumnIndex": n_cols},
            "cell": {"userEnteredFormat": {
                "textFormat": {"bold": True, "fontSize": 10,
                               "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
                "backgroundColor": {"red": 0.12, "green": 0.23, "blue": 0.37},
                "horizontalAlignment": "CENTER",
            }},
            "fields": "userEnteredFormat",
        }},
        {"updateSheetProperties": {
            "properties": {"sheetId": sheet_id,
                           "gridProperties": {"frozenRowCount": 1}},
            "fields": "gridProperties.frozenRowCount",
        }},
        {"setBasicFilter": {"filter": {
            "range": {"sheetId": sheet_id, "startRowIndex": 0,
                      "endRowIndex": n_rows + 1,
                      "startColumnIndex": 0, "endColumnIndex": n_cols},
        }}},
    ]

    for ri, (_, row) in enumerate(df_out.iterrows()):
        sig_type = str(row.get("信号类型", "B2"))
        bg = color_map.get(sig_type, color_map["B2"])
        requests.append({"repeatCell": {
            "range": {"sheetId": sheet_id,
                      "startRowIndex": ri + 1, "endRowIndex": ri + 2,
                      "startColumnIndex": 0, "endColumnIndex": n_cols},
            "cell": {"userEnteredFormat": {
                "backgroundColor": bg,
                "horizontalAlignment": "CENTER",
                "textFormat": {"fontSize": 9},
            }},
            "fields": "userEnteredFormat",
        }})

    # 列宽自适应
    requests.append({"autoResizeDimensions": {
        "dimensions": {
            "sheetId": sheet_id,
            "dimension": "COLUMNS",
            "startIndex": 0,
            "endIndex": n_cols,
        }
    }})

    book.batch_update({"requests": requests})
    print(f"✅ 已写入 Google Sheets tab：{tab_name}（{n_rows} 条信号）")
    print(f"   https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit")


# ══════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════

def main():
    sample_size = int(os.environ.get("B123_SAMPLE", 300))
    strict_mode = os.environ.get("B123_STRICT", "1") != "0"
    lookback    = int(os.environ.get("B123_LOOKBACK", 5 if strict_mode else 30))
    period_days = int(os.environ.get("B123_DAYS", 800))
    types_str   = os.environ.get("B123_TYPES", "B2,B3")
    min_score   = int(os.environ.get("B123_MIN_SCORE", 25))
    workers     = int(os.environ.get("B123_WORKERS", 1))
    sheet_id    = os.environ.get("SPREADSHEET_ID", "")
    key_file    = os.environ.get("CB_KEY", "service_account.json")

    target_types = {t.strip().upper() for t in types_str.split(",") if t.strip()}

    print(f"=== 缠论买点选股 {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")
    df = scan_b123(
        sample_size=sample_size,
        lookback_days=lookback,
        period_days=period_days,
        target_types=target_types,
        min_score=min_score,
        workers=workers,
        strict_mode=strict_mode,
    )

    if df.empty:
        print("⚠️  未找到符合条件的买点信号。")
        sys.exit(0)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = f"买点选股_{ts}.xlsx"
    df.to_excel(out, index=False)
    print(f"\n✅ Excel: {out}")

    # 分类型汇总
    print(f"\n{'='*60}")
    for sig_type in ["B3", "B2", "B1"]:
        sub = df[df["信号类型"] == sig_type]
        if sub.empty:
            continue
        print(f"\n── {sig_type} 信号（{len(sub)} 只）──")
        show_cols = ["代码", "名称", "信号日期", "距今(交易日)",
                     "信号价", "当前价", "距信号涨幅%", "质量分", "趋势评分"]
        show_cols = [c for c in show_cols if c in sub.columns]
        print(sub[show_cols].head(20).to_string(index=False))

    if sheet_id:
        write_to_sheet(df, sheet_id, key_file)
    else:
        print("\n⚠️  未设置 SPREADSHEET_ID，跳过 Sheets 写入")


if __name__ == "__main__":
    main()
