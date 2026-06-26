"""
multi_level.py  ── 多级别缠论（日线 / 60分钟 / 30分钟）
────────────────────────────────────────────────────
缠论 pipeline 与「级别」无关——只要喂进对应级别的 K 线即可。
本模块给每只票同时跑 日线 / 60分 / 30分 三个级别，找出更短线的买点，
并做「级别共振」判断：

  日线趋势向上（给方向） + 60分或30分出现 B2/B3 买点（给短线入场点）
  → 才算高确定性的短线买点

这是对原项目「只有日线买点（偏中线）」的补强，对应需求「不要长线」。

数据源：分钟线用新浪 stock_zh_a_minute（period=30/60），不复权（adjust=''），
        今日 bar 完整、历史 1~2 年，且东财分钟接口挂掉时不受影响。
        日线沿用 _fetch_histories（qfq）。

复用：scan_b123._detect_b123 做各级别的 B1/B2/B3 检测与评分。

用法：
  python multi_level.py 600732              # 单只多级别诊断
  python multi_level.py 600732 爱旭股份
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import numpy as np
import pandas as pd

import local_patch  # noqa: F401
import akshare as ak

from chanlun_core import (
    from_dataframe, merge_klines, find_fractals, find_strokes,
    find_segments, find_pivots, Direction,
)
from scan_b123 import _detect_b123, _fetch_histories


# 级别配置：
#   lookback 检测信号时回看的最大 K 线数
#   fresh    共振判定时认为「仍可操作」的最大距今 K 线数（比 lookback 更严）
LEVELS = {
    "daily": {"label": "日线",   "lookback": 20, "fresh": 10},
    "60":    {"label": "60分钟", "lookback": 24, "fresh": 16},  # 4 根/日
    "30":    {"label": "30分钟", "lookback": 32, "fresh": 24},  # 8 根/日
}

# 当前价低于信号价超过此比例，视为已跌破买点 → 信号失效（可用 ML_BREAK_TOL 调整）
_BREAK_TOL = float(os.environ.get("ML_BREAK_TOL", "0.01"))


# ══════════════════════════════════════════════════════════════════════
# 数据获取
# ══════════════════════════════════════════════════════════════════════
def _code_to_sina(code: str) -> str:
    """600xxx/601/603/605/688 → sh；000/001/002/003/300 → sz。"""
    code = str(code).zfill(6)
    return ("sh" if code[0] == "6" else "sz") + code


def fetch_level_df(code: str, period: str) -> pd.DataFrame | None:
    """
    返回某级别 K 线，列为中文（日期/开盘/最高/最低/收盘/成交量），
    供 from_dataframe 与 _detect_b123 直接消费。period ∈ {daily,60,30}。
    """
    if period == "daily":
        hist = _fetch_histories([str(code).zfill(6)], period_days=800, workers=1)
        return hist.get(str(code).zfill(6))

    symbol = _code_to_sina(code)
    try:
        df = ak.stock_zh_a_minute(symbol=symbol, period=period, adjust="")
    except Exception:  # noqa: BLE001
        return None
    if df is None or df.empty:
        return None

    df = df.rename(columns={
        "day": "日期", "open": "开盘", "high": "最高",
        "low": "最低", "close": "收盘", "volume": "成交量",
    })
    for c in ["开盘", "最高", "最低", "收盘", "成交量"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["开盘", "最高", "最低", "收盘"]).reset_index(drop=True)
    df["日期"] = df["日期"].astype(str)
    return df if len(df) >= 120 else None


# ══════════════════════════════════════════════════════════════════════
# 单级别分析
# ══════════════════════════════════════════════════════════════════════
def _macd_bar(closes: np.ndarray) -> tuple[float, str]:
    """返回 (最新 MACD 柱, 近 3 根趋势 '走强'/'走弱'/'-')。"""
    def ema(v, n):
        k = 2 / (n + 1)
        out = [float(v[0])]
        for x in v[1:]:
            out.append(out[-1] * (1 - k) + float(x) * k)
        return np.array(out)
    if len(closes) < 30:
        return 0.0, "-"
    diff = ema(closes, 12) - ema(closes, 26)
    dea = ema(diff, 9)
    bar = 2 * (diff - dea)
    trend = "-"
    if len(bar) >= 3:
        trend = "走强" if bar[-1] > bar[-2] > bar[-3] else (
            "走弱" if bar[-1] < bar[-2] < bar[-3] else "-")
    return round(float(bar[-1]), 4), trend


def analyze_level(code: str, name: str, period: str,
                  target_types: set[str], min_score: int) -> dict:
    """跑单级别 pipeline，返回该级别的末笔状态 + 近期 B1/B2/B3 信号。"""
    cfg = LEVELS[period]
    out = {"级别": cfg["label"], "可用": False, "末笔": "-",
           "MACD柱": 0.0, "MACD趋势": "-", "信号": [], "当前价": None}

    df = fetch_level_df(code, period)
    if df is None:
        return out

    closes = df["收盘"].to_numpy(dtype=float)
    out["当前价"] = round(float(closes[-1]), 2)
    out["MACD柱"], out["MACD趋势"] = _macd_bar(closes)

    # 末笔方向
    try:
        raws = from_dataframe(df)
        merged = merge_klines(raws)
        fxs = find_fractals(merged)
        strokes = find_strokes(merged, fxs, new_stroke=True)
        if strokes:
            ls = strokes[-1]
            out["末笔"] = "向上" if ls.direction == Direction.UP else "向下"
    except Exception:  # noqa: BLE001
        pass

    # 复用日线买点检测器（级别无关）
    recs = _detect_b123(
        code=code, name=name, cur_price=out["当前价"], hist=df,
        lookback_days=cfg["lookback"], target_types=target_types,
        min_score=min_score)
    out["可用"] = True
    cur = out["当前价"] or 0.0
    sigs = []
    for r in recs:
        sig_price = r["信号价"]
        dist = r["距今(交易日)"]
        fresh_ok = dist <= cfg["fresh"]
        # 当前价未跌破信号价（容差 _BREAK_TOL）才算买点仍有效
        price_ok = sig_price <= 0 or cur >= sig_price * (1 - _BREAK_TOL)
        valid = bool(fresh_ok and price_ok)
        if valid:
            reason = ""
        elif not fresh_ok:
            reason = f"过旧({dist}>{cfg['fresh']}根)"
        else:
            reason = f"已跌破信号价{sig_price}"
        sigs.append({
            "类型": r["信号类型"], "日期": str(r["信号日期"]),
            "距今": dist, "信号价": sig_price,
            "质量分": r["质量分"], "趋势评分": r["趋势评分"],
            "有效": valid, "失效原因": reason,
        })
    out["信号"] = sigs
    return out


# ══════════════════════════════════════════════════════════════════════
# 多级别共振
# ══════════════════════════════════════════════════════════════════════
def analyze_multi(code: str, name: str = "",
                  levels: list[str] | None = None,
                  target_types: set[str] | None = None,
                  min_score: int = 20) -> dict:
    levels = levels or ["daily", "60", "30"]
    target_types = target_types or {"B2", "B3"}
    result = {"代码": str(code).zfill(6), "名称": name, "级别": {}}
    for p in levels:
        result["级别"][p] = analyze_level(code, name, p, target_types, min_score)
    result["共振"] = _resonance(result["级别"])
    return result


def _resonance(levels: dict) -> dict:
    """
    共振判断：
      日线末笔向上（方向） + (60分 或 30分 有 B2/B3 信号)（短线入场）
      → '共振买点'；日线向上但分钟无信号 → '等待'；日线向下 → '逆势(观望)'
    """
    daily = levels.get("daily", {})
    m60 = levels.get("60", {})
    m30 = levels.get("30", {})

    def _has_valid(lv):
        return any(s.get("有效") for s in lv.get("信号", []))

    daily_up = daily.get("末笔") == "向上"
    m60_ok, m30_ok = _has_valid(m60), _has_valid(m30)
    short_sig = m60_ok or m30_ok
    sig_levels = []
    if m60_ok:
        sig_levels.append("60分")
    if m30_ok:
        sig_levels.append("30分")

    if daily_up and short_sig:
        verdict, tag = "共振买点", "✅"
    elif daily_up and not short_sig:
        verdict, tag = "日线向上·等分钟买点", "⏳"
    elif not daily_up and short_sig:
        verdict, tag = "分钟有信号·日线未转(博反弹)", "⚠️"
    else:
        verdict, tag = "逆势·观望", "❌"

    return {"结论": verdict, "标记": tag,
            "信号级别": sig_levels, "日线末笔": daily.get("末笔", "-")}


# ══════════════════════════════════════════════════════════════════════
# 命令行：单只多级别诊断
# ══════════════════════════════════════════════════════════════════════
def _print_report(res: dict):
    print(f"\n{'='*56}")
    print(f"  {res['名称']}（{res['代码']}）多级别缠论诊断")
    print(f"{'='*56}")
    for p in ["daily", "60", "30"]:
        lv = res["级别"].get(p)
        if not lv:
            continue
        if not lv["可用"]:
            print(f"\n【{lv['级别']}】数据不可用")
            continue
        print(f"\n【{lv['级别']}】现价 {lv['当前价']}  末笔{lv['末笔']}  "
              f"MACD柱 {lv['MACD柱']:+.4f}({lv['MACD趋势']})")
        if lv["信号"]:
            for s in lv["信号"]:
                tag = "✓有效" if s["有效"] else f"✗失效({s['失效原因']})"
                print(f"   {s['类型']} 信号  {s['日期']}  距今{s['距今']}根  "
                      f"信号价{s['信号价']}  质量{s['质量分']} 趋势{s['趋势评分']}  [{tag}]")
        else:
            print("   近期无 B2/B3 买点")

    r = res["共振"]
    print(f"\n{'─'*56}")
    print(f"  {r['标记']} 共振结论：{r['结论']}")
    if r["信号级别"]:
        print(f"     买点级别：{'、'.join(r['信号级别'])}")
    print(f"{'='*56}")


def main():
    if len(sys.argv) < 2:
        print("用法: python multi_level.py <代码> [名称]")
        sys.exit(1)
    code = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"=== 多级别缠论诊断 {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    res = analyze_multi(code, name)
    _print_report(res)


if __name__ == "__main__":
    main()
