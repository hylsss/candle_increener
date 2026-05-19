"""
chanlun_viz.py
────────────────────────────────────
缠论可视化：把 K线 + 笔 + 线段 + 中枢 + 买卖点 + MACD 叠加成一张图，
用于人工复核算法识别是否合理。

用法：
  python chanlun_viz.py 600519
  python chanlun_viz.py 000001 --days 800 --out /tmp/000001.png
  python chanlun_viz.py 002594 --no-show          # 只存文件不弹窗
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg") if "--no-show" in sys.argv else None
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

import local_patch  # noqa: F401
import akshare as ak

from chanlun_core import (
    Direction, FractalType,
    from_dataframe, merge_klines, find_fractals, find_strokes,
    find_segments, find_pivots, detect_divergence, find_signals,
    calc_macd,
)


# 中文字体支持（macOS / Linux fallback）
plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Heiti TC", "Arial Unicode MS",
    "Noto Sans CJK SC", "WenQuanYi Zen Hei", "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


def _fetch(code: str, days: int) -> pd.DataFrame:
    """优先东财，失败回新浪。"""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        if df is not None and not df.empty:
            return _normalize(df).tail(days).reset_index(drop=True)
    except Exception as e:
        print(f"  东财失败({e})，回退新浪...")

    prefix = "sh" if code.startswith(("6", "9")) else "sz"
    df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
    df = df.tail(days).reset_index(drop=True)
    df = df.rename(columns={"date": "日期", "open": "开盘", "high": "最高",
                            "low": "最低", "close": "收盘", "volume": "成交量"})
    return _normalize(df)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名 + 类型。"""
    for c in ["开盘", "收盘", "最高", "最低", "成交量"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["开盘", "收盘", "最高", "最低"]).reset_index(drop=True)
    df["日期"] = df["日期"].astype(str)
    return df


def draw(code: str, days: int = 500, out: str | None = None, show: bool = True,
         require_zero_axis: bool = False) -> None:
    print(f"获取 {code} 日线数据 ({days} 日)...")
    df = _fetch(code, days)
    if df.empty:
        print(f"❌ {code}: 无数据")
        return

    raws = from_dataframe(df)
    merged = merge_klines(raws)
    fxs = find_fractals(merged)
    strokes = find_strokes(merged, fxs, new_stroke=True)
    segments = find_segments(strokes)
    pivots = find_pivots(segments)
    divs = detect_divergence(segments, merged, raws, require_zero_axis=require_zero_axis)
    signals = find_signals(segments, pivots, divs)
    macd = calc_macd([r.close for r in raws])

    print(f"  原始K {len(raws)} 合并K {len(merged)} 分型 {len(fxs)} "
          f"笔 {len(strokes)} 段 {len(segments)} 中枢 {len(pivots)} "
          f"背驰 {len(divs)} 信号 {len(signals)}")

    fig, (ax_k, ax_macd) = plt.subplots(
        2, 1, figsize=(18, 10),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    x = list(range(len(df)))
    dates = df["日期"].tolist()

    # ── K线 ──
    for i, row in df.iterrows():
        o, c, h, l = row["开盘"], row["收盘"], row["最高"], row["最低"]
        color = "red" if c >= o else "green"
        ax_k.plot([i, i], [l, h], color="black", linewidth=0.4, alpha=0.5)
        ax_k.add_patch(Rectangle(
            (i - 0.3, min(o, c)), 0.6, max(abs(c - o), 0.001),
            facecolor=color, edgecolor=color, linewidth=0.3, alpha=0.8,
        ))

    # ── 笔（细虚线灰色）──
    for s in strokes:
        x1 = merged[s.start_fx.mid_idx].raw_indices[0]
        x2 = merged[s.end_fx.mid_idx].raw_indices[-1]
        ax_k.plot([x1, x2], [s.start_price, s.end_price],
                  color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    # ── 线段（粗实线，红涨绿跌）──
    for seg in segments:
        x1 = merged[seg.start_fx.mid_idx].raw_indices[0]
        x2 = merged[seg.end_fx.mid_idx].raw_indices[-1]
        col = "#d62728" if seg.direction == Direction.UP else "#2ca02c"
        ax_k.plot([x1, x2], [seg.start_price, seg.end_price],
                  color=col, linewidth=2.2, alpha=0.85)

    # ── 中枢（黄色填充矩形 + 边框）──
    for p in pivots:
        x1 = merged[p.segments[0].strokes[0].start_fx.mid_idx].raw_indices[0]
        x2 = merged[p.segments[-1].strokes[-1].end_fx.mid_idx].raw_indices[-1]
        ax_k.add_patch(Rectangle(
            (x1, p.zd), max(x2 - x1, 1), p.zg - p.zd,
            facecolor="#ffe066", edgecolor="#f59f00",
            linewidth=1.2, alpha=0.30,
        ))
        ax_k.text(x1, p.zg, f" ZG={p.zg:.2f}", fontsize=7,
                  color="#7a5b00", va="bottom")
        ax_k.text(x1, p.zd, f" ZD={p.zd:.2f}", fontsize=7,
                  color="#7a5b00", va="top")

    # ── 买卖点（▲买/▼卖）──
    date_to_x = {d: i for i, d in enumerate(dates)}
    for sig in signals:
        x_idx = date_to_x.get(str(sig.dt))
        if x_idx is None:
            continue
        if sig.is_buy:
            ax_k.scatter([x_idx], [sig.price], marker="^", s=200,
                         color="#0066cc", edgecolors="black", linewidths=0.8, zorder=5)
            ax_k.annotate(sig.signal_type, (x_idx, sig.price),
                          xytext=(0, -18), textcoords="offset points",
                          ha="center", fontsize=8, fontweight="bold",
                          color="#0066cc")
        else:
            ax_k.scatter([x_idx], [sig.price], marker="v", s=200,
                         color="#cc0066", edgecolors="black", linewidths=0.8, zorder=5)
            ax_k.annotate(sig.signal_type, (x_idx, sig.price),
                          xytext=(0, 10), textcoords="offset points",
                          ha="center", fontsize=8, fontweight="bold",
                          color="#cc0066")

    # ── MACD 子图 ──
    diffs = [m.diff for m in macd]
    deas = [m.dea for m in macd]
    bars = [m.bar for m in macd]
    bar_colors = ["#d62728" if b > 0 else "#2ca02c" for b in bars]
    ax_macd.bar(x, bars, color=bar_colors, alpha=0.7, width=0.85)
    ax_macd.plot(x, diffs, color="black", linewidth=0.8, label="DIFF")
    ax_macd.plot(x, deas, color="orange", linewidth=0.8, label="DEA")
    ax_macd.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_macd.legend(loc="upper left", fontsize=8)
    ax_macd.set_ylabel("MACD", fontsize=9)

    # ── 标题与坐标 ──
    title = (f"{code}  缠论分析  {dates[0]} → {dates[-1]}  "
             f"|  段 {len(segments)}  中枢 {len(pivots)}  "
             f"背驰 {len(divs)}  信号 {len(signals)}")
    ax_k.set_title(title, fontsize=12, fontweight="bold")
    ax_k.set_ylabel("价格", fontsize=9)
    ax_k.grid(True, alpha=0.25, linestyle="--")
    ax_macd.grid(True, alpha=0.25, linestyle="--")

    # X 轴日期刻度
    step = max(1, len(x) // 12)
    ax_macd.set_xticks(x[::step])
    ax_macd.set_xticklabels(dates[::step], rotation=30, ha="right", fontsize=8)
    ax_macd.set_xlim(-1, len(x))

    plt.tight_layout()

    if out:
        plt.savefig(out, dpi=130, bbox_inches="tight")
        print(f"✅ 保存: {out}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="缠论可视化")
    ap.add_argument("code", help="股票代码，如 600519")
    ap.add_argument("--days", type=int, default=500, help="回看交易日 (default 500)")
    ap.add_argument("--out", default=None, help="保存路径，如 /tmp/xx.png")
    ap.add_argument("--no-show", action="store_true", help="只保存不弹窗")
    ap.add_argument("--zero", action="store_true", help="背驰严格 0 轴判定")
    args = ap.parse_args()

    out = args.out
    if out is None and args.no_show:
        out = f"chanlun_{args.code}.png"

    draw(code=args.code.zfill(6), days=args.days, out=out,
         show=not args.no_show, require_zero_axis=args.zero)


if __name__ == "__main__":
    main()
