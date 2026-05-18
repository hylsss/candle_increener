"""
烟雾测试：用真实 A 股日线数据跑一遍缠论引擎，肉眼检查输出是否合理。
非单元测试，手动运行：
    python tests/smoke_real_stock.py 600519
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import akshare as ak

from chanlun_core import (
    Direction, FractalType,
    from_dataframe, merge_klines, find_fractals, find_strokes,
)


def fetch_daily(code: str, days: int = 400):
    """优先东财，失败回退新浪。"""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start, end_date=end, adjust="qfq",
        )
        if not df.empty:
            return df
    except Exception as e:
        print(f"  东财失败({e})，回退新浪...")

    # 新浪源：需要 sh/sz 前缀
    prefix = "sh" if code.startswith(("6", "9")) else "sz"
    df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
    # 新浪源列名为英文：date/open/high/low/close/volume
    df = df.tail(days).reset_index(drop=True)
    # 统一列名为中文（适配 from_dataframe 默认）
    df = df.rename(columns={
        "date": "日期", "open": "开盘", "high": "最高",
        "low": "最低", "close": "收盘", "volume": "成交量",
    })
    return df


def main(code: str = "600519"):
    print(f"获取 {code} 日线数据...")
    df = fetch_daily(code)
    if df.empty:
        print(f"  ❌ 无数据")
        return

    raws = from_dataframe(df)
    print(f"  原始K线 {len(raws)} 根：{raws[0].dt} → {raws[-1].dt}")

    merged = merge_klines(raws)
    print(f"  合并K线 {len(merged)} 根（压缩比 {len(merged)/len(raws):.2%}）")

    fxs = find_fractals(merged)
    tops = [f for f in fxs if f.ftype == FractalType.TOP]
    bots = [f for f in fxs if f.ftype == FractalType.BOTTOM]
    print(f"  分型 {len(fxs)}（顶 {len(tops)} / 底 {len(bots)}）")

    for tag, count in [("新笔", True), ("老笔", False)]:
        strokes = find_strokes(merged, fxs, new_stroke=count)
        ups = sum(1 for s in strokes if s.direction == Direction.UP)
        downs = sum(1 for s in strokes if s.direction == Direction.DOWN)
        print(f"  {tag} {len(strokes)}（上 {ups} / 下 {downs}）")
        # 打印最近 5 笔
        for s in strokes[-5:]:
            arrow = "↑" if s.direction == Direction.UP else "↓"
            print(f"    {arrow} {s.start_fx.dt} {s.start_price:.2f}"
                  f" → {s.end_fx.dt} {s.end_price:.2f}"
                  f"（振幅 {abs(s.end_price - s.start_price):.2f}）")


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "600519"
    main(code)
