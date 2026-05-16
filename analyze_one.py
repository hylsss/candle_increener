"""
analyze_one.py
────────────────────────────────────
对单只 A 股做完整的蜡烛图诊断，复用 full_scan.py 里的全部判定逻辑。

用法:
    python analyze_one.py 600732
    python analyze_one.py 600732 --period 2y
    python analyze_one.py 600732 --name 爱旭股份   # 跳过联网取名
"""

import argparse
import sys

import akshare as ak
import pandas as pd

from full_scan import (
    _add_indicators,
    _calc_stop_target,
    _candle,
    _detect_patterns,
    _detect_trend,
    _detect_windows,
    _fetch_history,
    _find_key_levels,
    _round,
    _scan_one,
)


def _lookup_name(code):
    """从实时行情里查股票名称，失败返回空串"""
    try:
        spot = ak.stock_zh_a_spot_em()
        row = spot.loc[spot["代码"].astype(str).str.zfill(6) == code]
        if not row.empty:
            return str(row.iloc[0]["名称"])
    except Exception as exc:
        print(f"⚠️  未能获取股票名称：{exc}", file=sys.stderr)
    return ""


def _print_report(result, hist):
    """把诊断结果排版打印出来"""
    bar = "═" * 60
    print(bar)
    print(f"  {result['代码']}  {result['名称']}    {result['信号']}")
    print(bar)

    last = hist.iloc[-1]
    last_date = str(last.get("日期", ""))
    print(f"  最新交易日 : {last_date}")
    print(f"  当前价     : {result['当前价']}")
    print(f"  趋势类型   : {result['趋势类型']}（评分 {result['趋势评分']}/100）")
    print(f"  ATR(14)    : {result['ATR']}")
    print()

    print("  ── 形态 ──────────────────────────────────────")
    print(f"  买入形态   : {result['买入形态']}")
    print(f"  卖出形态   : {result['卖出形态']}")
    print(f"  关键位置   : {result['关键位置']}")
    print(f"  窗口状态   : {result['窗口状态']}")
    print()

    print("  ── 三档买入价 ────────────────────────────────")
    print(f"  ①激进买入 : {result['①激进买入']}  （当前价直接介入）")
    print(f"  ②回调买入 : {result['②回调买入']}  （等回踩 MA20）")
    print(f"  ③突破买入 : {result['③突破买入']}  （站稳近 20 日高点）")
    print()

    print("  ── 风险控制 ──────────────────────────────────")
    print(f"  止损价     : {result['止损价']}   （距当前 {result['止损距离%']}%）")
    print(f"  目标价     : {result['目标价']}   （空间 {result['目标空间%']}%）")
    print(f"  风险收益比 : {result['风险收益比']}")
    print(bar)

    # 末尾再打印最近 8 根 K 线供肉眼复核
    print("\n  最近 8 个交易日 K 线：")
    tail_cols = [c for c in ["日期", "开盘", "最高", "最低", "收盘", "成交量"] if c in hist.columns]
    print(hist[tail_cols].tail(8).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="单只 A 股蜡烛图诊断")
    parser.add_argument("code", help="6 位股票代码，例如 600732")
    parser.add_argument("--name", default=None, help="股票名称（省略则联网查询）")
    parser.add_argument("--period", default="1y",
                        choices=["3mo", "6mo", "1y", "2y"],
                        help="历史区间，默认 1y")
    args = parser.parse_args()

    code = args.code.zfill(6)
    name = args.name if args.name is not None else _lookup_name(code)

    hist = _fetch_history(code, args.period)
    if hist.empty:
        print(f"❌ 没有取到 {code} 的历史数据", file=sys.stderr)
        sys.exit(1)
    if len(hist) < 60:
        print(f"❌ {code} 历史样本仅 {len(hist)} 根，至少需要 60 根", file=sys.stderr)
        sys.exit(1)

    result = _scan_one({"代码": code, "名称": name}, args.period)
    if result is None:
        print(f"❌ {code} 数据不足，无法生成诊断", file=sys.stderr)
        sys.exit(1)

    hist_with_ind = _add_indicators(hist)
    _print_report(result, hist_with_ind)


if __name__ == "__main__":
    main()
