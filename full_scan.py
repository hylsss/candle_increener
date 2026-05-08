"""
full_scan.py
────────────────────────────────────
A 股全市场蜡烛图扫描核心逻辑。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd


OUTPUT_COLUMNS = [
    "代码",
    "名称",
    "信号",
    "当前价",
    "趋势评分",
    "风险收益比",
    "①激进买入",
    "②回调买入",
    "③突破买入",
    "止损价",
    "目标价",
    "ATR",
    "主要形态",
    "扫描时间",
]


def get_universe(min_price=3.0, max_price=500.0, min_turnover=0.3):
    """
    获取 A 股股票池。

    min_turnover 单位按「亿元」理解，避免把极低成交额标的纳入扫描。
    """
    spot = ak.stock_zh_a_spot_em()
    required = {"代码", "名称", "最新价", "成交额"}
    missing = required - set(spot.columns)
    if missing:
        raise ValueError(f"行情字段缺失：{', '.join(sorted(missing))}")

    df = spot.copy()
    df["当前价"] = pd.to_numeric(df["最新价"], errors="coerce")
    df["成交额_亿元"] = pd.to_numeric(df["成交额"], errors="coerce") / 100_000_000

    mask = (
        df["当前价"].between(min_price, max_price)
        & (df["成交额_亿元"] >= min_turnover)
        & ~df["名称"].astype(str).str.contains("ST", case=False, na=False)
    )
    return df.loc[mask, ["代码", "名称", "当前价", "成交额_亿元"]].reset_index(drop=True)


def run_full_scan(universe, period="6mo", workers=10):
    """并发扫描股票池，返回按信号和评分排序后的结果。"""
    if universe.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows = []
    max_workers = max(1, int(workers or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_scan_one, row.to_dict(), period): row["代码"]
            for _, row in universe.iterrows()
        }
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:
                print(f"⚠️  {futures[future]} 扫描失败：{exc}")
                continue
            if row:
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    signal_rank = {
        "🟢 买入观察": 0,
        "🟡 关注候选": 1,
        "⚪ 无明显信号": 2,
        "🔴 回避/止盈": 3,
    }
    df["_rank"] = df["信号"].map(signal_rank).fillna(9)
    df = df.sort_values(["_rank", "趋势评分", "风险收益比"], ascending=[True, False, False])
    return df.drop(columns="_rank").reset_index(drop=True)


def save_excel(df, path):
    """保存扫描结果到 Excel。"""
    df.to_excel(path, index=False)
    print(f"✅ Excel 已保存：{path}")


def _scan_one(stock, period):
    code = str(stock["代码"]).zfill(6)
    name = str(stock["名称"])
    hist = _fetch_history(code, period)
    if len(hist) < 60:
        return None

    hist = _add_indicators(hist)
    if len(hist) < 21:
        return None

    last = hist.iloc[-1]
    prev = hist.iloc[-2]

    close = float(last["收盘"])
    atr = float(last["ATR"])
    ma20 = float(last["MA20"])
    ma60 = float(last["MA60"])
    high20 = float(hist["最高"].tail(20).max())
    low20 = float(hist["最低"].tail(20).min())

    patterns = _detect_patterns(hist)
    trend_score = _trend_score(last, prev, hist)
    stop = max(0.01, min(low20, close - 1.5 * atr))
    target = max(high20, close + 2.0 * atr)
    rr = (target - close) / max(close - stop, 0.01)

    breakout_buy = high20 if close < high20 else close
    pullback_buy = ma20 if close >= ma20 else close
    aggressive_buy = close

    signal = _signal(close, ma20, ma60, trend_score, rr, patterns)
    return {
        "代码": code,
        "名称": name,
        "信号": signal,
        "当前价": _round(close),
        "趋势评分": int(trend_score),
        "风险收益比": _round(rr),
        "①激进买入": _round(aggressive_buy),
        "②回调买入": _round(pullback_buy),
        "③突破买入": _round(breakout_buy),
        "止损价": _round(stop),
        "目标价": _round(target),
        "ATR": _round(atr),
        "主要形态": "、".join(patterns) if patterns else "无明显形态",
        "扫描时间": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def _fetch_history(code, period):
    days = _period_days(period)
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    if df.empty:
        return df

    cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额"]
    df = df[[c for c in cols if c in df.columns]].copy()
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["开盘", "收盘", "最高", "最低"]).reset_index(drop=True)


def _add_indicators(df):
    out = df.copy()
    out["MA5"] = out["收盘"].rolling(5).mean()
    out["MA20"] = out["收盘"].rolling(20).mean()
    out["MA60"] = out["收盘"].rolling(60).mean()
    out["VOL20"] = out["成交量"].rolling(20).mean()

    prev_close = out["收盘"].shift(1)
    tr = pd.concat(
        [
            out["最高"] - out["最低"],
            (out["最高"] - prev_close).abs(),
            (out["最低"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()
    return out.dropna().reset_index(drop=True)


def _detect_patterns(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    patterns = []

    body = abs(last["收盘"] - last["开盘"])
    candle_range = max(last["最高"] - last["最低"], 0.01)
    lower_shadow = min(last["开盘"], last["收盘"]) - last["最低"]
    upper_shadow = last["最高"] - max(last["开盘"], last["收盘"])

    if last["收盘"] > last["开盘"] and prev["收盘"] < prev["开盘"]:
        if last["收盘"] >= prev["开盘"] and last["开盘"] <= prev["收盘"]:
            patterns.append("看涨吞没")

    if lower_shadow >= body * 2 and upper_shadow <= candle_range * 0.25:
        patterns.append("锤子线")

    prev_high20 = df["最高"].iloc[-21:-1].max()
    if last["收盘"] > prev_high20 and last["成交量"] > last["VOL20"] * 1.2:
        patterns.append("放量突破")

    if last["收盘"] > last["MA20"] and prev["收盘"] <= prev["MA20"]:
        patterns.append("站上20日线")

    return patterns


def _trend_score(last, prev, df):
    score = 0
    if last["收盘"] > last["MA20"]:
        score += 25
    if last["MA20"] > last["MA60"]:
        score += 25
    if last["收盘"] > prev["收盘"]:
        score += 15
    if last["成交量"] > last["VOL20"]:
        score += 15
    if last["收盘"] >= df["收盘"].tail(20).quantile(0.75):
        score += 20
    return min(score, 100)


def _signal(close, ma20, ma60, trend_score, rr, patterns):
    bullish_pattern = any(p in patterns for p in ["看涨吞没", "锤子线", "放量突破", "站上20日线"])
    if trend_score >= 70 and rr >= 1.3 and bullish_pattern:
        return "🟢 买入观察"
    if trend_score >= 55 and close >= ma20 and rr >= 1.0:
        return "🟡 关注候选"
    if trend_score <= 35 and close < ma60:
        return "🔴 回避/止盈"
    return "⚪ 无明显信号"


def _period_days(period):
    mapping = {
        "3mo": 100,
        "6mo": 220,
        "1y": 370,
        "2y": 740,
    }
    return mapping.get(str(period).lower(), 220)


def _round(value):
    if pd.isna(value):
        return ""
    return round(float(value), 2)
