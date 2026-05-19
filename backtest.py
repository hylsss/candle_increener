"""
backtest.py
────────────────────────────────────
简版策略回测：近 N 个交易日，每日跑 pick_strategies，
统计 4 策略在 5/10/20 日持仓周期下的胜率/均收/最大回撤。

设计要点：
- 对每只票预拉 2 年历史，全程只取数一次（耗时大头）
- 回测每个历史日 T：把每只票的 hist 切到 hist[:T]，重算指标和形态
- 选股后用同一票的完整 hist 取 T+N 日收盘作为出场价
- 全市场 2835 只 × 120 日太重，所以只采样 SAMPLE_SIZE 只（成交额排序前N）

参数（环境变量）：
  BT_SAMPLE_SIZE=500     回测覆盖前 N 只成交额最高的票
  BT_LOOKBACK_DAYS=120   往回追溯多少个交易日
  BT_HOLD_DAYS=5,10,20   持仓周期（逗号分隔）
  BT_WORKERS=8           历史拉取并发数
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd

import local_patch  # noqa: F401
import akshare as ak

from full_scan import (
    OUTPUT_COLUMNS, _add_indicators, _scan_one,
    pick_strategies, STRATEGY_NAMES,
    get_universe,
)


def backtest(sample_size: int = 500,
             lookback_days: int = 120,
             hold_days=(5, 10, 20),
             period: str = "2y",
             workers: int = 8) -> dict:
    """
    返回 {策略名: DataFrame}，每条记录是一次"信号→出场"完整交易。
    DataFrame 列：信号日期, 代码, 名称, 入场价, +5d收益%, +10d收益%, +20d收益%, ...
    """
    print(f"▶ 回测设置：样本 {sample_size} 只 × 回看 {lookback_days} 日 × 持仓 {hold_days}")

    # 1. 取股票池（前 sample_size 成交额）
    universe = get_universe(min_price=3.0, max_price=500.0,
                            min_turnover=0.5, main_board_only=True)
    universe = universe.sort_values("成交额_亿元", ascending=False).head(sample_size)
    universe = universe.reset_index(drop=True)
    print(f"📋 实际样本：{len(universe)} 只")

    # 2. 并发预拉每只票的完整历史（2 年）
    histories = _fetch_all_histories(universe, period=period, workers=workers)
    print(f"✅ 取到历史：{len(histories)} 只（{len(universe) - len(histories)} 只无数据已剔除）")

    if not histories:
        return {n: pd.DataFrame() for n in STRATEGY_NAMES.values()}

    # 3. 计算回测日期范围（拿最长票的最后 lookback_days 个交易日）
    all_dates = sorted({d for df in histories.values() for d in df["日期"].tolist()})
    if len(all_dates) <= max(hold_days) + 60:
        print("⚠️  历史天数不足，无法回测")
        return {n: pd.DataFrame() for n in STRATEGY_NAMES.values()}

    # 留出至少 max(hold_days) 天前瞻 + 60 天指标预热
    eligible = all_dates[60:-max(hold_days)]
    if len(eligible) > lookback_days:
        eligible = eligible[-lookback_days:]
    print(f"📅 回测区间：{eligible[0]} → {eligible[-1]}（{len(eligible)} 个交易日）")

    # 4. 对每个历史日重做 pick，记录信号
    all_trades = {n: [] for n in STRATEGY_NAMES.values()}
    name_map = dict(zip(universe["代码"].astype(str), universe["名称"].astype(str)))

    for i, T in enumerate(eligible, start=1):
        if i % 10 == 0 or i == 1:
            print(f"   [{i}/{len(eligible)}] 回测日 {T} ...")

        # 每只票切到 T 日为止，重算 _scan_one
        rows = []
        for code, full_hist in histories.items():
            hist_T = full_hist[full_hist["日期"] <= T]
            if len(hist_T) < 60:
                continue
            row = _scan_one_with_hist(code, name_map.get(code, ""), hist_T)
            if row:
                rows.append(row)

        if not rows:
            continue
        df_T = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        picks_T = pick_strategies(df_T)

        # 5. 每个策略的命中票 → 记录入场价 + 前瞻 N 日收益
        for strat_name, pdf in picks_T.items():
            for _, p in pdf.iterrows():
                code = str(p["代码"]).zfill(6)
                full_hist = histories.get(code)
                if full_hist is None:
                    continue
                fwd = _forward_returns(full_hist, T, hold_days)
                if not fwd:
                    continue
                all_trades[strat_name].append({
                    "信号日期": T,
                    "代码": code,
                    "名称": p["名称"],
                    "信号": p["信号"],
                    "趋势评分": p["趋势评分"],
                    "入场价": fwd["entry"],
                    **{f"+{n}d收益%": fwd[f"r{n}"] for n in hold_days},
                })

    return {n: pd.DataFrame(rows) for n, rows in all_trades.items()}


def summarize(trades_by_strat: dict, hold_days=(5, 10, 20)) -> pd.DataFrame:
    """生成汇总表：每策略 × 每持仓周期的胜率/均收/中位数/最大回撤"""
    rows = []
    for strat, df in trades_by_strat.items():
        if df.empty:
            rows.append({"策略": strat, "样本数": 0})
            continue
        rec = {"策略": strat, "样本数": len(df)}
        for n in hold_days:
            col = f"+{n}d收益%"
            if col not in df.columns:
                continue
            s = df[col].dropna()
            rec[f"{n}日胜率%"] = round((s > 0).mean() * 100, 1) if len(s) else 0
            rec[f"{n}日均收%"] = round(s.mean(), 2)
            rec[f"{n}日中位%"] = round(s.median(), 2)
            rec[f"{n}日最差%"] = round(s.min(), 2) if len(s) else 0
            rec[f"{n}日最好%"] = round(s.max(), 2) if len(s) else 0
        rows.append(rec)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────

def _fetch_all_histories(universe, period, workers):
    """并发预拉所有票的历史，返回 {code: DataFrame}"""
    days = {"3mo": 100, "6mo": 220, "1y": 400, "2y": 780}.get(period, 780)
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    end_date   = datetime.now().strftime("%Y%m%d")

    results = {}
    codes = list(universe["代码"].astype(str))

    def _one(code):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_date, end_date=end_date, adjust="qfq",
            )
            if df is None or df.empty:
                return code, None
            cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额"]
            df = df[[c for c in cols if c in df.columns]].copy()
            for c in ["开盘", "收盘", "最高", "最低", "成交量", "成交额"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["开盘", "收盘", "最高", "最低"]).reset_index(drop=True)
            df["日期"] = df["日期"].astype(str)
            return code, df
        except Exception as e:
            print(f"   ⚠️ {code} 取数失败：{e}")
            return code, None

    done = 0
    total = len(codes)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, c): c for c in codes}
        for f in as_completed(futs):
            code, df = f.result()
            done += 1
            if df is not None and len(df) >= 80:
                results[code] = df
            if done % 50 == 0:
                print(f"   ...预拉历史 {done}/{total}")
    return results


def _scan_one_with_hist(code, name, hist):
    """复用 full_scan._scan_one 的逻辑，但 hist 已经是切片到 T 日的版本。
    需要绕过 _fetch_history，直接走 _add_indicators + 后续判定。"""
    from full_scan import (
        _add_indicators, _detect_trend, _detect_patterns, _find_key_levels,
        _detect_windows, _calc_stop_target, _predict_buy_triggers,
        _determine_signal, _round, OUTPUT_COLUMNS,
    )

    if len(hist) < 60:
        return None
    hist = _add_indicators(hist)
    if len(hist) < 21:
        return None

    last, prev = hist.iloc[-1], hist.iloc[-2]
    close = float(last["收盘"])
    atr   = float(last["ATR"]) if not pd.isna(last["ATR"]) else 0.01

    trend_type, trend_score = _detect_trend(last, hist)
    buy_p, sell_p = _detect_patterns(hist)
    key_levels = _find_key_levels(last, hist)
    window_status = _detect_windows(hist)
    stop, target = _calc_stop_target(close, atr, hist, buy_p)

    risk   = max(close - stop, 0.01)
    reward = max(target - close, 0.01)
    rr     = reward / risk

    ma20   = float(last["MA20"])
    high20 = float(hist["最高"].tail(20).max())
    pullback_buy = ma20 if close > ma20 else None

    triggers = _predict_buy_triggers(close, hist, last, atr, trend_type)
    signal = _determine_signal(trend_type, trend_score, rr, buy_p, sell_p, close, last)

    return {
        "代码": str(code).zfill(6),
        "名称": name,
        "信号": signal,
        "当前价": _round(close),
        "趋势类型": trend_type,
        "趋势评分": int(trend_score),
        "买入形态": "、".join(buy_p) if buy_p else "无",
        "卖出形态": "、".join(sell_p) if sell_p else "无",
        "关键位置": "、".join(key_levels) if key_levels else "无",
        "窗口状态": window_status,
        "①激进买入": _round(close),
        "②回调买入": _round(pullback_buy) if pullback_buy is not None else "—",
        "③突破买入": _round(high20),
        "🚀突破触发价": triggers["breakout_price"],
        "📍均线触发价": triggers["ma_price"],
        "🌅抄底关注价": triggers["bottom_price"],
        "触发条件": triggers["desc"],
        "止损价": _round(stop),
        "止损距离%": _round((close - stop) / close * 100),
        "目标价": _round(target),
        "目标空间%": _round((target - close) / close * 100),
        "风险收益比": _round(rr),
        "ATR": _round(atr),
        # 缠论字段在原 backtest 框架里不计算（每日切片重跑成本高）；
        # 真正的缠论回测请用 chanlun_backtest.py
        "缠论信号": "",
        "缠论日期": "",
        "缠论价":   "",
        "扫描时间": "",
    }


def _forward_returns(full_hist, signal_date, hold_days):
    """计算入场后 N 日收益。入场价 = T+1 日开盘，T+N 日收盘出。"""
    idx_list = full_hist.index[full_hist["日期"] == signal_date].tolist()
    if not idx_list:
        return None
    T = idx_list[0]
    if T + 1 >= len(full_hist):
        return None
    entry = float(full_hist.iloc[T + 1]["开盘"])
    if entry <= 0:
        return None
    out = {"entry": round(entry, 2)}
    for n in hold_days:
        if T + n < len(full_hist):
            exit_p = float(full_hist.iloc[T + n]["收盘"])
            out[f"r{n}"] = round((exit_p - entry) / entry * 100, 2)
        else:
            out[f"r{n}"] = None
    return out


# ─────────────────────────────────────────────────────────

def main():
    sample_size   = int(os.environ.get("BT_SAMPLE_SIZE", 500))
    lookback_days = int(os.environ.get("BT_LOOKBACK_DAYS", 120))
    hold_days     = tuple(int(x) for x in os.environ.get("BT_HOLD_DAYS", "5,10,20").split(","))
    workers       = int(os.environ.get("BT_WORKERS", 8))

    print(f"=== 开始回测 {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    trades = backtest(
        sample_size=sample_size,
        lookback_days=lookback_days,
        hold_days=hold_days,
        workers=workers,
    )

    summary = summarize(trades, hold_days=hold_days)
    print("\n=== 📊 回测汇总 ===")
    print(summary.to_string(index=False))

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = f"回测_{ts}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="📊 汇总", index=False)
        for strat, df in trades.items():
            if not df.empty:
                df.to_excel(w, sheet_name=strat[:31], index=False)
    print(f"\n✅ 回测结果已保存：{out}")


if __name__ == "__main__":
    main()
