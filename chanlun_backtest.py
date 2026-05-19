"""
chanlun_backtest.py
────────────────────────────────────
缠论 1/2/3 类买卖点回测：评估 B1/B2/B3/S1/S2/S3 信号的前瞻收益。

⚠️ Snapshot 模式：pipeline 一次跑完全历史，存在轻微"未来函数"——
   线段/中枢的回溯确认会让历史信号在算法里比真实可见的早几根。
   结果偏乐观。严格 walk-forward 需要按日重跑（约慢 100x），后续按需添加。

用法：
  python chanlun_backtest.py                       # 默认 100 只 × ~3 年
  CB_SAMPLE=300 CB_DAYS=1500 python chanlun_backtest.py
环境变量：
  CB_SAMPLE   覆盖前 N 只成交额最高的票  (default 100)
  CB_DAYS     拉取多少日历日的历史        (default 1500)
  CB_WORKERS  并发取数线程                (default 1; mini_racer 在多线程下会崩)
  CB_HOLD     前瞻持仓周期，逗号分隔      (default 5,10,20)
  CB_ZERO     1=严格 0 轴判定背驰；0=宽松 (default 0)
  CB_CODES    直接指定股票代码 (逗号分隔)，跳过 spot 取池
              例：CB_CODES=600519,000001,002594
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd

import local_patch  # noqa: F401
import akshare as ak

from full_scan import get_universe
from chanlun_core import (
    Direction,
    from_dataframe, merge_klines, find_fractals, find_strokes,
    find_segments, find_pivots, detect_divergence, find_signals,
)

SIG_TYPES = ["B1", "B2", "B3", "S1", "S2", "S3"]


def chanlun_backtest(sample_size: int = 100,
                     period_days: int = 1500,
                     hold_days=(5, 10, 20),
                     workers: int = 8,
                     require_zero_axis: bool = False,
                     codes: list[str] | None = None) -> pd.DataFrame:
    """
    返回长表 DataFrame，每行一个 (股票, 信号) 记录，含 +5d/+10d/+20d 收益。
    codes 不为空时跳过 get_universe，直接用指定代码列表。
    """
    print(f"▶ 缠论回测 (snapshot): 历史 {period_days} 日")
    if require_zero_axis:
        print("  背驰判定: 严格 0 轴模式")

    if codes:
        codes = [str(c).zfill(6) for c in codes]
        name_map = {c: "" for c in codes}
        print(f"📋 指定股票: {len(codes)} 只")
    else:
        uni = get_universe(min_price=3.0, max_price=500.0,
                           min_turnover=0.5, main_board_only=True)
        uni = uni.sort_values("成交额_亿元", ascending=False).head(sample_size)
        uni = uni.reset_index(drop=True)
        name_map = dict(zip(uni["代码"].astype(str), uni["名称"].astype(str)))
        codes = list(uni["代码"].astype(str))
        print(f"📋 股票池: {len(codes)} 只 (按成交额取头部)")

    histories = _fetch_histories(codes, period_days, workers)
    print(f"✅ 历史数据: {len(histories)} 只 (剔除 {len(codes) - len(histories)} 无数据)")

    if not histories:
        return pd.DataFrame()

    all_trades: list[dict] = []
    no_signal_count = 0
    for code, hist in histories.items():
        trades = _evaluate_one(code, name_map.get(code, ""), hist,
                               hold_days, require_zero_axis)
        if trades:
            all_trades.extend(trades)
        else:
            no_signal_count += 1

    print(f"📊 出信号 {len(histories) - no_signal_count} 只；无信号 {no_signal_count} 只")
    return pd.DataFrame(all_trades)


def chanlun_summary(trades_df: pd.DataFrame, hold_days=(5, 10, 20)) -> pd.DataFrame:
    """
    按 signal_type 分组的胜率/均收/中位数/最差/最好 汇总。

    收益统一用"买入持有"语义计算（exit - entry）/entry。
      - B1/B2/B3 (买点)：正收益 = 胜
      - S1/S2/S3 (卖点)：S 的本质是"应避免/应离场"，回测里用反向收益：
                          胜率 = 信号后下跌的比率（即 raw 收益为负的比率）
    输出 "胜率" 列已对 S 类做了符号翻转，便于横向比较。
    """
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for sig in SIG_TYPES:
        sub = trades_df[trades_df["信号类型"] == sig]
        if sub.empty:
            rows.append({"信号": sig, "样本数": 0})
            continue
        rec = {"信号": sig, "样本数": len(sub)}
        for n in hold_days:
            col = f"+{n}d收益%"
            if col not in sub.columns:
                continue
            s = sub[col].dropna()
            if not len(s):
                continue
            # 卖点视角：S 类的"胜"是下跌
            wins = (s < 0) if sig.startswith("S") else (s > 0)
            rec[f"{n}d胜率%"] = round(wins.mean() * 100, 1)
            rec[f"{n}d均收%"] = round(s.mean(), 2)
            rec[f"{n}d中位%"] = round(s.median(), 2)
            rec[f"{n}d最差%"] = round(s.min(), 2)
            rec[f"{n}d最好%"] = round(s.max(), 2)
        rows.append(rec)
    return pd.DataFrame(rows)


# ── 内部 ──────────────────────────────────────────────────────────────

def _evaluate_one(code: str, name: str, hist: pd.DataFrame,
                  hold_days, require_zero_axis: bool) -> list[dict]:
    """对单只股票跑完整 pipeline，收集所有信号 + 前瞻收益。"""
    if len(hist) < 100:
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
        divs = detect_divergence(segments, merged, raws,
                                 require_zero_axis=require_zero_axis)
        signals = find_signals(segments, pivots, divs)
    except Exception as e:
        print(f"   ⚠️ {code} pipeline 异常: {e}")
        return []

    # 日期 → 行号映射
    date_to_idx = {str(d): i for i, d in enumerate(hist["日期"])}

    trades = []
    for sig in signals:
        idx = date_to_idx.get(str(sig.dt))
        if idx is None or idx + 1 >= len(hist):
            continue
        entry = float(hist.iloc[idx + 1]["开盘"])
        if entry <= 0:
            continue

        rec = {
            "代码": code,
            "名称": name,
            "信号类型": sig.signal_type,
            "买卖": "买" if sig.is_buy else "卖",
            "信号日期": sig.dt,
            "信号价": round(sig.price, 2),
            "入场价": round(entry, 2),
            "备注": sig.note,
        }
        for n in hold_days:
            tgt = idx + 1 + n
            if tgt < len(hist):
                exit_p = float(hist.iloc[tgt]["收盘"])
                rec[f"+{n}d收益%"] = round((exit_p - entry) / entry * 100, 2)
            else:
                rec[f"+{n}d收益%"] = None
        trades.append(rec)

    return trades


def _fetch_histories(codes, period_days, workers):
    """并发预拉。日历日 → 取数窗口。"""
    start_date = (datetime.now() - timedelta(days=int(period_days * 1.5))).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")

    def _one(code):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_date, end_date=end_date, adjust="qfq",
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
            return code, df
        except Exception:
            return code, None

    results = {}
    done = 0
    total = len(codes)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, c): c for c in codes}
        for f in as_completed(futs):
            code, df = f.result()
            done += 1
            if df is not None and len(df) >= 100:
                results[code] = df
            if done % 25 == 0 or done == total:
                print(f"   ...预拉历史 {done}/{total}")
    return results


def main():
    sample_size = int(os.environ.get("CB_SAMPLE", 100))
    period_days = int(os.environ.get("CB_DAYS", 1500))
    workers = int(os.environ.get("CB_WORKERS", 1))
    hold_days = tuple(int(x) for x in os.environ.get("CB_HOLD", "5,10,20").split(","))
    require_zero_axis = os.environ.get("CB_ZERO", "0") == "1"
    codes_env = os.environ.get("CB_CODES", "").strip()
    codes = [c.strip() for c in codes_env.split(",") if c.strip()] if codes_env else None

    print(f"=== 缠论回测 {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    trades = chanlun_backtest(
        sample_size=sample_size, period_days=period_days,
        hold_days=hold_days, workers=workers,
        require_zero_axis=require_zero_axis,
        codes=codes,
    )

    if trades.empty:
        print("⚠️ 无信号产出。")
        sys.exit(0)

    print(f"\n=== 信号分布 (共 {len(trades)} 个) ===")
    print(trades["信号类型"].value_counts().sort_index().to_string())

    summary = chanlun_summary(trades, hold_days)
    print(f"\n=== 📊 缠论回测汇总 (snapshot 模式) ===")
    print("注意: B 类胜率 = 正收益占比; S 类胜率 = 负收益占比 (信号后下跌为胜)")
    print(summary.to_string(index=False))

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = f"缠论回测_{ts}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="📊 汇总", index=False)
        trades.to_excel(w, sheet_name="全部信号", index=False)
        for sig in SIG_TYPES:
            sub = trades[trades["信号类型"] == sig]
            if not sub.empty:
                sub.to_excel(w, sheet_name=sig, index=False)
    print(f"\n✅ 保存: {out}")


if __name__ == "__main__":
    main()
