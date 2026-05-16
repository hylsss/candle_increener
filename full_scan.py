"""
full_scan.py
────────────────────────────────────
A 股全市场蜡烛图扫描核心逻辑
基于尼森（Nison）蜡烛图理论：
  好机会 = 趋势清楚 + 关键位置 + 蜡烛信号 + 风险收益划算 + 有止损纪律
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd


OUTPUT_COLUMNS = [
    "代码", "名称", "信号",
    "当前价", "趋势类型", "趋势评分",
    "买入形态", "卖出形态", "关键位置", "窗口状态",
    "①激进买入", "②回调买入", "③突破买入",
    "🚀突破触发价", "📍均线触发价", "🌅抄底关注价", "触发条件",
    "止损价", "止损距离%", "目标价", "目标空间%", "风险收益比",
    "ATR", "扫描时间",
]


def get_universe(min_price=3.0, max_price=500.0, min_turnover=0.3,
                 main_board_only=True):
    """
    获取 A 股股票池。用新浪源（本机东财不可达）。
    main_board_only=True 时只保留沪深主板（剔除 300/301 创业板、688 科创板、
    8xx/4xx 北交所、9xx B 股）。同时过滤 ST 和极低流动性标的。
    """
    spot = ak.stock_zh_a_spot()
    required = {"代码", "名称", "最新价", "成交额"}
    missing = required - set(spot.columns)
    if missing:
        raise ValueError(f"行情字段缺失：{', '.join(sorted(missing))}")

    df = spot.copy()
    # 新浪源代码自带交易所前缀（sh600519 / sz000001 / bj920000），剥成纯 6 位
    df["代码"] = df["代码"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True)

    df["当前价"] = pd.to_numeric(df["最新价"], errors="coerce")
    df["成交额_亿元"] = pd.to_numeric(df["成交额"], errors="coerce") / 100_000_000

    mask = (
        df["当前价"].between(min_price, max_price)
        & (df["成交额_亿元"] >= min_turnover)
        & ~df["名称"].astype(str).str.contains("ST", case=False, na=False)
    )

    if main_board_only:
        # 保留：60x（沪主板）、00x（深主板/中小板）。其它一律剔除。
        mask &= df["代码"].str.match(r"^(60[0135]|00[0123])")

    return df.loc[mask, ["代码", "名称", "当前价", "成交额_亿元"]].reset_index(drop=True)


def run_full_scan(universe, period="1y", workers=10):
    """并发扫描股票池，返回按信号和评分排序后的结果"""
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
        "🔴 卖出/止盈": 3,
    }
    df["_rank"] = df["信号"].map(signal_rank).fillna(9)
    df = df.sort_values(["_rank", "趋势评分", "风险收益比"], ascending=[True, False, False])
    return df.drop(columns="_rank").reset_index(drop=True)


STRATEGY_NAMES = {
    "breakout": "🚀 突破追涨",
    "pullback": "🔄 强势回踩",
    "reversal": "🌅 底部反转",
    "high_rr":  "📈 高RR精选",
}


def pick_strategies(df, max_per_strategy=30):
    """
    按 4 个策略从扫描结果里选股。返回 dict[策略名 -> DataFrame]。

    🚀 突破追涨：当前价距 🚀突破触发价 ≤ 3% + 趋势 ≥ 60
    🔄 强势回踩：上升趋势 ≥ 65 + 当前价距 MA20 ≤ 3% + 出现下影线/吞噬类买入信号
    🌅 底部反转：趋势=底部转强 + 出现 启明星/锤子/看涨吞噬/刺穿 之一
    📈 高RR精选：信号=🟢 买入观察 + R:R ≥ 2.5 + 趋势 ≥ 60
    """
    empty = df.iloc[:0].copy()
    if df.empty:
        return {n: empty for n in STRATEGY_NAMES.values()}

    d = df.copy()
    d["_price"] = pd.to_numeric(d["当前价"], errors="coerce")
    d["_trend"] = pd.to_numeric(d["趋势评分"], errors="coerce")
    d["_rr"]    = pd.to_numeric(d["风险收益比"], errors="coerce")
    d["_bo"]    = pd.to_numeric(d["🚀突破触发价"], errors="coerce")
    d["_ma"]    = pd.to_numeric(d["📍均线触发价"], errors="coerce")
    d["_buy"]   = d["买入形态"].fillna("")

    drop_aux = ["_price", "_trend", "_rr", "_bo", "_ma", "_buy", "距触发%", "距MA20%"]

    # 🚀 突破追涨（只要 🟢 买入观察）
    bo_gap = (d["_bo"] - d["_price"]) / d["_price"]
    bo = d[(d["信号"] == "🟢 买入观察")
           & (d["_bo"] > 0) & (bo_gap <= 0.03) & (bo_gap >= -0.005)
           & (d["_trend"] >= 60)].copy()
    if not bo.empty:
        bo["距触发%"] = ((bo["_bo"] - bo["_price"]) / bo["_price"] * 100).round(2)
        bo = bo.sort_values(["距触发%", "_trend"], ascending=[True, False])

    # 🔄 强势回踩
    pb_pats = ("锤子线", "看涨吞噬", "十字星确认", "启明星", "刺穿形态", "倒锤子")
    pb_mask = d["_buy"].apply(lambda s: any(p in s for p in pb_pats))
    ma_gap = (d["_price"] - d["_ma"]).abs() / d["_ma"]
    pb = d[(d["趋势类型"] == "上升趋势") & (d["_trend"] >= 65)
           & (d["_ma"] > 0) & (ma_gap <= 0.03) & pb_mask].copy()
    if not pb.empty:
        pb["距MA20%"] = ((pb["_price"] - pb["_ma"]) / pb["_ma"] * 100).round(2)
        pb = pb.sort_values(["_trend", "_rr"], ascending=[False, False])

    # 🌅 底部反转（趋势 ≥ 50 + 至少 🟡 关注候选，排除 ⚪ 弱信号诱多）
    rv_pats = ("启明星", "锤子线", "看涨吞噬", "倒锤子", "刺穿形态")
    rv_mask = d["_buy"].apply(lambda s: any(p in s for p in rv_pats))
    rv = d[(d["趋势类型"] == "底部转强") & (d["_trend"] >= 50)
           & (d["信号"] != "⚪ 无明显信号") & rv_mask].copy()
    if not rv.empty:
        rv = rv.sort_values(["_rr", "_trend"], ascending=[False, False])

    # 📈 高 RR 精选
    hr = d[(d["信号"] == "🟢 买入观察") & (d["_rr"] >= 2.5) & (d["_trend"] >= 60)].copy()
    if not hr.empty:
        hr = hr.sort_values("_rr", ascending=False)

    def _clean(picked):
        return picked.head(max_per_strategy).drop(
            columns=[c for c in drop_aux if c in picked.columns],
            errors="ignore",
        ).reset_index(drop=True)

    return {
        STRATEGY_NAMES["breakout"]: _clean(bo),
        STRATEGY_NAMES["pullback"]: _clean(pb),
        STRATEGY_NAMES["reversal"]: _clean(rv),
        STRATEGY_NAMES["high_rr"]:  _clean(hr),
    }


def save_excel(df, path, picks=None):
    """保存扫描结果到 Excel：主表 + 各策略 sheet"""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="📊 全部扫描结果"[:31], index=False)
        if picks:
            for name, pick_df in picks.items():
                pick_df.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"✅ Excel 已保存：{path}（{len(df)} 只主表）")
    if picks:
        for name, pick_df in picks.items():
            print(f"   {name}: {len(pick_df)} 只")


# ══════════════════════════════════════════════════════════════════════
# 内部实现
# ══════════════════════════════════════════════════════════════════════

def _scan_one(stock, period):
    code = str(stock["代码"]).zfill(6)
    name = str(stock["名称"])
    hist = _fetch_history(code, period)
    if len(hist) < 60:
        return None

    hist = _add_indicators(hist)
    if len(hist) < 21:
        return None

    last  = hist.iloc[-1]
    prev  = hist.iloc[-2]
    close = float(last["收盘"])
    atr   = float(last["ATR"]) if not pd.isna(last["ATR"]) else 0.01

    trend_type, trend_score = _detect_trend(last, hist)
    buy_patterns, sell_patterns = _detect_patterns(hist)
    key_levels  = _find_key_levels(last, hist)
    window_status = _detect_windows(hist)
    stop, target  = _calc_stop_target(close, atr, hist, buy_patterns)

    risk   = max(close - stop, 0.01)
    reward = max(target - close, 0.01)
    rr     = reward / risk

    ma20    = float(last["MA20"])
    high20  = float(hist["最高"].tail(20).max())
    pullback_buy = ma20 if close > ma20 else None    # 跌破 MA20 时回调买入无意义
    breakout_buy = high20

    triggers = _predict_buy_triggers(close, hist, last, atr, trend_type)

    signal = _determine_signal(
        trend_type, trend_score, rr, buy_patterns, sell_patterns, close, last
    )

    return {
        "代码":      code,
        "名称":      name,
        "信号":      signal,
        "当前价":    _round(close),
        "趋势类型":  trend_type,
        "趋势评分":  int(trend_score),
        "买入形态":  "、".join(buy_patterns)  if buy_patterns  else "无",
        "卖出形态":  "、".join(sell_patterns) if sell_patterns else "无",
        "关键位置":  "、".join(key_levels)    if key_levels    else "无",
        "窗口状态":  window_status,
        "①激进买入": _round(close),
        "②回调买入": _round(pullback_buy) if pullback_buy is not None else "—",
        "③突破买入": _round(breakout_buy),
        "🚀突破触发价": triggers["breakout_price"],
        "📍均线触发价": triggers["ma_price"],
        "🌅抄底关注价": triggers["bottom_price"],
        "触发条件":  triggers["desc"],
        "止损价":    _round(stop),
        "止损距离%": _round((close - stop) / close * 100),
        "目标价":    _round(target),
        "目标空间%": _round((target - close) / close * 100),
        "风险收益比": _round(rr),
        "ATR":       _round(atr),
        "扫描时间":  datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def _fetch_history(code, period):
    days = _period_days(period)
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    end_date   = datetime.now().strftime("%Y%m%d")

    df = ak.stock_zh_a_hist(
        symbol=code, period="daily",
        start_date=start_date, end_date=end_date, adjust="qfq",
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
    out["MA5"]   = out["收盘"].rolling(5).mean()
    out["MA20"]  = out["收盘"].rolling(20).mean()
    out["MA60"]  = out["收盘"].rolling(60).mean()
    out["MA120"] = out["收盘"].rolling(120).mean()
    out["MA200"] = out["收盘"].rolling(200).mean()
    out["VOL5"]  = out["成交量"].rolling(5).mean()
    out["VOL20"] = out["成交量"].rolling(20).mean()

    prev_close = out["收盘"].shift(1)
    tr = pd.concat([
        out["最高"] - out["最低"],
        (out["最高"] - prev_close).abs(),
        (out["最低"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()

    return out.dropna(subset=["MA20", "MA60", "VOL20", "ATR"]).reset_index(drop=True)


# ── 趋势分析 ──────────────────────────────────────────────────────────

def _detect_trend(last, hist):
    """返回 (趋势类型, 趋势评分 0-100)"""
    close = float(last["收盘"])
    ma20  = float(last["MA20"])
    ma60  = float(last["MA60"])
    score = 0

    # MA 多头排列
    if close > ma20:  score += 20
    if ma20  > ma60:  score += 20
    if not pd.isna(last.get("MA120", float("nan"))):
        ma120 = float(last["MA120"])
        if ma60  > ma120: score += 10
        if close > ma120: score +=  5
    if not pd.isna(last.get("MA200", float("nan"))):
        ma200 = float(last["MA200"])
        if close > ma200: score +=  5

    # 价格处于近 20 日高位区间
    recent_close = hist["收盘"].tail(20)
    if close >= float(recent_close.quantile(0.70)):
        score += 10

    # 成交量放大
    if float(last["成交量"]) > float(last["VOL20"]) * 1.1:
        score += 10

    # 高点不断抬高（近 20 日 vs 前 20 日）
    if len(hist) >= 40:
        if hist["最高"].iloc[-20:].max() > hist["最高"].iloc[-40:-20].max():
            score += 10
        if hist["最低"].iloc[-20:].min() > hist["最低"].iloc[-40:-20].min():
            score += 10

    score = min(score, 100)

    # 趋势分类
    if score >= 70 and close > ma20 and ma20 > ma60:
        return "上升趋势", score

    if ma20 > ma60 and close <= ma20 and close >= ma60 * 0.95:
        return "强势回调", score

    if score <= 30 and close < ma60:
        return "下跌趋势", score

    # 底部转强：近 30 日最低点已出现，价格反弹超 5%
    if len(hist) >= 30:
        low30 = float(hist["最低"].tail(30).min())
        if close > low30 * 1.05 and score >= 35:
            return "底部转强", score

    return "横盘震荡", score


# ── 形态检测 ──────────────────────────────────────────────────────────

def _candle(c):
    """解构单根 K 线的基本要素"""
    op = float(c["开盘"])
    cl = float(c["收盘"])
    hi = float(c["最高"])
    lo = float(c["最低"])
    body = abs(cl - op)
    rng  = max(hi - lo, 0.001)
    ls   = min(op, cl) - lo        # 下影线长度
    us   = hi - max(op, cl)        # 上影线长度
    return op, cl, hi, lo, body, rng, ls, us, cl > op, cl < op


def _detect_patterns(df):
    """
    检测蜡烛图形态。
    返回 (buy_patterns, sell_patterns) 两个列表。

    买入形态：锤子线、看涨吞噬、启明星、长白实体、倒锤子、
              刺穿形态、十字星确认、三白兵、放量突破、向上窗口、站上20日线
    卖出形态：吊颈线、流星线、看跌吞噬、乌云盖顶、黄昏星、
              长上影线、高位十字星、三黑鸦、向下窗口、跌破20日线
    """
    if len(df) < 3:
        return [], []

    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    prev2 = df.iloc[-3]

    l_op, l_cl, l_hi, l_lo, l_body, l_rng, l_ls, l_us, l_bull, l_bear = _candle(last)
    p_op, p_cl, p_hi, p_lo, p_body, p_rng, p_ls, p_us, p_bull, p_bear = _candle(prev)
    p2_op,p2_cl,p2_hi,p2_lo,p2_body,p2_rng,p2_ls,p2_us,p2_bull,p2_bear = _candle(prev2)

    atr = float(last["ATR"])
    MIN_BODY = atr * 0.05   # 避免把微型十字星当成实体形态

    # 价格在近 20 日区间中的相对位置（用于区分锤子/吊颈、倒锤子/流星）
    h20 = float(df["最高"].tail(20).max())
    l20 = float(df["最低"].tail(20).min())
    rng20 = max(h20 - l20, 0.001)
    pos   = (l_cl - l20) / rng20      # 0=低位, 1=高位
    at_bottom = pos < 0.30
    at_top    = pos > 0.70

    buy_p  = []
    sell_p = []

    # ── 买入形态 ──────────────────────────────────────

    # 1. 锤子线：低位长下影
    if (l_ls >= l_body * 2.0 and l_us <= l_rng * 0.20
            and l_body >= MIN_BODY and at_bottom):
        buy_p.append("锤子线")

    # 2. 看涨吞噬：阳线实体完全吞掉前阴线实体
    if l_bull and p_bear and l_op <= p_cl and l_cl >= p_op:
        buy_p.append("看涨吞噬")

    # 3. 启明星：阴线 + 小实体 + 阳线，阳线收盘超越第一根中点
    if (p2_bear and p2_body >= atr * 0.5
            and p_body <= atr * 0.4
            and l_bull and l_cl > (p2_op + p2_cl) / 2):
        buy_p.append("启明星")

    # 4. 长白实体：强势大阳线
    if l_bull and l_body >= atr * 1.8 and l_us <= l_body * 0.5:
        buy_p.append("长白实体")

    # 5. 倒锤子：低位长上影（次日需确认）
    if (l_us >= l_body * 2.0 and l_ls <= l_rng * 0.20
            and l_body >= MIN_BODY and at_bottom):
        buy_p.append("倒锤子")

    # 6. 刺穿形态：阴线后阳线低开，收盘穿越前阴线中点
    if (p_bear and l_bull
            and l_op <= p_lo
            and l_cl > (p_op + p_cl) / 2
            and l_cl < p_op):
        buy_p.append("刺穿形态")

    # 7. 十字星确认：十字星后阳线站上十字星高点
    if p_body <= p_rng * 0.10 and l_bull and l_cl > p_hi:
        buy_p.append("十字星确认")

    # 8. 三白兵：三根连续阳线，每根在前根实体内开盘
    if (l_bull and p_bull and p2_bull
            and l_op >= p_cl * 0.98 and l_op <= p_cl
            and p_op >= p2_cl * 0.98 and p_op <= p2_cl
            and l_body >= atr * 0.4 and p_body >= atr * 0.4):
        buy_p.append("三白兵")

    # 9. 放量突破前高
    if len(df) >= 21:
        prev_high20 = float(df["最高"].iloc[-21:-1].max())
        if l_cl > prev_high20 and float(last["成交量"]) > float(last["VOL20"]) * 1.3:
            buy_p.append("放量突破")

    # 10. 向上窗口（跳空高开且守住）
    if l_lo > p_hi:
        buy_p.append("向上窗口")

    # 11. 站上 20 日线
    if l_cl > float(last["MA20"]) and p_cl < float(prev["MA20"]):
        buy_p.append("站上20日线")

    # ── 卖出形态 ──────────────────────────────────────

    # 1. 吊颈线：高位长下影（外形同锤子，但在高位）
    if (l_ls >= l_body * 2.0 and l_us <= l_rng * 0.20
            and l_body >= MIN_BODY and at_top):
        sell_p.append("吊颈线")

    # 2. 流星线：高位长上影
    if (l_us >= l_body * 2.0 and l_ls <= l_rng * 0.20
            and l_body >= MIN_BODY and at_top):
        sell_p.append("流星线")

    # 3. 看跌吞噬：阴线实体完全吞掉前阳线实体
    if l_bear and p_bull and l_op >= p_cl and l_cl <= p_op:
        sell_p.append("看跌吞噬")

    # 4. 乌云盖顶：阳线后阴线高开，收盘跌入前阳线下半段
    if (p_bull and l_bear
            and l_op > p_hi
            and l_cl < (p_op + p_cl) / 2
            and l_cl > p_op):
        sell_p.append("乌云盖顶")

    # 5. 黄昏星：阳线 + 小实体 + 阴线，阴线收盘跌破第一根中点
    if (p2_bull and p2_body >= atr * 0.5
            and p_body <= atr * 0.4
            and l_bear and l_cl < (p2_op + p2_cl) / 2):
        sell_p.append("黄昏星")

    # 6. 长上影线：高位，空方明显压制
    if l_us >= l_body * 2.5 and l_us >= atr * 0.4 and at_top:
        sell_p.append("长上影线")

    # 7. 高位十字星：高位出现犹豫
    if l_body <= l_rng * 0.10 and at_top:
        sell_p.append("高位十字星")

    # 8. 三黑鸦：三根连续阴线
    if (l_bear and p_bear and p2_bear
            and l_op <= p_cl * 1.02 and l_op >= p_cl * 0.98
            and p_op <= p2_cl * 1.02 and p_op >= p2_cl * 0.98
            and l_body >= atr * 0.4 and p_body >= atr * 0.4):
        sell_p.append("三黑鸦")

    # 9. 向下窗口（跳空低开）
    if l_hi < p_lo:
        sell_p.append("向下窗口")

    # 10. 跌破 20 日线
    if l_cl < float(last["MA20"]) and p_cl > float(prev["MA20"]):
        sell_p.append("跌破20日线")

    return buy_p, sell_p


# ── 关键位置 ──────────────────────────────────────────────────────────

def _find_key_levels(last, hist):
    """识别当前价格附近的关键支撑/压力位置"""
    close = float(last["收盘"])
    levels = []
    tol = 0.025     # 2.5% 容差范围

    def near(ref):
        return abs(close - ref) / max(ref, 0.001) <= tol

    # 均线位置
    for label, col in [("20日均线", "MA20"), ("60日均线", "MA60"),
                        ("120日均线", "MA120"), ("200日均线", "MA200")]:
        val = last.get(col, float("nan"))
        if not pd.isna(val) and near(float(val)):
            levels.append(label)

    # 前期支撑（近 60 日局部低点）
    lows = _local_extremes(hist.tail(60), "低")
    if any(near(lo) and lo <= close * 1.005 for lo in lows):
        levels.append("前期支撑")

    # 前期压力（近 60 日局部高点）
    highs = _local_extremes(hist.tail(60), "高")
    if any(near(hi) and hi >= close * 0.995 for hi in highs):
        levels.append("前期压力")

    # 向上窗口支撑（近 30 日）
    n = len(hist)
    for i in range(max(1, n - 30), n):
        gap_bottom = float(hist.iloc[i - 1]["最高"])
        gap_top    = float(hist.iloc[i]["最低"])
        if gap_top > gap_bottom and near(gap_bottom):
            levels.append("窗口支撑")
            break

    return levels


def _local_extremes(df, col_suffix):
    """在 3 根 K 线窗口内找局部极值（低点或高点）"""
    col = "最" + col_suffix
    if col not in df.columns or len(df) < 3:
        return []
    prices = df[col].values
    result = []
    for i in range(1, len(prices) - 1):
        if col_suffix == "低" and prices[i] <= prices[i - 1] and prices[i] <= prices[i + 1]:
            result.append(float(prices[i]))
        elif col_suffix == "高" and prices[i] >= prices[i - 1] and prices[i] >= prices[i + 1]:
            result.append(float(prices[i]))
    return result


# ── 窗口（缺口）状态 ──────────────────────────────────────────────────

def _detect_windows(hist):
    """检测最近一个窗口（缺口）并描述其当前状态"""
    if len(hist) < 5:
        return "无缺口"

    close  = float(hist.iloc[-1]["收盘"])
    recent = hist.tail(20)
    n      = len(recent)

    for i in range(n - 1, 0, -1):
        p_hi = float(recent.iloc[i - 1]["最高"])
        p_lo = float(recent.iloc[i - 1]["最低"])
        c_lo = float(recent.iloc[i]["最低"])
        c_hi = float(recent.iloc[i]["最高"])

        if c_lo > p_hi:                          # 向上窗口
            status = "守住" if close >= p_hi else "已跌穿"
            return f"上升窗口（{_round(p_hi)}-{_round(c_lo)}，{status}）"

        if c_hi < p_lo:                          # 向下窗口
            status = "守住" if close <= p_lo else "已反扑"
            return f"下降窗口（{_round(c_hi)}-{_round(p_lo)}，{status}）"

    return "无明显缺口"


def _find_unfilled_down_window_top(hist):
    """最近一个未填补的下降窗口的上沿价（前一根的最低价）；无则 None。"""
    if len(hist) < 5:
        return None
    close  = float(hist.iloc[-1]["收盘"])
    recent = hist.tail(20)
    n      = len(recent)
    for i in range(n - 1, 0, -1):
        p_hi = float(recent.iloc[i - 1]["最高"])
        p_lo = float(recent.iloc[i - 1]["最低"])
        c_lo = float(recent.iloc[i]["最低"])
        c_hi = float(recent.iloc[i]["最高"])
        if c_lo > p_hi:               # 最近的是向上窗口，无下降窗口需追踪
            return None
        if c_hi < p_lo and close < p_lo:    # 未填补的下降窗口
            return p_lo
    return None


# ── 预测买点：把形态成立条件反推成触发阈值 ────────────────────────
def _predict_buy_triggers(close, hist, last, atr, trend_type):
    """
    输出三类前瞻触发价 + 一行文字说明：
      - 🚀 突破触发价：max(20日高, 未填补下降窗口上沿)，收盘站上 + 放量 1.3×
      - 📍 均线触发价：MA20 — 上升趋势=回踩不破，下跌趋势=收盘站上
      - 🌅 抄底关注价：仅下跌/底部转强且贴近 30 日低点时给出
    """
    ma20  = float(last["MA20"])
    high20 = float(hist["最高"].tail(20).max())
    low30  = float(hist["最低"].tail(30).min()) if len(hist) >= 30 else float(hist["最低"].min())

    gap_top  = _find_unfilled_down_window_top(hist)
    breakout = max(high20, gap_top) if gap_top else high20

    show_bottom  = trend_type in ("下跌趋势", "底部转强") and close <= low30 * 1.08
    bottom_price = _round(low30) if show_bottom else "—"

    parts = []
    if close < breakout * 0.999:
        gap_note = f"（含窗口上沿 {_round(gap_top)}）" if gap_top and gap_top < high20 else ""
        parts.append(f"🚀 收盘≥{_round(breakout)} + 量>1.3×VOL20 → 突破{gap_note}")

    if close > ma20:
        parts.append(f"📍 回踩{_round(ma20)}不破 → 趋势延续")
    else:
        parts.append(f"📍 收盘≥{_round(ma20)} → 站上MA20反转")

    if show_bottom:
        parts.append(f"🌅 触{_round(low30)}后反弹收阳 → 锤子/启明星抄底")

    return {
        "breakout_price": _round(breakout),
        "ma_price":       _round(ma20),
        "bottom_price":   bottom_price,
        "desc":           " | ".join(parts) if parts else "—",
    }


# ── 止损与目标 ────────────────────────────────────────────────────────

def _calc_stop_target(close, atr, hist, buy_patterns):
    """
    止损：取形态止损和 ATR 止损中较高者（止损越紧越好）
    目标：取前高和 ATR 扩展中较大者
    """
    last  = hist.iloc[-1]
    low5  = float(hist["最低"].tail(5).min())
    low20 = float(hist["最低"].tail(20).min())
    high20 = float(hist["最高"].tail(20).max())

    # 止损候选：越高越好（离当前价越近）
    stop_candidates = [
        close - 2.0 * atr,        # 宽 ATR 兜底
        close - 1.5 * atr,        # 标准 ATR
        low5  * 0.995,             # 近 5 日低点下方
    ]

    # 形态止损：利用形态本身的低点，止损最紧
    if "锤子线" in buy_patterns or "看涨吞噬" in buy_patterns:
        stop_candidates.append(float(last["最低"]) * 0.995)
    if "启明星" in buy_patterns and len(hist) >= 3:
        star_low = float(hist.iloc[-3:]["最低"].min())
        stop_candidates.append(star_low * 0.995)
    if "三白兵" in buy_patterns and len(hist) >= 3:
        stop_candidates.append(float(hist.iloc[-3]["最低"]) * 0.995)

    # 取所有候选中最高且低于当前价的值（止损最紧）
    valid_stops = [s for s in stop_candidates if s < close]
    stop = max(valid_stops) if valid_stops else close * 0.93
    stop = max(stop, close * 0.85)   # 硬底：最大亏损 15%

    # 目标候选：越高越好
    target_candidates = [
        close + 2.0 * atr,
        high20,
    ]
    if "放量突破" in buy_patterns or "向上窗口" in buy_patterns:
        target_candidates.append(close + 3.0 * atr)
    if "三白兵" in buy_patterns or "长白实体" in buy_patterns:
        target_candidates.append(close + 2.5 * atr)

    target = max(t for t in target_candidates if t > close)

    return stop, target


# ── 综合信号 ──────────────────────────────────────────────────────────

def _determine_signal(trend_type, trend_score, rr, buy_p, sell_p, close, last):
    """
    综合趋势、形态、风险收益比给出最终信号。
    优先级：卖出信号 > 买入观察 > 关注候选 > 无明显信号
    """
    ma20 = float(last["MA20"])
    ma60 = float(last["MA60"])

    strong_buy  = any(p in buy_p  for p in ["看涨吞噬", "启明星", "放量突破", "三白兵", "长白实体"])
    strong_sell = any(p in sell_p for p in ["看跌吞噬", "黄昏星", "乌云盖顶", "三黑鸦"])

    # 卖出优先：只有出现真实卖出形态 + 弱势才标红
    if strong_sell and trend_type in ("横盘震荡", "下跌趋势"):
        return "🔴 卖出/止盈"
    if sell_p and trend_type == "下跌趋势":
        return "🔴 卖出/止盈"
    # 趋势弱 + 无买点 → "无信号"（而不是"该卖"），避免把酝酿反转的底部票误判
    if close < ma60 and not buy_p:
        return "⚪ 无明显信号"

    # 买入
    if trend_type in ("上升趋势", "强势回调", "底部转强"):
        if strong_buy and rr >= 2.0 and trend_score >= 60:
            return "🟢 买入观察"
        if (buy_p or trend_score >= 65) and rr >= 1.5 and close >= ma20:
            return "🟡 关注候选"
        if trend_score >= 55 and rr >= 1.0 and close >= ma20:
            return "🟡 关注候选"

    return "⚪ 无明显信号"


# ── 工具函数 ──────────────────────────────────────────────────────────

def _period_days(period):
    return {"3mo": 100, "6mo": 220, "1y": 400, "2y": 780}.get(str(period).lower(), 400)


def _round(value):
    if pd.isna(value):
        return ""
    return round(float(value), 2)
