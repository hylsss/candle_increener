"""
sector_resonance.py  ── 板块-个股共振选股
────────────────────────────────────────────
每天先排出最强的板块，再只在这些板块的成分股里跑缠论买点，
最后用「板块强度 × 个股信号质量」共振评分挑出最值得关注的票。

设计目标（对应需求）：
  · 不要长线/宽泛的票 → 池子=今日强势板块成分股，天然聚焦、偏短线
  · 针对每天板块分析选股 → 板块给方向，缠论给买点，两维交叉验证

数据源策略（东财接口不稳，做了降级与缓存）：
  · 板块排名：东财 stock_board_industry_name_em（主）→ 新浪 stock_sector_spot（备）
  · 成分股：  东财 stock_board_industry_cons_em，按板块名缓存到 sector_cache/，
              每天只需拉 TOP-N 个板块的成分（约 6 次调用），且缓存复用，
              过期（默认 7 天）才重新拉。把对东财的依赖降到最低。

复用现有引擎：
  · scan_b123._detect_b123   单股 B1/B2/B3 信号检测 + 质量评分
  · scan_b123._fetch_histories  历史日线批量拉取
  · full_scan.get_universe   流动性/价格/主板过滤

用法：
  python sector_resonance.py                 # 默认 TOP6 板块 × 每板块前 3 只，B2+B3
  SR_TOP_SECTORS=8 python sector_resonance.py
  SR_PER_SECTOR=5 python sector_resonance.py
  SR_TYPES=B2,B3 python sector_resonance.py
  SR_MIN_SCORE=30 python sector_resonance.py
  SR_REFRESH=1 python sector_resonance.py    # 强制刷新成分股缓存

环境变量：
  SR_TOP_SECTORS  取前几个强势板块         default 6
  SR_PER_SECTOR   每板块最多输出几只        default 3
  SR_TYPES        信号类型（逗号分隔）       default B2,B3
  SR_LOOKBACK     信号有效期（交易日）       default 20
  SR_DAYS         历史数据日历天数           default 800
  SR_MIN_SCORE    个股最低质量分（0-100）    default 25
  SR_WORKERS      历史拉取并发数             default 1
  SPREADSHEET_ID  Google Sheets 表格 ID（设了才推送，建当日彩色 tab）
  CB_KEY          service account 密钥文件   default service_account.json
  SR_CACHE_TTL    成分缓存有效天数           default 7
  SR_REFRESH      =1 强制刷新成分缓存
  SR_MULTI_CONFIRM =1 对候选做 60/30 分钟多级别确认（见 multi_level.py）
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 关闭代理，避免 akshare 请求被拦截（与 run_b123.sh 一致）
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import pandas as pd

import local_patch  # noqa: F401  本地补丁，必须在 akshare 之前
import akshare as ak

from full_scan import get_universe
from scan_b123 import _detect_b123, _fetch_histories


CACHE_DIR = Path(__file__).parent / "sector_cache"


# ══════════════════════════════════════════════════════════════════════
# 通用：带退避重试（东财接口经常 RemoteDisconnected）
# ══════════════════════════════════════════════════════════════════════
def _retry(fn, tries: int = 5, base_sleep: float = 1.5, label: str = ""):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(base_sleep * (i + 1))
    if label:
        print(f"   ⚠️  {label} 重试 {tries} 次仍失败: {str(last)[:60]}")
    return None


# ══════════════════════════════════════════════════════════════════════
# 1. 板块排名（东财主源 + 新浪备源）
# ══════════════════════════════════════════════════════════════════════
def rank_sectors(top_n: int = 6) -> pd.DataFrame:
    """
    返回今日板块强弱排名（已截断到 top_n）。
    列：板块, 涨跌幅, 成交额亿, 综合评分, 资金净额亿(可选), 来源
    综合评分 = 涨跌幅 × log10(成交额亿+1)   （涨幅+成交额为主）
    若东财资金流可用，则对评分做轻度加成（资金流为辅）。
    """
    df = _rank_sectors_em()
    if df is None or df.empty:
        print("   东财板块行情不可用，降级到新浪行业…")
        df = _rank_sectors_sina()
    if df is None or df.empty:
        raise RuntimeError("板块排名失败：东财与新浪均不可用")

    # 资金流为辅：能取到就给 net>0 的板块加成（最多 +15%）
    flow = _sector_fund_flow_map()
    if flow:
        def _bonus(row):
            net = flow.get(_norm_name(row["板块"]))
            if net is None:
                return row["综合评分"]
            factor = 1.0 + max(min(net / 30.0, 0.15), -0.10)  # ±按净额温和调整
            return row["综合评分"] * factor
        df["资金净额亿"] = df["板块"].map(lambda n: flow.get(_norm_name(n)))
        df["综合评分"] = df.apply(_bonus, axis=1)

    df = df.sort_values("综合评分", ascending=False).reset_index(drop=True)
    return df.head(top_n)


def _rank_sectors_em() -> pd.DataFrame | None:
    raw = _retry(ak.stock_board_industry_name_em, label="东财板块行情")
    if raw is None or raw.empty:
        return None
    df = raw.copy()
    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
    # 东财成交额单位为元
    amt_col = "总成交额" if "总成交额" in df.columns else "成交额"
    df["成交额亿"] = pd.to_numeric(df.get(amt_col, 0), errors="coerce") / 1e8
    df = df.dropna(subset=["涨跌幅", "成交额亿"])
    df = df.rename(columns={"板块名称": "板块"})
    df["综合评分"] = df["涨跌幅"] * df["成交额亿"].apply(
        lambda x: math.log10(x + 1) if x > 0 else 0)
    df["来源"] = "东财"
    df["label"] = df["板块"]   # 东财成分接口按板块中文名取，label 即板块名
    return df[["板块", "涨跌幅", "成交额亿", "综合评分", "来源", "label"]]


def _rank_sectors_sina() -> pd.DataFrame | None:
    raw = _retry(lambda: ak.stock_sector_spot(indicator="新浪行业"),
                 label="新浪行业行情")
    if raw is None or raw.empty:
        return None
    df = raw.copy()
    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
    df["成交额亿"] = pd.to_numeric(df["总成交额"], errors="coerce") / 1e8
    df = df.dropna(subset=["涨跌幅", "成交额亿"])
    df = df[df["成交额亿"] > 0]
    df["综合评分"] = df["涨跌幅"] * df["成交额亿"].apply(
        lambda x: math.log10(x + 1) if x > 0 else 0)
    df["来源"] = "新浪"
    # 新浪成分接口 stock_sector_detail 按 label（如 new_dzqj）取
    df["label"] = df["label"] if "label" in df.columns else df["板块"]
    return df[["板块", "涨跌幅", "成交额亿", "综合评分", "来源", "label"]]


def _sector_fund_flow_map() -> dict | None:
    """东财行业主力净流入（亿）。取不到返回 None（不阻断主流程）。"""
    raw = _retry(
        lambda: ak.stock_sector_fund_flow_rank(indicator="今日",
                                               sector_type="行业资金流"),
        tries=3, label="东财行业资金流（辅）")
    if raw is None or raw.empty:
        return None
    raw["净额亿"] = pd.to_numeric(raw["今日主力净流入-净额"], errors="coerce") / 1e8
    return {_norm_name(n): v for n, v in zip(raw["名称"], raw["净额亿"])
            if pd.notna(v)}


def _norm_name(name: str) -> str:
    """板块名归一化，便于跨源匹配（去掉常见后缀差异）。"""
    s = str(name).strip()
    for suf in ("行业", "板块", "Ⅲ", "Ⅱ", "Ⅰ"):
        s = s.replace(suf, "")
    return s


# ══════════════════════════════════════════════════════════════════════
# 2. 板块成分股（东财，按需拉取 + 落盘缓存）
# ══════════════════════════════════════════════════════════════════════
def get_sector_members(board_name: str, source: str, label: str,
                       ttl_days: int = 7, force: bool = False) -> list[str]:
    """
    返回某板块成分股代码列表（6 位）。优先读缓存，过期/缺失才拉接口。
    成分股走与排名「同源」的接口，避免跨源板块名对不上：
      · 东财排名 → stock_board_industry_cons_em(板块中文名)
      · 新浪排名 → stock_sector_detail(label，如 new_dzqj)
    拉取失败时回退旧缓存，再不行返回空（该板块当天跳过，不阻断整体）。
    """
    CACHE_DIR.mkdir(exist_ok=True)
    key = label if source == "新浪" else board_name
    safe = "".join(c for c in str(key) if c.isalnum() or c in "（）()_")
    cache_file = CACHE_DIR / f"{source}_{safe}.json"

    if not force and cache_file.exists():
        age = (datetime.now()
               - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age <= ttl_days:
            try:
                return json.loads(cache_file.read_text())["codes"]
            except Exception:  # noqa: BLE001
                pass

    codes = (_fetch_members_sina(label) if source == "新浪"
             else _fetch_members_em(board_name))
    if not codes:
        # 拉取失败但有旧缓存就用旧的（哪怕过期），否则空
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())["codes"]
            except Exception:  # noqa: BLE001
                return []
        return []

    cache_file.write_text(json.dumps(
        {"board": board_name, "source": source, "codes": codes,
         "updated": datetime.now().isoformat()}, ensure_ascii=False))
    return codes


def _fetch_members_em(board_name: str) -> list[str]:
    cons = _retry(
        lambda: ak.stock_board_industry_cons_em(symbol=board_name),
        tries=4, label=f"东财成分股[{board_name}]")
    if cons is None or cons.empty or "代码" not in cons.columns:
        return []
    return [str(c).zfill(6) for c in cons["代码"].tolist()]


def _fetch_members_sina(label: str) -> list[str]:
    d = _retry(lambda: ak.stock_sector_detail(sector=label),
               tries=4, label=f"新浪成分股[{label}]")
    if d is None or d.empty or "code" not in d.columns:
        return []
    return [str(c).zfill(6) for c in d["code"].tolist()]


# ══════════════════════════════════════════════════════════════════════
# 3. 主流程：板块 → 成分股 → 缠论买点 → 共振评分
# ══════════════════════════════════════════════════════════════════════
def scan_resonance(top_sectors: int = 6,
                   per_sector: int = 3,
                   target_types: set[str] | None = None,
                   lookback_days: int = 20,
                   period_days: int = 800,
                   min_score: int = 25,
                   workers: int = 1,
                   cache_ttl: int = 7,
                   force_refresh: bool = False) -> pd.DataFrame:
    if target_types is None:
        target_types = {"B2", "B3"}

    # —— 板块排名 ——
    sectors = rank_sectors(top_n=top_sectors)
    print(f"\n📊 今日强势板块 TOP{top_sectors}（{sectors.iloc[0]['来源']}源）:")
    for _, r in sectors.iterrows():
        flow = f"  资金{r['资金净额亿']:+.1f}亿" if "资金净额亿" in r and pd.notna(
            r.get("资金净额亿")) else ""
        print(f"   {r['板块']:　<10} 涨{r['涨跌幅']:+.2f}%  "
              f"成交{r['成交额亿']:.0f}亿  评分{r['综合评分']:.2f}{flow}")

    # 板块强度归一化到 0-100（用于共振分）
    smin, smax = sectors["综合评分"].min(), sectors["综合评分"].max()
    span = (smax - smin) or 1.0
    sectors = sectors.copy()
    sectors["板块强度"] = 50 + 50 * (sectors["综合评分"] - smin) / span

    # —— 流动性池（一次性）——
    uni = get_universe(min_price=3.0, max_price=500.0,
                       min_turnover=0.5, main_board_only=True)
    uni_codes = set(uni["代码"].astype(str))
    name_map = dict(zip(uni["代码"].astype(str), uni["名称"].astype(str)))
    price_map = dict(zip(uni["代码"].astype(str), uni["当前价"]))

    all_records: list[dict] = []
    for _, sec in sectors.iterrows():
        board = sec["板块"]
        members = get_sector_members(
            board, source=sec["来源"], label=sec.get("label", board),
            ttl_days=cache_ttl, force=force_refresh)
        # 与流动性池取交集（剔除 ST/低流动/非主板）
        codes = [c for c in members if c in uni_codes]
        print(f"\n▶ {board}: 成分 {len(members)} → 池内 {len(codes)} 只")
        if not codes:
            continue

        histories = _fetch_histories(codes, period_days, workers)
        sec_records: list[dict] = []
        for code, hist in histories.items():
            recs = _detect_b123(
                code=code, name=name_map.get(code, ""),
                cur_price=price_map.get(code), hist=hist,
                lookback_days=lookback_days,
                target_types=target_types, min_score=min_score)
            for rec in recs:
                rec["板块"] = board
                rec["板块涨跌%"] = round(float(sec["涨跌幅"]), 2)
                rec["板块强度"] = round(float(sec["板块强度"]), 1)
                # 共振分 = 0.40 板块强度 + 0.45 个股质量 + 0.15 趋势
                rec["共振分"] = round(
                    0.40 * sec["板块强度"]
                    + 0.45 * rec["质量分"]
                    + 0.15 * rec["趋势评分"], 1)
                sec_records.append(rec)

        # 板块内按共振分取前 per_sector 只
        sec_records.sort(key=lambda r: r["共振分"], reverse=True)
        kept = sec_records[:per_sector]
        print(f"   命中 {len(sec_records)} 条 → 取前 {len(kept)} 只")
        all_records.extend(kept)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.sort_values(["共振分", "质量分"], ascending=False).reset_index(drop=True)

    # 可选：对最终候选做多级别（60/30分）确认
    if os.environ.get("SR_MULTI_CONFIRM", "") == "1":
        df = _attach_multi_confirm(df, target_types)
    return df


def _attach_multi_confirm(df: pd.DataFrame, target_types: set[str]) -> pd.DataFrame:
    """对每个候选跑 60/30 分钟级别，标注是否有更短线买点共振。"""
    try:
        from multi_level import analyze_multi
    except Exception:  # noqa: BLE001
        return df
    print("\n🔬 多级别确认（60/30分钟）…")
    verdicts, marks, lv = [], [], []
    for _, row in df.iterrows():
        try:
            res = analyze_multi(row["代码"], row["名称"],
                                levels=["daily", "60", "30"],
                                target_types=target_types, min_score=20)
            r = res["共振"]
            verdicts.append(r["结论"]); marks.append(r["标记"])
            lv.append("、".join(r["信号级别"]) if r["信号级别"] else "")
        except Exception:  # noqa: BLE001
            verdicts.append("-"); marks.append(""); lv.append("")
    df["多级别结论"] = verdicts
    df["分钟买点"] = lv
    df["共振标记"] = marks
    return df


# ══════════════════════════════════════════════════════════════════════
# 4. Google Sheets 写入（复用 scan_b123 的连接 + 格式化风格）
# ══════════════════════════════════════════════════════════════════════
_SHEET_COLS = [
    "代码", "名称", "板块", "板块涨跌%", "信号类型", "信号日期",
    "距今(交易日)", "信号价", "当前价", "距信号涨幅%",
    "质量分", "趋势评分", "板块强度", "共振分",
    "共振标记", "多级别结论", "分钟买点", "扫描时间",
]


def write_to_sheet(df: pd.DataFrame, spreadsheet_id: str,
                   key_file: str = "service_account.json"):
    """把共振选股结果写入 Google Sheets，新建带配色/筛选的当日 tab。"""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        print("⚠️  gspread 未安装，跳过 Sheets 写入")
        return

    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    book = _retry(
        lambda: gspread.authorize(
            Credentials.from_service_account_file(key_file, scopes=scopes)
        ).open_by_key(spreadsheet_id),
        tries=3, base_sleep=2, label="Google Sheets 连接")
    if book is None:
        print("⚠️  Sheets 连接失败（网络/凭证），已跳过推送；本地 Excel 仍可用")
        return

    cols = [c for c in _SHEET_COLS if c in df.columns]
    cols += [c for c in df.columns if c not in cols]  # 其余列兜底
    df_out = df[cols].copy()
    for col in df_out.columns:
        if "日期" in col or "时间" in col:
            df_out[col] = df_out[col].astype(str)

    tab = f"🎯 板块共振 {datetime.now().strftime('%Y-%m-%d %H%M')}"
    try:
        ws = book.worksheet(tab); ws.clear()
    except Exception:  # noqa: BLE001
        ws = book.add_worksheet(title=tab, rows=max(len(df_out) + 5, 10),
                                cols=len(df_out.columns))

    def _safe(v):
        if isinstance(v, float) and math.isnan(v):
            return ""
        return str(v) if hasattr(v, "isoformat") else v

    rows = [list(df_out.columns)]
    rows += [[_safe(v) for v in r] for _, r in df_out.iterrows()]
    ws.update(rows, "A1", value_input_option="RAW")

    sid = ws._properties["sheetId"]
    n_cols, n_rows = len(df_out.columns), len(df_out)
    color_map = {"B3": {"red": 0.82, "green": 0.90, "blue": 0.98},
                 "B2": {"red": 0.85, "green": 0.97, "blue": 0.85},
                 "B1": {"red": 0.99, "green": 0.97, "blue": 0.82}}
    reqs = [
        {"repeatCell": {
            "range": {"sheetId": sid, "startRowIndex": 0, "endRowIndex": 1,
                      "startColumnIndex": 0, "endColumnIndex": n_cols},
            "cell": {"userEnteredFormat": {
                "textFormat": {"bold": True, "fontSize": 10,
                               "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
                "backgroundColor": {"red": 0.12, "green": 0.23, "blue": 0.37},
                "horizontalAlignment": "CENTER"}},
            "fields": "userEnteredFormat"}},
        {"updateSheetProperties": {
            "properties": {"sheetId": sid,
                           "gridProperties": {"frozenRowCount": 1}},
            "fields": "gridProperties.frozenRowCount"}},
        {"setBasicFilter": {"filter": {
            "range": {"sheetId": sid, "startRowIndex": 0,
                      "endRowIndex": n_rows + 1,
                      "startColumnIndex": 0, "endColumnIndex": n_cols}}}},
    ]
    for ri, (_, row) in enumerate(df_out.iterrows()):
        bg = color_map.get(str(row.get("信号类型", "B2")), color_map["B2"])
        reqs.append({"repeatCell": {
            "range": {"sheetId": sid, "startRowIndex": ri + 1,
                      "endRowIndex": ri + 2, "startColumnIndex": 0,
                      "endColumnIndex": n_cols},
            "cell": {"userEnteredFormat": {
                "backgroundColor": bg, "horizontalAlignment": "CENTER",
                "textFormat": {"fontSize": 9}}},
            "fields": "userEnteredFormat"}})
    reqs.append({"autoResizeDimensions": {"dimensions": {
        "sheetId": sid, "dimension": "COLUMNS",
        "startIndex": 0, "endIndex": n_cols}}})
    book.batch_update({"requests": reqs})
    print(f"✅ 已写入 Google Sheets tab：{tab}（{n_rows} 只）")
    print(f"   https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit")


# ══════════════════════════════════════════════════════════════════════
# 5. 入口
# ══════════════════════════════════════════════════════════════════════
def main():
    top_sectors = int(os.environ.get("SR_TOP_SECTORS", 6))
    per_sector  = int(os.environ.get("SR_PER_SECTOR", 3))
    lookback    = int(os.environ.get("SR_LOOKBACK", 20))
    period_days = int(os.environ.get("SR_DAYS", 800))
    min_score   = int(os.environ.get("SR_MIN_SCORE", 25))
    workers     = int(os.environ.get("SR_WORKERS", 1))
    cache_ttl   = int(os.environ.get("SR_CACHE_TTL", 7))
    force       = os.environ.get("SR_REFRESH", "") == "1"
    types_str   = os.environ.get("SR_TYPES", "B2,B3")
    target_types = {t.strip().upper() for t in types_str.split(",") if t.strip()}

    print(f"=== 板块共振选股 {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    print(f"  TOP{top_sectors} 板块 × 每板块 {per_sector} 只 | "
          f"信号 {'+'.join(sorted(target_types))} | "
          f"有效期 {lookback} 交易日 | 最低质量分 {min_score}")

    df = scan_resonance(
        top_sectors=top_sectors, per_sector=per_sector,
        target_types=target_types, lookback_days=lookback,
        period_days=period_days, min_score=min_score,
        workers=workers, cache_ttl=cache_ttl, force_refresh=force)

    if df.empty:
        print("\n⚠️  今日强势板块内未找到符合条件的买点信号。")
        sys.exit(0)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = f"板块共振选股_{ts}.xlsx"
    try:
        df.to_excel(out, index=False)
        print(f"\n✅ Excel: {out}")
    except Exception as e:  # noqa: BLE001
        csv = out.replace(".xlsx", ".csv")
        df.to_csv(csv, index=False)
        print(f"\n✅ CSV: {csv}（Excel 写入失败: {str(e)[:40]}）")

    print(f"\n{'='*64}")
    print("🎯 今日板块共振买点（按共振分排序）")
    print(f"{'='*64}")
    show = ["代码", "名称", "板块", "板块涨跌%", "信号类型",
            "信号日期", "距今(交易日)", "信号价", "当前价",
            "距信号涨幅%", "质量分", "趋势评分", "共振分",
            "共振标记", "多级别结论", "分钟买点"]
    show = [c for c in show if c in df.columns]
    print(df[show].to_string(index=False))

    # Google Sheets 写入（设了 SPREADSHEET_ID 才写）
    sheet_id = os.environ.get("SPREADSHEET_ID", "")
    key_file = os.environ.get("CB_KEY", "service_account.json")
    if sheet_id:
        write_to_sheet(df, sheet_id, key_file)
    else:
        print("\nℹ️  未设置 SPREADSHEET_ID，跳过 Google Sheets 写入")
        print("   推送到表格：SPREADSHEET_ID=<表格ID> python sector_resonance.py")


if __name__ == "__main__":
    main()
