"""
sheets_writer.py
─────────────────────────────────────────
把扫描结果写入 Google Sheets
─────────────────────────────────────────
使用前准备：
  1. 去 console.cloud.google.com 创建项目
  2. 启用 Google Sheets API + Google Drive API
  3. 创建「服务账号」，下载 JSON 密钥，改名为 service_account.json
  4. 在 Google Sheets 里把表格分享给服务账号的邮件地址（编辑权限）
"""

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from datetime import datetime


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# 信号对应的颜色（Google Sheets RGBA 格式，0–1 范围）
SIG_COLORS = {
    "🟢 买入观察": {"red": 0.84, "green": 0.96, "blue": 0.84},
    "🟡 关注候选": {"red": 1.0,  "green": 0.98, "blue": 0.8},
    "🔴 回避/止盈": {"red": 1.0,  "green": 0.87, "blue": 0.87},
    "⚪ 无明显信号": {"red": 0.96, "green": 0.96, "blue": 0.96},
}

HEADER_COLOR = {"red": 0.12, "green": 0.23, "blue": 0.37}   # 深蓝


def connect(key_file: str = "service_account.json"):
    """建立 gspread 连接"""
    creds = Credentials.from_service_account_file(key_file, scopes=SCOPES)
    return gspread.authorize(creds)


def write_to_sheet(df: pd.DataFrame,
                   spreadsheet_id: str,
                   key_file: str = "service_account.json"):
    """
    把扫描结果写入指定 Google Sheets。
    - 第一个 Sheet "扫描结果" 写全部数据
    - 第二个 Sheet "买入观察" 只写买入标的
    - 在 A1 写入本次扫描时间和统计

    spreadsheet_id: 从 Sheets URL 里取，
      https://docs.google.com/spreadsheets/d/【这里】/edit
    """
    gc     = connect(key_file)
    book   = gc.open_by_key(spreadsheet_id)
    now    = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── 写全部结果 ──────────────────────────────────
    _write_tab(book, "📊 全部扫描结果", df, now)

    # ── 写买入观察 ──────────────────────────────────
    buy_df = df[df["信号"] == "🟢 买入观察"].copy()
    _write_tab(book, "🟢 买入观察", buy_df, now)

    # ── 写关注候选 ──────────────────────────────────
    watch_df = df[df["信号"] == "🟡 关注候选"].copy()
    _write_tab(book, "🟡 关注候选", watch_df, now)

    # ── 写卖出/止盈 ─────────────────────────────────
    sell_df = df[df["信号"] == "🔴 卖出/止盈"].copy()
    _write_tab(book, "🔴 卖出止盈", sell_df, now)

    print(f"✅ 已写入 Google Sheets（{now}）")
    print(f"   全部：{len(df)} 只 · 买入观察：{len(buy_df)} 只 · "
          f"关注候选：{len(watch_df)} 只 · 卖出止盈：{len(sell_df)} 只")


def _write_tab(book, tab_name: str, df: pd.DataFrame, timestamp: str):
    """写入或新建单个 Sheet Tab"""
    # 获取或新建 worksheet
    try:
        ws = book.worksheet(tab_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = book.add_worksheet(title=tab_name, rows=max(len(df) + 5, 10), cols=max(len(df.columns), 1))

    cols     = list(df.columns)
    n_cols   = len(cols)
    n_rows   = len(df)

    # ── 构建要写入的所有数据（一次批量写，减少 API 调用）──
    sig_counts = df["信号"].value_counts().to_dict() if "信号" in cols else {}
    summary    = (f"扫描时间：{timestamp}    共 {n_rows} 只    "
                  + "    ".join(f"{k}：{v}" for k, v in sig_counts.items()))

    all_rows = [
        [summary] + [""] * (n_cols - 1),   # 第1行：摘要
        cols,                               # 第2行：列标题
    ]
    for _, row in df.iterrows():
        all_rows.append([_safe(v) for v in row])

    ws.update("A1", all_rows, value_input_option="USER_ENTERED")

    # ── 批量格式化（用 batchUpdate 一次搞定）──────────
    sheet_id  = ws._properties["sheetId"]
    requests  = []

    # 先解除旧的摘要行合并，避免重复运行时 mergeCells 撞上已有合并区域
    requests.append({"unmergeCells": {
        "range": _rng(sheet_id, 0, 0, 1, n_cols)
    }})

    # 1. 合并第一行（摘要行跨列）
    requests.append({"mergeCells": {
        "range": _rng(sheet_id, 0, 0, 1, n_cols),
        "mergeType": "MERGE_ALL"
    }})

    # 2. 摘要行格式
    requests.append({"repeatCell": {
        "range": _rng(sheet_id, 0, 0, 1, n_cols),
        "cell": {"userEnteredFormat": {
            "textFormat":        {"fontSize": 9, "italic": True},
            "horizontalAlignment": "LEFT",
            "backgroundColor":   {"red": 0.93, "green": 0.93, "blue": 0.93},
        }},
        "fields": "userEnteredFormat"
    }})

    # 3. 标题行格式（深蓝底白字）
    requests.append({"repeatCell": {
        "range": _rng(sheet_id, 1, 0, 2, n_cols),
        "cell": {"userEnteredFormat": {
            "textFormat":          {"bold": True, "fontSize": 10,
                                    "foregroundColor": {"red":1,"green":1,"blue":1}},
            "backgroundColor":     HEADER_COLOR,
            "horizontalAlignment": "CENTER",
        }},
        "fields": "userEnteredFormat"
    }})

    # 4. 冻结前两行
    requests.append({"updateSheetProperties": {
        "properties": {
            "sheetId": sheet_id,
            "gridProperties": {"frozenRowCount": 2}
        },
        "fields": "gridProperties.frozenRowCount"
    }})

    # 5. 数据行按信号着色
    for ri, (_, row) in enumerate(df.iterrows()):
        data_row = ri + 2   # 0-indexed: 行0=摘要, 行1=标题, 行2+=数据
        sig = str(row.get("信号", ""))
        bg  = SIG_COLORS.get(sig, SIG_COLORS["⚪ 无明显信号"])
        requests.append({"repeatCell": {
            "range": _rng(sheet_id, data_row, 0, data_row+1, n_cols),
            "cell": {"userEnteredFormat": {
                "backgroundColor":     bg,
                "horizontalAlignment": "CENTER",
                "textFormat":          {"fontSize": 9},
            }},
            "fields": "userEnteredFormat"
        }})

    # 6. 自动筛选（从标题行开始）
    if n_rows > 0:
        requests.append({"setBasicFilter": {
            "filter": {
                "range": _rng(sheet_id, 1, 0, n_rows+2, n_cols)
            }
        }})

    # 7. 列宽
    widths = {
        "代码": 80, "名称": 100, "信号": 130, "当前价": 80,
        "趋势类型": 100, "趋势评分": 80,
        "买入形态": 180, "卖出形态": 180, "关键位置": 140, "窗口状态": 200,
        "①激进买入": 95, "②回调买入": 95, "③突破买入": 95,
        "止损价": 80, "止损距离%": 80, "目标价": 80, "目标空间%": 80,
        "风险收益比": 90, "ATR": 70, "扫描时间": 140,
        # 兼容旧列名
        "趋势评分": 80, "主要形态": 160,
    }
    for ci, col in enumerate(cols):
        px = widths.get(col, 90)
        requests.append({"updateDimensionProperties": {
            "range": {"sheetId": sheet_id, "dimension": "COLUMNS",
                      "startIndex": ci, "endIndex": ci+1},
            "properties": {"pixelSize": px},
            "fields": "pixelSize"
        }})

    book.batch_update({"requests": requests})


def _rng(sheet_id, r1, c1, r2, c2):
    return {"sheetId": sheet_id,
            "startRowIndex": r1, "endRowIndex": r2,
            "startColumnIndex": c1, "endColumnIndex": c2}

def _safe(v):
    """把 numpy/nan 转成 Python 原生，避免 gspread 序列化错误"""
    import numpy as np, math
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return float(v) if not math.isnan(v) else ""
    if isinstance(v, float) and math.isnan(v): return ""
    return v
