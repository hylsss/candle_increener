"""
local_patch.py
────────────────────────────────────
东方财富的 push2his 接口在本机网络 + GitHub Actions runner（Azure IP）
下都被服务端断开/拒连，把取 K 线函数改走新浪源：

  ak.stock_zh_a_hist → ak.stock_zh_a_daily   (sina)

经验证：本机 + Actions 都用 sina 才能稳定跑通；东财在两边都不可用。
"""

import akshare as ak
import pandas as pd


def _to_sina_symbol(code):
    code = str(code).zfill(6)
    first = code[0]
    if first in ("6", "5", "9"):
        return "sh" + code
    if first == "8" or code.startswith("4"):
        return "bj" + code
    return "sz" + code


_SINA_TO_PROJECT_COLS = {
    "date": "日期",
    "open": "开盘",
    "close": "收盘",
    "high": "最高",
    "low":  "最低",
    "volume": "成交量",
    "amount": "成交额",
}


def _stock_zh_a_hist_via_sina(symbol, period="daily", start_date="19700101",
                              end_date="22220101", adjust=""):
    if period != "daily":
        raise NotImplementedError(f"local_patch only supports period='daily', got {period!r}")

    df = ak.stock_zh_a_daily(
        symbol=_to_sina_symbol(symbol),
        start_date=start_date,
        end_date=end_date,
        adjust=adjust or "",
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=list(_SINA_TO_PROJECT_COLS.values()))

    out = df.rename(columns=_SINA_TO_PROJECT_COLS)
    keep = [c for c in _SINA_TO_PROJECT_COLS.values() if c in out.columns]
    out = out[keep].copy()
    out["日期"] = pd.to_datetime(out["日期"]).dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


ak.stock_zh_a_hist = _stock_zh_a_hist_via_sina
