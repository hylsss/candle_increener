"""
notifier.py
────────────────────────────────────
扫描完成后通过 PushPlus 推送精选买入名单到微信。

申请 token：
  1. 微信关注【pushplus推送加】公众号
  2. 网页 pushplus.plus 登录，复制 token
  3. 设环境变量 PUSHPLUS_TOKEN=<你的token>
     （GitHub Actions 同名 Secret）

不设 PUSHPLUS_TOKEN 时 push_picks 直接 no-op，便于本地静默运行。
"""

import os
from datetime import datetime

import pandas as pd
import requests


PUSHPLUS_ENDPOINT = "https://www.pushplus.plus/send"


def push_picks(picks: dict, top_n: int = 8, token: str = None) -> bool:
    """
    把 4 个策略各前 top_n 只票排成 markdown 表格推送。
    返回 True/False 表示发送是否成功；token 不存在直接返回 False 静默退出。
    """
    token = token or os.environ.get("PUSHPLUS_TOKEN", "").strip()
    if not token:
        print("ℹ️  未设置 PUSHPLUS_TOKEN，跳过微信推送")
        return False

    if not picks:
        print("ℹ️  无选股结果，跳过微信推送")
        return False

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    counts = {name: len(df) for name, df in picks.items()}
    title = (f"📊 A股扫描 {now[:10]} — "
             + " / ".join(f"{name.split()[0]}{n}" for name, n in counts.items()))

    content_parts = [f"### 扫描时间：{now}\n"]

    for name, df in picks.items():
        if df.empty:
            content_parts.append(f"\n## {name}\n_无命中_\n")
            continue
        content_parts.append(f"\n## {name}（{len(df)} 只，展示前 {min(top_n, len(df))}）\n")
        content_parts.append(_format_markdown_table(df.head(top_n)))

    content_parts.append(
        "\n---\n"
        "_完整名单见 Google Sheets / Excel artifact_\n"
        "_R:R = 风险收益比；触发价 = 突破买点；MA = 均线价_"
    )

    payload = {
        "token": token,
        "title": title,
        "content": "".join(content_parts),
        "template": "markdown",
    }

    try:
        resp = requests.post(PUSHPLUS_ENDPOINT, json=payload, timeout=15)
        data = resp.json()
        if data.get("code") == 200:
            print(f"✅ 微信推送成功（{title}）")
            return True
        print(f"❌ 微信推送失败：code={data.get('code')}  msg={data.get('msg')}")
        return False
    except Exception as exc:
        print(f"❌ 微信推送异常：{exc}")
        return False


def _format_markdown_table(df: pd.DataFrame) -> str:
    """挑出关键列输出紧凑 markdown 表格"""
    show_cols = ["代码", "名称", "信号", "当前价", "趋势评分",
                 "🚀突破触发价", "📍均线触发价", "风险收益比"]
    cols = [c for c in show_cols if c in df.columns]
    short_name = {
        "🚀突破触发价": "🚀触发",
        "📍均线触发价": "📍MA",
        "风险收益比": "R:R",
        "趋势评分": "趋势",
    }

    header = "| " + " | ".join(short_name.get(c, c) for c in cols) + " |"
    sep    = "|" + "|".join(["---"] * len(cols)) + "|"
    rows   = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")

    return "\n".join([header, sep] + rows) + "\n"


def _fmt(v):
    if pd.isna(v) or v == "" or v == "—":
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


if __name__ == "__main__":
    # 自测：构造假数据测推送
    test = {
        "🚀 突破追涨": pd.DataFrame([
            {"代码": "600519", "名称": "贵州茅台", "信号": "🟢 买入观察",
             "当前价": 1332.95, "趋势评分": 95, "🚀突破触发价": 1380.0,
             "📍均线触发价": 1395.0, "风险收益比": 2.5},
        ]),
        "🔄 强势回踩": pd.DataFrame(),
        "🌅 底部反转": pd.DataFrame(),
        "📈 高RR精选": pd.DataFrame(),
    }
    push_picks(test, top_n=3)
