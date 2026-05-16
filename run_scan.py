"""
run_scan.py  ── GitHub Actions 入口
────────────────────────────────────
读取环境变量 SPREADSHEET_ID，跑全市场扫描，结果写入 Google Sheets + 保存 Excel

本地运行可选环境变量：
  LIMIT=50   只扫描股票池前 N 只（用于快速验证）
  WORKERS=8  并发线程数（默认 10）
"""

import os
from datetime import datetime

import local_patch  # noqa: F401  本地：把东财接口劫持到新浪源
from full_scan import get_universe, run_full_scan, save_excel, pick_strategies
from sheets_writer import write_to_sheet

SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "")
KEY_FILE       = "service_account.json"


def main():
    print(f"=== 开始扫描 {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")

    # 1. 获取股票池（仅沪深主板：60x / 00x，剔除 ST/创业板/科创/北交所）
    universe = get_universe(min_price=3.0, max_price=500.0, min_turnover=0.3,
                            main_board_only=True)
    if universe.empty:
        print("❌ 获取股票列表失败")
        return
    print(f"📋 股票池：{len(universe)} 只（沪深主板）")

    limit = int(os.environ.get("LIMIT", 0))
    if limit > 0:
        print(f"⚠️  本地测试模式：仅扫描前 {limit} 只")
        universe = universe.head(limit)

    # 2. 并发扫描
    workers = int(os.environ.get("WORKERS", 10))
    result_df = run_full_scan(universe, period="1y", workers=workers)
    if result_df.empty:
        print("⚠️  扫描无结果，将写入空结果以清理旧数据")

    # 3. 4 策略选股
    picks = pick_strategies(result_df) if not result_df.empty else None

    # 4. 保存 Excel（多 sheet：主表 + 4 策略）
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    save_excel(result_df, f"A股扫描_{ts}.xlsx", picks=picks)

    # 5. 写入 Google Sheets
    if SPREADSHEET_ID:
        write_to_sheet(result_df, SPREADSHEET_ID, picks=picks, key_file=KEY_FILE)
    else:
        print("⚠️  未设置 SPREADSHEET_ID，跳过写入 Sheets")

    print("=== 扫描完成 ===")


if __name__ == "__main__":
    main()
