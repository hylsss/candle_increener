"""
run_scan.py  ── GitHub Actions 入口
────────────────────────────────────
读取环境变量 SPREADSHEET_ID，跑全市场扫描，结果写入 Google Sheets + 保存 Excel
"""

import os
from datetime import datetime
from full_scan import get_universe, run_full_scan, save_excel
from sheets_writer import write_to_sheet

SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "")
KEY_FILE       = "service_account.json"


def main():
    print(f"=== 开始扫描 {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")

    # 1. 获取股票池
    universe = get_universe(min_price=3.0, max_price=500.0, min_turnover=0.3)
    if universe.empty:
        print("❌ 获取股票列表失败")
        return

    # 2. 并发扫描（GitHub Actions 机器约 2 核，workers 不宜太高）
    result_df = run_full_scan(universe, period="6mo", workers=10)
    if result_df.empty:
        print("❌ 扫描无结果")
        return

    # 3. 保存 Excel（上传为 Artifact）
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    save_excel(result_df, f"A股扫描_{ts}.xlsx")

    # 4. 写入 Google Sheets
    if SPREADSHEET_ID:
        write_to_sheet(result_df, SPREADSHEET_ID, key_file=KEY_FILE)
    else:
        print("⚠️  未设置 SPREADSHEET_ID，跳过写入 Sheets")

    print("=== 扫描完成 ===")


if __name__ == "__main__":
    main()
