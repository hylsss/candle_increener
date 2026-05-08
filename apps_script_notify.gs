/**
 * Apps Script：蜡烛图扫描结果通知
 * ─────────────────────────────────────────────
 * 功能：
 *   1. 读取 Google Sheets 里的扫描结果
 *   2. 把「买入观察」标的整理成邮件发送
 *   3. 可设定每日自动触发
 *
 * 使用方式：
 *   1. 打开你的 Google Sheets → 扩展程序 → Apps Script
 *   2. 把下面代码全部粘贴进去
 *   3. 修改 SPREADSHEET_ID 和 NOTIFY_EMAIL
 *   4. 运行 setDailyTrigger() 设置定时触发（每天17:00）
 */

const SPREADSHEET_ID = "你的_Sheets_ID_粘贴这里";
const NOTIFY_EMAIL   = "你的邮箱@gmail.com";
const BUY_SHEET_NAME = "🟢 买入观察";
const ALL_SHEET_NAME = "📊 全部扫描结果";

// ─────────────────────────────────────────────
// 主函数：读取 Sheets 数据 → 发邮件
// ─────────────────────────────────────────────
function sendScanReport() {
  const ss      = SpreadsheetApp.openById(SPREADSHEET_ID);
  const buySheet = ss.getSheetByName(BUY_SHEET_NAME);
  const allSheet = ss.getSheetByName(ALL_SHEET_NAME);

  if (!buySheet && !allSheet) {
    Logger.log("找不到扫描结果 Sheet，可能 Python 还未运行完成。");
    return;
  }

  // 读取摘要行（第1行）
  const summary = allSheet
    ? allSheet.getRange(1, 1).getValue()
    : "暂无统计";

  // 读取买入观察标的
  const buyData = buySheet ? _readSheetData(buySheet) : [];

  if (buyData.length === 0) {
    Logger.log("今日无买入观察标的，不发送邮件。");
    return;
  }

  // 构建邮件 HTML
  const html = _buildEmailHtml(summary, buyData);

  // 发送邮件
  GmailApp.sendEmail(
    NOTIFY_EMAIL,
    `📈 A股蜡烛图扫描报告 · ${_today()} · 买入观察 ${buyData.length} 只`,
    "请用支持 HTML 的邮件客户端查看",
    { htmlBody: html }
  );

  Logger.log(`✅ 邮件已发送，买入观察 ${buyData.length} 只`);
}


// ─────────────────────────────────────────────
// 读取 Sheet 数据（跳过前两行：摘要+表头）
// ─────────────────────────────────────────────
function _readSheetData(sheet) {
  const data = sheet.getDataRange().getValues();
  if (data.length < 3) return [];

  const headers = data[1];   // 第2行是列名
  const rows    = data.slice(2).filter(r => r[0] !== "");  // 第3行起是数据

  return rows.map(row => {
    const obj = {};
    headers.forEach((h, i) => { obj[h] = row[i]; });
    return obj;
  });
}


// ─────────────────────────────────────────────
// 构建 HTML 邮件
// ─────────────────────────────────────────────
function _buildEmailHtml(summary, rows) {
  const cols = ["代码","名称","当前价","①激进买入","②回调买入",
                "③突破买入","止损价","目标价","风险收益比","主要形态"];

  let tableRows = rows.map(r => {
    const cells = cols.map(c => `<td style="${TD}">${r[c] ?? "─"}</td>`).join("");
    return `<tr>${cells}</tr>`;
  }).join("");

  let thCells = cols.map(c => `<th style="${TH}">${c}</th>`).join("");

  return `
  <div style="font-family:Arial,sans-serif;max-width:900px;margin:0 auto;">
    <div style="background:#0D47A1;color:white;padding:16px 20px;border-radius:8px 8px 0 0;">
      <h2 style="margin:0;font-size:18px;">📈 A股蜡烛图扫描报告</h2>
      <p style="margin:4px 0 0;font-size:12px;opacity:0.8;">${summary}</p>
    </div>

    <div style="background:#E3F2FD;padding:10px 20px;border-left:4px solid #1976D2;">
      <b style="color:#1B5E20;">🟢 买入观察标的 · 共 ${rows.length} 只</b>
    </div>

    <table style="width:100%;border-collapse:collapse;font-size:12px;">
      <thead><tr>${thCells}</tr></thead>
      <tbody>${tableRows}</tbody>
    </table>

    <div style="padding:12px 20px;background:#F5F5F5;border-top:1px solid #ddd;
                font-size:11px;color:#888;border-radius:0 0 8px 8px;">
      ⚠️ 本报告由程序自动生成，仅供参考，不构成投资建议。请结合基本面和市场情况自行判断。
    </div>
  </div>`;
}

const TH = `background:#1E3A5F;color:white;padding:8px 6px;text-align:center;
             font-size:11px;border:1px solid #2d5a8e;white-space:nowrap;`;
const TD = `padding:7px 6px;text-align:center;border:1px solid #ddd;
             background:#D6F5D6;`;


// ─────────────────────────────────────────────
// 设置每日自动触发（只需运行一次）
// ─────────────────────────────────────────────
function setDailyTrigger() {
  // 清除旧触发器
  ScriptApp.getProjectTriggers().forEach(t => ScriptApp.deleteTrigger(t));

  // 每天 17:00–18:00 之间触发（等 Python 扫描写完 Sheets 后再发邮件）
  ScriptApp.newTrigger("sendScanReport")
    .timeBased()
    .everyDays(1)
    .atHour(17)
    .create();

  Logger.log("✅ 定时触发已设置：每天 17:00 发送报告");
}


// ─────────────────────────────────────────────
// 工具函数
// ─────────────────────────────────────────────
function _today() {
  return Utilities.formatDate(new Date(), "Asia/Shanghai", "yyyy-MM-dd");
}
