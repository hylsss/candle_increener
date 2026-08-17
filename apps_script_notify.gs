/**
 * Apps Script：蜡烛图扫描结果通知
 * ─────────────────────────────────────────────
 * 功能：
 *   1. 读取 Google Sheets 里最新的严格缠论扫描结果
 *   2. 把 B1/B2/B3 结构合规信号整理成邮件发送
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
const CHAN_TAB_PREFIX = "🎯 缠论买点 ";

// ─────────────────────────────────────────────
// 主函数：读取 Sheets 数据 → 发邮件
// ─────────────────────────────────────────────
function sendScanReport() {
  const ss      = SpreadsheetApp.openById(SPREADSHEET_ID);
  const chanSheet = _latestChanlunSheet(ss);

  if (!chanSheet) {
    Logger.log(`找不到 ${CHAN_TAB_PREFIX} 开头的结果页。`);
    return;
  }

  const buyData = _readSheetData(chanSheet);
  const summary = `严格缠论结构筛选 · ${chanSheet.getName()} · ${buyData.length} 条信号`;

  if (buyData.length === 0) {
    Logger.log("今日无严格缠论信号，不发送邮件。");
    return;
  }

  // 构建邮件 HTML
  const html = _buildEmailHtml(summary, buyData);

  // 发送邮件
  GmailApp.sendEmail(
    NOTIFY_EMAIL,
    `🎯 严格缠论选股 · ${_today()} · ${buyData.length} 条`,
    "请用支持 HTML 的邮件客户端查看",
    { htmlBody: html }
  );

  Logger.log(`✅ 邮件已发送，${buyData.length} 条信号`);
}


// 找名称时间戳最新的严格缠论结果页。
function _latestChanlunSheet(ss) {
  const sheets = ss.getSheets()
    .filter(s => s.getName().startsWith(CHAN_TAB_PREFIX))
    .sort((a, b) => b.getName().localeCompare(a.getName()));
  return sheets.length ? sheets[0] : null;
}


// ─────────────────────────────────────────────
// 读取严格缠论结果页（第1行表头）
// ─────────────────────────────────────────────
function _readSheetData(sheet) {
  const data = sheet.getDataRange().getValues();
  if (data.length <= 1) return [];

  const headers = data[0];
  const rows = data.slice(1).filter(r => r[0] !== "");

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
  const cols = ["代码","名称","信号类型","信号日期","距今(交易日)","信号价",
                "当前价","距信号涨幅%","质量分","趋势中枢数","背驰比(C/A)","ZG","ZD"];
  const visibleCols = cols.filter(c => rows.some(r => Object.prototype.hasOwnProperty.call(r, c)));

  let tableRows = rows.map(r => {
    const cells = visibleCols.map(c => `<td style="${TD}">${r[c] ?? "─"}</td>`).join("");
    return `<tr>${cells}</tr>`;
  }).join("");

  let thCells = visibleCols.map(c => `<th style="${TH}">${c}</th>`).join("");

  return `
  <div style="font-family:Arial,sans-serif;max-width:900px;margin:0 auto;">
    <div style="background:#0D47A1;color:white;padding:16px 20px;border-radius:8px 8px 0 0;">
      <h2 style="margin:0;font-size:18px;">🎯 严格缠论选股报告</h2>
      <p style="margin:4px 0 0;font-size:12px;opacity:0.8;">${summary}</p>
    </div>

    <div style="background:#E3F2FD;padding:10px 20px;border-left:4px solid #1976D2;">
      <b style="color:#1B5E20;">B1/B2/B3 结构合规信号 · 共 ${rows.length} 条</b>
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
