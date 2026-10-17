// MJRV pump readings logger. Receives JSON from the Telegram OCR bot and appends one row per pump.
// Deploy: Deploy > New deployment > Web app, Execute as: Me, Who has access: Anyone.

const SHEET_ID = '1J9vXhhtMgiggSi2rl8kSBRXSEIceUpTiPIYGCxxyehY';  // MJRV Pump Readings
const SHEET_NAME = 'Readings';
const HEADERS = ['Logged At', 'Reading Time', 'Pump', 'Totalizer (L)', 'Price (PHP/L)',
                 'Litres Sold', 'Sales (PHP)', 'Sent By', 'Telegram Msg ID'];

function getSheet_() {
  const ss = SpreadsheetApp.openById(SHEET_ID);
  let sh = ss.getSheetByName(SHEET_NAME) || ss.getSheets()[0];
  if (sh.getName() !== SHEET_NAME) sh.setName(SHEET_NAME);
  if (sh.getLastRow() === 0 || sh.getRange(1, 1).getValue() !== HEADERS[0]) {
    sh.getRange(1, 1, 1, HEADERS.length).setValues([HEADERS]);
  }
  sh.getRange(1, 1, 1, HEADERS.length).setFontWeight('bold');
  sh.setFrozenRows(1);
  return sh;
}

// Run once from the editor to set up formats and grant permissions.
function setup() {
  SpreadsheetApp.openById(SHEET_ID).setSpreadsheetTimeZone('Asia/Manila');
  const sh = getSheet_();
  sh.getRange('A:A').setNumberFormat('yyyy-mm-dd hh:mm');
  sh.getRange('D:D').setNumberFormat('#,##0.00');
  sh.getRange('E:E').setNumberFormat('#,##0.00');
  sh.getRange('F:F').setNumberFormat('#,##0.00');
  sh.getRange('G:G').setNumberFormat('#,##0.00');
}

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    const d = JSON.parse(e.postData.contents);
    if (!Array.isArray(d.pumps) || d.pumps.length === 0) return out_({ ok: false, error: 'no pumps' });

    const sh = getSheet_();
    const last = sh.getLastRow();
    const data = last > 1 ? sh.getRange(2, 1, last - 1, HEADERS.length).getValues() : [];

    // Ignore Telegram re-deliveries of the same message.
    if (d.message_id && data.some(r => String(r[8]) === String(d.message_id))) {
      return out_({ ok: true, duplicate: true });
    }

    const now = new Date();
    const rows = d.pumps.map(p => {
      const total = Number(p.litres), price = Number(p.price);
      // Previous totalizer for this pump = most recent row with the same pump name.
      let prev = null;
      for (let i = data.length - 1; i >= 0; i--) {
        if (data[i][2] === p.name) { prev = Number(data[i][3]); break; }
      }
      const sold = prev === null ? '' : Math.round((total - prev) * 100) / 100;
      const sales = sold === '' ? '' : Math.round(sold * price * 100) / 100;
      return [now, d.reading_time || '', p.name, total, price, sold, sales, d.sent_by || '', d.message_id || ''];
    });

    sh.getRange(last + 1, 1, rows.length, HEADERS.length).setValues(rows);
    return out_({ ok: true, rows: rows.length, sold: rows.map(r => ({ pump: r[2], litres: r[5], sales: r[6] })) });
  } catch (err) {
    return out_({ ok: false, error: String(err) });
  } finally {
    lock.releaseLock();
  }
}

function out_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
