// MJRV pump readings logger. Receives JSON from the Telegram OCR bot and logs one row per pump.
// Rows are kept in reading-time order, and Litres Sold / Sales are recalculated per pump after
// every change, so photos sent late or out of order still give correct figures.
// Deploy: Deploy > New deployment > Web app, Execute as: Me, Who has access: Anyone.

const SHEET_ID = '1J9vXhhtMgiggSi2rl8kSBRXSEIceUpTiPIYGCxxyehY';  // MJRV Pump Readings
const SHEET_NAME = 'Readings';
const HEADERS = ['Logged At', 'Reading Time', 'Pump', 'Totalizer (L)', 'Price (PHP/L)',
                 'Litres Sold', 'Sales (PHP)', 'Sent By', 'Telegram Msg ID'];
const NCOL = HEADERS.length;

function getSheet_() {
  const ss = SpreadsheetApp.openById(SHEET_ID);
  let sh = ss.getSheetByName(SHEET_NAME) || ss.getSheets()[0];
  if (sh.getName() !== SHEET_NAME) sh.setName(SHEET_NAME);
  if (sh.getLastRow() === 0 || sh.getRange(1, 1).getValue() !== HEADERS[0]) {
    sh.getRange(1, 1, 1, NCOL).setValues([HEADERS]);
  }
  sh.getRange(1, 1, 1, NCOL).setFontWeight('bold');
  sh.setFrozenRows(1);
  return sh;
}

// Run once from the editor to set up formats and grant permissions.
function setup() {
  SpreadsheetApp.openById(SHEET_ID).setSpreadsheetTimeZone('Asia/Manila');
  const sh = getSheet_();
  sh.getRange('A:B').setNumberFormat('yyyy-mm-dd hh:mm');
  sh.getRange('D:G').setNumberFormat('#,##0.00');
  recalc_(sh);
}

// "2026/10/07 18:02" (camera stamp) -> Date. Falls back to null.
function parseTime_(v) {
  if (v instanceof Date) return v;
  const m = String(v || '').match(/(\d{4})[\/-](\d{2})[\/-](\d{2})\s*(\d{2}):(\d{2})/);
  // Camera stamp is Philippine time (UTC+8), independent of the script's timezone setting.
  return m ? new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4] - 8, +m[5])) : null;
}

function readRows_(sh) {
  const last = sh.getLastRow();
  return last > 1 ? sh.getRange(2, 1, last - 1, NCOL).getValues() : [];
}

function timeOf_(r) {
  return (parseTime_(r[1]) || parseTime_(r[0]) || new Date(0)).getTime();
}

// Sort by reading time and recompute Litres Sold / Sales for every pump.
function recalc_(sh, rows) {
  rows = rows || readRows_(sh);
  rows.forEach(r => { r[1] = parseTime_(r[1]) || r[1]; });
  rows.sort((a, b) => timeOf_(a) - timeOf_(b));
  const prev = {};
  rows.forEach(r => {
    const pump = r[2], total = Number(r[3]), price = Number(r[4]);
    if (pump in prev) {
      const sold = Math.round((total - prev[pump]) * 100) / 100;
      r[5] = sold;
      r[6] = Math.round(sold * price * 100) / 100;
    } else {
      r[5] = '';
      r[6] = '';
    }
    prev[pump] = total;
  });
  if (rows.length) sh.getRange(2, 1, rows.length, NCOL).setValues(rows);
  return rows;
}

// GET: per-pump history [[time ms, totalizer], ...] so the bot can sanity-check new reads.
function doGet() {
  const rows = readRows_(getSheet_());
  const hist = {};
  rows.forEach(r => { (hist[r[2]] = hist[r[2]] || []).push([timeOf_(r), Number(r[3])]); });
  return out_({ ok: true, history: hist });
}

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    const d = JSON.parse(e.postData.contents);
    if (!Array.isArray(d.pumps) || d.pumps.length === 0) return out_({ ok: false, error: 'no pumps' });

    const sh = getSheet_();
    const rows = readRows_(sh);

    // Ignore Telegram re-deliveries of the same message.
    if (d.message_id && rows.some(r => String(r[8]) === String(d.message_id))) {
      return out_({ ok: true, duplicate: true });
    }

    const now = new Date();
    const when = parseTime_(d.reading_time) || now;
    d.pumps.forEach(p => {
      // Same pump + same photo time already logged (photo re-sent): update it instead of adding.
      const same = d.reading_time ? rows.find(r => r[2] === p.name && timeOf_(r) === when.getTime()) : null;
      const row = [now, when, p.name, Number(p.litres), Number(p.price), '', '', d.sent_by || '', d.message_id || ''];
      if (same) same.splice(0, NCOL, ...row); else rows.push(row);
    });

    const sorted = recalc_(sh, rows);
    const sold = d.pumps.map(p => {
      const r = sorted.find(x => x[2] === p.name && timeOf_(x) === when.getTime());
      return { pump: p.name, litres: r ? r[5] : '', sales: r ? r[6] : '' };
    });
    return out_({ ok: true, rows: d.pumps.length, sold: sold });
  } catch (err) {
    return out_({ ok: false, error: String(err) });
  } finally {
    lock.releaseLock();
  }
}

function out_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
