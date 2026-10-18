// MJRV pump readings logger. Receives JSON from the Telegram OCR bot and logs one row per pump.
// Rows are kept in reading-time order, and Litres Sold / Sales are recalculated per pump after
// every change, so photos sent late or out of order still give correct figures.
// Photos without a camera date get an estimated Reading Time: meters only go up, so the
// totalizer shows which dated readings the photo sits between.
// Deploy: Deploy > New deployment > Web app, Execute as: Me, Who has access: Anyone.

const SHEET_ID = '1J9vXhhtMgiggSi2rl8kSBRXSEIceUpTiPIYGCxxyehY';  // MJRV Pump Readings
const SHEET_NAME = 'Readings';
const HEADERS = ['Logged At', 'Reading Time', 'Pump', 'Totalizer (L)', 'Price (PHP/L)',
                 'Litres Sold', 'Sales (PHP)', 'Sent By', 'Telegram Msg ID', 'Time Source'];
const NCOL = HEADERS.length;
const DAY = 86400000;
const DEFAULT_RATE = 200;  // litres/day, only used when a pump has a single dated reading

function getSheet_() {
  const ss = SpreadsheetApp.openById(SHEET_ID);
  let sh = ss.getSheetByName(SHEET_NAME) || ss.getSheets()[0];
  if (sh.getName() !== SHEET_NAME) sh.setName(SHEET_NAME);
  if (sh.getLastRow() === 0 || sh.getRange(1, 1).getValue() !== HEADERS[0] || sh.getRange(1, NCOL).getValue() !== HEADERS[NCOL - 1]) {
    sh.getRange(1, 1, 1, NCOL).setValues([HEADERS]);
  }
  sh.getRange(1, 1, 1, NCOL).setFontWeight('bold');
  sh.setFrozenRows(1);
  return sh;
}

// Run from the editor to set formats and tidy existing rows.
function setup() {
  SpreadsheetApp.openById(SHEET_ID).setSpreadsheetTimeZone('Asia/Manila');
  const sh = getSheet_();
  sh.getRange('A:B').setNumberFormat('yyyy-mm-dd hh:mm');
  sh.getRange('D:G').setNumberFormat('#,##0.00');
  recalc_(sh);
}

// "2026/10/07 18:02" (camera stamp, Philippine time UTC+8) -> Date.
function parseTime_(v) {
  if (v instanceof Date) return v;
  const m = String(v || '').match(/(\d{4})[\/-](\d{2})[\/-](\d{2})\s*(\d{2}):(\d{2})/);
  return m ? new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4] - 8, +m[5])) : null;
}

function readRows_(sh) {
  const last = sh.getLastRow();
  return last > 1 ? sh.getRange(2, 1, last - 1, NCOL).getValues() : [];
}

function timeOf_(r) {
  return (parseTime_(r[1]) || parseTime_(r[0]) || new Date(0)).getTime();
}

// Estimate when a reading was taken from where its totalizer falls among dated readings.
function estimateTime_(v, dated, fallback) {
  if (!dated.length) return fallback;
  const n = dated.length;
  for (let i = 0; i < n - 1; i++) {
    const [t0, v0] = dated[i], [t1, v1] = dated[i + 1];
    if (v >= v0 && v <= v1) return v1 === v0 ? t0 : t0 + (v - v0) / (v1 - v0) * (t1 - t0);
  }
  const rate = n > 1 && dated[n - 1][0] > dated[0][0]
    ? Math.max(1, (dated[n - 1][1] - dated[0][1]) / ((dated[n - 1][0] - dated[0][0]) / DAY))
    : DEFAULT_RATE;
  if (v < dated[0][1]) return dated[0][0] - (dated[0][1] - v) / rate * DAY;
  return Math.min(fallback, dated[n - 1][0] + (v - dated[n - 1][1]) / rate * DAY);
}

// Tidy and recompute everything: drop exact repeats, estimate undated times, sort, recompute sold/sales.
function recalc_(sh, rows) {
  rows = rows || readRows_(sh);
  const before = sh.getLastRow() - 1;

  rows.forEach(r => {
    r[1] = parseTime_(r[1]) || r[1];
    if (!r[9] || !parseTime_(r[1])) {  // older rows: no reading time, or one within 2 minutes of logging, means there was no camera date
      const a = parseTime_(r[0]), b = parseTime_(r[1]);
      r[9] = !b || (a && Math.abs(a - b) < 120000) ? 'estimated' : 'photo';
    }
  });

  // Same pump + same totalizer = the same reading sent twice: keep the first.
  const seen = {};
  rows = rows.filter(r => {
    const k = r[2] + '|' + Number(r[3]).toFixed(2);
    if (seen[k]) return false;
    seen[k] = true;
    return true;
  });

  // Estimate times for undated rows from each pump's dated readings.
  const dated = {};
  rows.filter(r => r[9] !== 'estimated').forEach(r => {
    (dated[r[2]] = dated[r[2]] || []).push([timeOf_(r), Number(r[3])]);
  });
  Object.values(dated).forEach(d => d.sort((a, b) => a[0] - b[0]));
  const est = rows.filter(r => r[9] === 'estimated');
  est.forEach(r => { r._t = estimateTime_(Number(r[3]), dated[r[2]] || [], parseTime_(r[0]).getTime()); });
  // Pumps in one photo were read at the same moment: use their median estimate.
  const byMsg = {};
  est.forEach((r, i) => { const k = r[8] || 'row' + i; (byMsg[k] = byMsg[k] || []).push(r); });
  Object.values(byMsg).forEach(g => {
    const ts = g.map(r => r._t).sort((a, b) => a - b);
    const t = ts[Math.floor(ts.length / 2)];
    g.forEach(r => { r[1] = new Date(Math.round(t / 60000) * 60000); delete r._t; });
  });

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
  if (before > rows.length) sh.getRange(rows.length + 2, 1, before - rows.length, NCOL).clearContent();
  return rows;
}

// GET: per-pump history [[time ms, totalizer], ...] (dated readings only) for the bot's sanity checks.
function doGet() {
  const rows = readRows_(getSheet_());
  const hist = {};
  rows.forEach(r => {
    if (r[9] === 'estimated') return;
    (hist[r[2]] = hist[r[2]] || []).push([timeOf_(r), Number(r[3])]);
  });
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
    const stamped = parseTime_(d.reading_time);
    const source = stamped ? (d.typed ? 'typed' : 'photo') : (d.typed ? 'typed' : 'estimated');
    const when = stamped || now;
    d.pumps.forEach(p => {
      // Same pump + same photo time (photo re-sent): update instead of adding.
      const same = stamped ? rows.find(r => r[2] === p.name && timeOf_(r) === when.getTime()) : null;
      const row = [now, when, p.name, Number(p.litres), Number(p.price), '', '', d.sent_by || '', d.message_id || '', source];
      if (same) same.splice(0, NCOL, ...row); else rows.push(row);
    });

    const sorted = recalc_(sh, rows);
    const sold = d.pumps.map(p => {
      const r = sorted.find(x => x[2] === p.name && Number(x[3]).toFixed(2) === Number(p.litres).toFixed(2));
      return { pump: p.name, litres: r ? r[5] : '', sales: r ? r[6] : '',
               time: r ? Utilities.formatDate(r[1], 'Asia/Manila', 'yyyy/MM/dd HH:mm') : '' };
    });
    return out_({ ok: true, rows: d.pumps.length, sold: sold, source: source });
  } catch (err) {
    return out_({ ok: false, error: String(err) });
  } finally {
    lock.releaseLock();
  }
}

function out_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
