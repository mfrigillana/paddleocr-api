"""PaddleOCR (PP-OCRv4 via ONNX Runtime) API + Telegram bot, sized for Render's free tier (512 MB RAM)."""
import hashlib
import os
from datetime import datetime, timedelta, timezone
import re
import threading
from contextlib import asynccontextmanager

import cv2
import httpx
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from rapidocr_onnxruntime import RapidOCR

API_KEY = os.getenv("API_KEY", "")
MAX_SIDE = int(os.getenv("MAX_SIDE", "1280"))        # downscale big photos to save RAM
MAX_BYTES = int(os.getenv("MAX_BYTES", str(8 * 1024 * 1024)))

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_ALLOWED = {c.strip() for c in os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "").split(",") if c.strip()}
TG_SECRET = hashlib.sha256(TG_TOKEN.encode()).hexdigest()[:32] if TG_TOKEN else ""
TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"
USE_CLS = os.getenv("USE_CLS", "0") == "1"  # off: the rotate step flips LCD digits (97.50 -> "0516")
PUMP_LABELS = [x.strip() for x in os.getenv("PUMP_LABELS", "Regular,Premium 95,Diesel").split(",")]
SHEET_URL = os.getenv("GSHEET_WEBHOOK_URL", "")  # Apps Script web app /exec URL
PUBLIC_URL = os.getenv("RENDER_EXTERNAL_URL", "")  # set automatically by Render

# One thread each: free tier has 0.1 CPU; more threads only add memory.
engine = RapidOCR(
    intra_op_num_threads=1,
    inter_op_num_threads=1,
    det_limit_side_len=int(os.getenv("DET_LIMIT", "736")),
    det_limit_type="max",   # cap detector input size -> caps peak RAM
    rec_batch_num=1,
)
lock = threading.Lock()  # one OCR job at a time keeps peak RAM predictable


@asynccontextmanager
async def lifespan(_app):
    # Register the Telegram webhook on every boot, so there's nothing to set up by hand.
    if TG_TOKEN and PUBLIC_URL:
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.post(f"{TG_API}/setWebhook", json={
                    "url": f"{PUBLIC_URL}/telegram",
                    "secret_token": TG_SECRET,
                    "allowed_updates": ["message"],
                    "drop_pending_updates": False,
                })
                print("telegram setWebhook:", r.json().get("description"))
        except Exception as e:  # never block the API from starting
            print("telegram setWebhook failed:", e)
    yield


app = FastAPI(title="PaddleOCR API", lifespan=lifespan)


def ocr_image(data: bytes):
    """Returns (lines in working-image coords, full-res image, scale full->working)."""
    full = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if full is None:
        raise ValueError("not an image (jpg/png/webp/bmp)")
    h, w = full.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    img = full if scale == 1.0 else cv2.resize(full, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    with lock:
        result = engine(img, use_cls=USE_CLS)[0]
    del img
    lines = [
        {"text": t, "confidence": round(float(c), 4), "box": [[int(x), int(y)] for x, y in b]}
        for b, t, c in (result or [])
    ]
    return lines, full, scale


def ocr_bytes(data: bytes) -> dict:
    lines, _, _ = ocr_image(data)
    return {"text": "\n".join(l["text"] for l in lines), "lines": lines}


NUM = re.compile(r"^\d{1,7}\.\d{2}$")
PRICE = re.compile(r"^\d{2,3}\.\d{2}$")
STAMP = re.compile(r"(\d{4}/\d{2}/\d{2})\s*(\d{2}:\d{2})")


def _tok(l: dict) -> dict:
    xs = [p[0] for p in l["box"]]
    ys = [p[1] for p in l["box"]]
    return {"t": l["text"].replace(" ", ""), "raw": l["text"], "conf": l["confidence"],
            "x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
            "cx": sum(xs) / 4, "cy": sum(ys) / 4, "h": max(ys) - min(ys), "w": max(xs) - min(xs)}


def _reread(full, tok: dict, scale: float) -> list:
    """Re-read one LCD number from full-resolution crops around its box. Returns every text seen.
    Extra room on the right: a final thin "1" is the digit most often dropped."""
    if full is None:
        return []
    H, W = full.shape[:2]
    texts = []
    for padl, padr, pady in ((0.10, 0.30, 0.25), (0.05, 0.15, 0.15), (0.15, 0.45, 0.35)):
        x0 = int(max(0, (tok["x0"] - padl * tok["w"]) / scale))
        x1 = int(min(W, (tok["x1"] + padr * tok["w"]) / scale))
        y0 = int(max(0, (tok["y0"] - pady * tok["h"]) / scale))
        y1 = int(min(H, (tok["y1"] + pady * tok["h"]) / scale))
        crop = full[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        if crop.shape[0] < 64:
            f = 64 / crop.shape[0]
            crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        solid = cv2.erode(gray, np.ones((3, 3), np.uint8), iterations=1)  # join LCD segments
        for v in (crop, cv2.cvtColor(solid, cv2.COLOR_GRAY2BGR)):
            with lock:
                res = engine(v, use_det=False, use_cls=False)[0]
            if res and res[0][0]:
                texts.append(res[0][0])
    return texts


MAX_PER_DAY = float(os.getenv("MAX_LITRES_PER_DAY", "3000"))
PH_TZ = timezone(timedelta(hours=8))


def stamp_ms(stamp: str | None) -> int:
    if stamp:
        try:
            return int(datetime.strptime(stamp, "%Y/%m/%d %H:%M").replace(tzinfo=PH_TZ).timestamp() * 1000)
        except ValueError:
            pass
    return int(datetime.now(PH_TZ).timestamp() * 1000)


def _candidates(texts: list, first_conf: float, exp_digits: int | None = None) -> list:
    """Possible totalizer values from OCR text. Lower score = more trusted. (value, score, filled_in)"""
    out = {}
    exp_n = exp_digits or 7
    # If any read has every digit, the meter was seen in full: never invent a digit.
    complete = any(len(re.sub(r"\D", "", x)) >= exp_n for x in texts)

    def add(v, sc, filled):
        if v not in out or sc < out[v][0]:
            out[v] = (sc, filled)

    for i, raw in enumerate(texts):
        t = raw.replace(" ", "")
        d = re.sub(r"\D", "", t)
        base = 0.0 if i == 0 and first_conf >= 0.9 else 0.5
        if NUM.match(t):
            add(float(t), base, False)
        full_len = len(d) == exp_digits if exp_digits else len(d) >= 7
        if full_len:  # the tiny decimal point is often missed; totalizers always show 2 decimals
            add(float(d[:-2] + "." + d[-2:]), base + 1, False)
        short = (len(d) == exp_digits - 1) if exp_digits else (len(d) == 6 or re.search(r"\.\d$", t))
        if short and not complete:  # exactly one digit missed, most often a thin "1" at the end
            for pos in range(len(d) + 1):
                for x in "0123456789":
                    nd = d[:pos] + x + d[pos:]
                    add(float(nd[:-2] + "." + nd[-2:]), base + 3 + (0 if x == "1" else 1) + (0 if pos == len(d) else 0.5), True)
    return [(v, sc, f) for v, (sc, f) in out.items()]


def _pick(cands: list, hist: list, t_ms: int, dated: bool = True):
    """Choose the reading that fits the meter history (meters only go up, by a plausible amount)."""
    if not cands:
        return None, "unreadable"
    if hist and not dated:
        # Undated photo: it may be any age, so only rule out the impossible. The sheet estimates its date.
        lo, hi = min(v for _, v in hist) - MAX_PER_DAY * 365, max(v for _, v in hist) + MAX_PER_DAY * 30
        good = sorted((c for c in cands if lo <= c[0] <= hi), key=lambda c: c[1])
        return (good[0], None) if good else (None, "doesn't fit the previous reading")
    if hist:
        before = [v for t, v in hist if t < t_ms]
        after = [v for t, v in hist if t > t_ms]
        prev = max(before) if before else None
        nxt = min(after) if after else None
        t_prev = max((t for t, _ in hist if t < t_ms), default=None)
        t_next = min((t for t, _ in hist if t > t_ms), default=None)
        d_prev = max(1.0, (t_ms - t_prev) / 86400000) if t_prev else 1.0
        d_next = max(1.0, (t_next - t_ms) / 86400000) if t_next else 1.0

        def ok(v):
            return ((prev is None or prev <= v <= prev + MAX_PER_DAY * d_prev)
                    and (nxt is None or nxt - MAX_PER_DAY * d_next <= v <= nxt))

        good = sorted((c for c in cands if ok(c[0])), key=lambda c: c[1])
        if good:
            return good[0], None
        return None, "doesn't fit the previous reading"
    # No history yet: only trust clean reads or a missed decimal point.
    safe = sorted((c for c in cands if c[1] <= 1.5), key=lambda c: c[1])
    return (safe[0], None) if safe else (None, "unclear")


def pump_readings(lines: list, full=None, scale: float = 1.0, width: float | None = None,
                  history: dict | None = None) -> dict | None:
    """Pump panel photo -> {reading_time, pumps:[{name, litres|None, price, raw}]}.
    Prices are the anchor (they read reliably); each pump's litres is the number directly above its price."""
    toks = [_tok(l) for l in lines]
    for t in toks:  # LCD price read without its decimal point, e.g. "9700" -> "97.00"
        if re.fullmatch(r"\d{4,5}", t["t"]) and 2000 <= int(t["t"]) <= 20000:
            t["t"] = t["t"][:-2] + "." + t["t"][-2:]
    prices = [t for t in toks if PRICE.match(t["t"]) and 20 <= float(t["t"]) <= 300]
    if len(prices) < 2:
        return None
    prices.sort(key=lambda t: t["cy"])
    rows, cur = [], [prices[0]]
    for t in prices[1:]:
        if abs(t["cy"] - cur[-1]["cy"]) < 0.6 * max(t["h"], cur[-1]["h"]):
            cur.append(t)
        else:
            rows.append(cur)
            cur = [t]
    rows.append(cur)
    row = max(rows, key=len)
    if len(row) < 2:
        return None
    n = len(PUMP_LABELS)
    if len(row) > n:  # keep the big price screens
        row = sorted(row, key=lambda t: t["h"], reverse=True)[:n]
    row.sort(key=lambda t: t["cx"])

    # Fixed layout: left-to-right is always PUMP_LABELS (Regular, Premium, Diesel).
    # The meter screens (row of big numbers just above the prices) mark the columns. If a price
    # wasn't read, each found price goes to the nearest meter column.
    ph = sum(t["h"] for t in row) / len(row)
    py = sum(t["cy"] for t in row) / len(row)
    meters = [t for t in toks if t not in row and len(re.sub(r"\D", "", t["t"])) >= 6
              and 0 < py - t["cy"] < 3.5 * ph]
    meters.sort(key=lambda t: t["cx"])
    cols = meters if len(meters) == n else None
    if len(row) == n:
        slots = list(range(n))
    elif cols:
        slots = [min(range(n), key=lambda i: abs(cols[i]["cx"] - P["cx"])) for P in row]
        if len(set(slots)) != len(slots):
            slots = list(range(len(row)))
    else:
        if width is None:
            width = (full.shape[1] * scale) if full is not None else max(t["x1"] for t in toks)
        slots = [min(n - 1, int(P["cx"] / (width / n))) for P in row]
        if len(set(slots)) != len(slots):
            slots = list(range(len(row)))

    stamp = None
    for l in lines:
        m = STAMP.search(l["text"].replace(" ", ""))
        if m:
            stamp = f"{m.group(1)} {m.group(2)}"
            break
    t_ms = stamp_ms(stamp)

    def lit_tok(P):
        above = [t for t in toks if t is not P and t["cy"] < P["cy"] and P["cy"] - t["cy"] < 3 * P["h"]
                 and t["x1"] > P["x0"] - P["w"] and t["x0"] < P["x1"] + 0.5 * P["w"]
                 and sum(c.isdigit() for c in t["t"]) >= 4]
        return max(above, key=lambda t: t["cy"]) if above else None

    found = {}
    by_slot = dict(zip(slots, row))
    for slot in range(n):
        P = by_slot.get(slot)
        if P is None and not cols:
            continue
        name = PUMP_LABELS[slot]
        # How many digits this meter should show: from its history, else from other clean reads on the photo.
        h = (history or {}).get(name, [])
        if h:
            exp = len(f"{max(v for _, v in h):.2f}".replace(".", ""))
        else:
            lens = [len(re.sub(r"\D", "", t["t"])) for t in (lit_tok(Q) for Q in row)
                    if t and NUM.match(t["t"]) and t["conf"] >= 0.9]
            exp = max(set(lens), key=lens.count) if lens else None
        L = lit_tok(P) if P is not None else cols[slot]
        choice, why = None, "not found"
        if L is not None:
            texts = [L["raw"]]
            if not (NUM.match(L["t"]) and L["conf"] >= 0.9):
                texts += _reread(full, L, scale)
            choice, why = _pick(_candidates(texts, L["conf"], exp), (history or {}).get(name, []), t_ms, bool(stamp))
            if choice is None and len(texts) == 1:  # clean read rejected by history: look again
                texts += _reread(full, L, scale)
                choice, why = _pick(_candidates(texts, L["conf"], exp), (history or {}).get(name, []), t_ms, bool(stamp))
        found[slot] = {"name": name, "litres": choice[0] if choice else None,
                       "filled": bool(choice and choice[2]), "why": why,
                       "price": float(P["t"]) if P is not None else None, "raw": L["raw"] if L else ""}
    pumps = [found.get(i, {"name": PUMP_LABELS[i], "litres": None, "price": None, "raw": ""})
             for i in range(n)]
    return {"reading_time": stamp, "pumps": pumps}


MANUAL = re.compile(r"^\s*(.+?)\s+([\d,]+\.\d{1,2})\s+(\d{2,3}(?:\.\d{1,2})?)\s*$")


def manual_readings(text: str) -> dict | None:
    """Typed correction, one pump per line: '<pump> <litres> <price>', e.g. 'Regular 65815.01 97.50'."""
    names = {n.lower(): n for n in PUMP_LABELS}
    pumps, stamp = [], None
    for line in text.strip().splitlines():
        m = STAMP.fullmatch(line.strip().replace(" ", "")) if not pumps and stamp is None else None
        if m:  # optional first line: 2026/10/07 18:02
            stamp = f"{m.group(1)} {m.group(2)}"
            continue
        m = MANUAL.match(line)
        if not m or m.group(1).strip().lower() not in names:
            return None
        pumps.append({"name": names[m.group(1).strip().lower()],
                      "litres": float(m.group(2).replace(",", "")), "price": float(m.group(3))})
    return {"reading_time": stamp, "pumps": pumps} if pumps else None


def format_readings(r: dict) -> str:
    if all(p["litres"] is None and not p.get("raw") for p in r["pumps"]):
        prices = ", ".join(f"{p['name']} ₱{p['price']:.2f}" for p in r["pumps"] if p["price"])
        return (f"I can see the prices ({prices}) but not the meter totals, so nothing was saved.\n"
                "Show the totalizer (L) screen on each pump, then take the photo again.")
    out = [f"Reading: {r['reading_time']}"] if r["reading_time"] else []
    bad = []
    for p in r["pumps"]:
        if p["litres"] is None:
            if p.get("why") == "doesn't fit the previous reading":
                out.append(f"{p['name']}: read \"{p['raw']}\" but it doesn't fit the readings in the sheet")
            else:
                out.append(f"{p['name']}: couldn't read the meter" + (f" (saw \"{p['raw']}\")" if p.get("raw") else ""))
            bad.append(p)
        elif p["price"] is None:
            out.append(f"{p['name']}: {p['litres']:,.2f} L (couldn't read the price)")
            bad.append(p)
        else:
            out.append(f"{p['name']}: {p['litres']:,.2f} L{'*' if p.get('filled') else ''} @ ₱{p['price']:.2f}/L")
    if any(p.get("filled") for p in r["pumps"]):
        out.append("* one faint digit was filled in to match the meter history")
    if bad:
        out.append("\nNot saved for the pump(s) above. Retake the photo, or reply with:")
        for p in bad:
            lit = f"{p['litres']:.2f}" if p["litres"] is not None else "<litres>"
            pr = f"{p['price']:.2f}" if p["price"] is not None else "<price>"
            out.append(f"{p['name']} {lit} {pr}")
    return "\n".join(out)


async def sheet_history() -> dict:
    """Per-pump [[time ms, totalizer], ...] from the sheet; {} if unavailable."""
    if not SHEET_URL:
        return {}
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
            return (await c.get(SHEET_URL)).json().get("history", {})
    except Exception as e:
        print("sheet history failed:", repr(e))
        return {}


async def log_to_sheet(r: dict, msg: dict) -> str:
    """POST readings to the Google Apps Script web app. Returns a short status line for the reply."""
    if not SHEET_URL:
        return ""
    sender = msg.get("from") or {}
    r = {**r, "pumps": [p for p in r["pumps"] if p["litres"] is not None and p["price"] is not None]}
    if not r["pumps"]:
        return ""
    payload = {**r, "message_id": f"tg{msg['chat']['id']}_{msg.get('message_id')}",
               "sent_by": sender.get("username") or sender.get("first_name") or ""}
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
            res = (await c.post(SHEET_URL, json=payload)).json()
    except Exception as e:
        print("sheet log failed:", repr(e))
        return "\n\nNot saved to sheet (connection error)."
    if not res.get("ok"):
        print("sheet log error:", res)
        return "\n\nNot saved to sheet."
    if res.get("duplicate"):
        return "\n\nAlready in sheet."
    lines = ["\n\nSaved to sheet."]
    if res.get("source") == "estimated":
        t = next((x.get("time") for x in res.get("sold", []) if x.get("time")), "")
        lines.append(f"No date stamp on the photo, so the date was estimated from the meters: {t}")
    for s in res.get("sold", []):
        if s.get("litres") == "":
            continue
        if s["litres"] < 0:
            lines.append(f"{s['pump']}: lower than the previous reading, please check")
        else:
            lines.append(f"{s['pump']} since last: {s['litres']:,.2f} L = ₱{s['sales']:,.2f}")
    return "\n".join(lines)


@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "invalid api key")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    try:
        return await run_in_threadpool(ocr_bytes, data)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------------- Telegram ----------------

HELP = ("Send me a photo of the pump meters and I'll read and log them.\n"
        "To enter a reading by hand: Regular 65815.01 97.50 (one pump per line).\n"
        "For a past reading, put the date first: 2026/10/07 18:02 on its own line.\n"
        "Tip: send it as a File (not compressed) for small print.")


async def tg(method: str, **payload):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{TG_API}/{method}", json=payload)
        return r.json()


async def tg_download(file_id: str) -> bytes:
    info = await tg("getFile", file_id=file_id)
    path = info["result"]["file_path"]
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(f"https://api.telegram.org/file/bot{TG_TOKEN}/{path}")
        r.raise_for_status()
        return r.content


async def tg_reply(chat_id, text: str, reply_to=None):
    for i in range(0, max(len(text), 1), 4000):  # Telegram limit is 4096 chars
        await tg("sendMessage", chat_id=chat_id, text=text[i:i + 4000] or "-",
                 reply_to_message_id=reply_to if i == 0 else None)


async def handle_message(msg: dict):
    chat_id = msg["chat"]["id"]
    mid = msg.get("message_id")

    if TG_ALLOWED and str(chat_id) not in TG_ALLOWED:
        await tg_reply(chat_id, f"Not authorised. Your chat ID is {chat_id}.")
        return

    file_id = None
    if msg.get("photo"):
        file_id = msg["photo"][-1]["file_id"]  # largest size Telegram kept
    elif (msg.get("document") or {}).get("mime_type", "").startswith("image/"):
        file_id = msg["document"]["file_id"]
        if msg["document"].get("file_size", 0) > MAX_BYTES:
            await tg_reply(chat_id, "That file is too big (8 MB max).", mid)
            return
    elif (msg.get("document") or {}).get("mime_type") == "application/pdf":
        await tg_reply(chat_id, "PDFs aren't supported yet. Please send a photo or screenshot of the page.", mid)
        return

    if not file_id:
        r = manual_readings(msg.get("text") or "")
        if r:
            r["typed"] = True
            await tg_reply(chat_id, format_readings(r) + await log_to_sheet(r, msg), mid)
        else:
            await tg_reply(chat_id, f"{HELP}\n\nYour chat ID: {chat_id}")
        return

    try:
        await tg("sendChatAction", chat_id=chat_id, action="typing")
        data = await tg_download(file_id)
        history = await sheet_history()

        def work():
            lines, full, scale = ocr_image(data)
            return lines, pump_readings(lines, full, scale, history=history)

        lines, r = await run_in_threadpool(work)
        if r:
            reply = format_readings(r) + await log_to_sheet(r, msg)
        else:
            reply = "\n".join(l["text"] for l in lines) or "No text found."
        await tg_reply(chat_id, reply, mid)
    except Exception as e:
        print("telegram OCR error:", repr(e))
        await tg_reply(chat_id, "Sorry, I couldn't read that image. Try a clearer photo.", mid)


@app.post("/telegram")
async def telegram_webhook(request: Request, background: BackgroundTasks,
                           x_telegram_bot_api_secret_token: str = Header(default="")):
    if not TG_TOKEN or x_telegram_bot_api_secret_token != TG_SECRET:
        raise HTTPException(403)
    update = await request.json()
    msg = update.get("message")
    if msg:
        # Answer Telegram right away; OCR runs after, so slow pages don't cause re-deliveries.
        background.add_task(handle_message, msg)
    return {"ok": True}
